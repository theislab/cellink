from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from cellink.at.require import SlotRequirement
from cellink.at.resolver import get_model_matrix

__all__ = ["BaseModel", "fetch_raw_slot", "to_numpy"]


def to_numpy(raw) -> np.ndarray:
    """Convert a resolved slot (ndarray, Series, or DataFrame) to a plain numpy array."""
    if isinstance(raw, pd.Series | pd.DataFrame):
        return raw.to_numpy()
    if isinstance(raw, np.ndarray):
        return raw
    return np.asarray(raw)


def fetch_raw_slot(
    data,
    obj,
    slot: str,
    *,
    target_level: Literal["donor", "cell"] | None = None,
    add_intercept: bool = False,
):
    """Resolve one slot's value against ``data``, returning a DataFrame with a dummy-index flag.

    ``obj`` is either already array-like (``np.ndarray``/``pd.Series``/``pd.DataFrame``,
    used as-is) or a formula/column-name string, resolved against ``data`` via
    :func:`cellink.at.resolver.get_model_matrix`. ``add_intercept=True`` keeps an
    intercept column for string formulas: pass ``True`` for covariate-like slots
    such as ``F``, ``False`` for outcome-like slots such as ``Y``, so a bare
    column name doesn't spuriously pick up an intercept column.
    """

    def _get_df(p):
        if isinstance(p, pd.DataFrame):
            df, is_dummy = p, False
        elif isinstance(p, pd.Series):
            df, is_dummy = p.to_frame(), False
        else:
            df, is_dummy = pd.DataFrame(p), True
        df.attrs["has_dummy_index"] = is_dummy
        return df

    if isinstance(obj, np.ndarray | pd.Series | pd.DataFrame):
        return _get_df(obj)

    if isinstance(obj, str):
        if data is None:
            raise ValueError("Mandatory to provide `data` to use formula/column-name syntax.")
        formula = obj if add_intercept else f"{obj} - 1"
        df = get_model_matrix(data, formula, target_level=target_level)
        df.attrs["has_dummy_index"] = False
        return df

    raise TypeError(f"Cannot fetch slot '{slot}' from {obj!r} (expected ndarray, Series, DataFrame, or str)")


class BaseModel(ABC):
    """Base class for cellink.at models that fetch every required slot in one ``run()`` call."""

    required_slots: list[str] = []
    slot_requirements: dict[str, Sequence[SlotRequirement]] = {}
    add_intercept_slots: list[str] = []

    def __init__(self, *, data=None, target_level: Literal["donor", "cell"] | None = None, **specs) -> None:
        self.data = data
        self._specs = specs
        self.target_level = target_level

        self._post_init()

    @abstractmethod
    def _post_init(self) -> None:
        pass

    def _fetch_raw(self, obj, slot):
        return fetch_raw_slot(
            self.data,
            obj,
            slot,
            target_level=self.target_level,
            add_intercept=slot in self.add_intercept_slots,
        )

    def run(self, **overrides):
        """Fetch data, validate, and run the method."""
        specs = {**self._specs, **overrides}
        missing = [s for s in self.required_slots if s not in specs]
        if missing:
            raise ValueError(f"Missing required slots: {missing}")

        raws = {s: self._fetch_raw(specs[s], s) for s in self.required_slots}
        indices = {k: df.index for k, df in raws.items() if not df.attrs["has_dummy_index"]}
        if indices:
            first_key, first_idx = next(iter(indices.items()))
            for key, idx in indices.items():
                if not first_idx.equals(idx):
                    raise ValueError(f"Mismatched indices between '{first_key}' and '{key}': {first_idx} vs {idx}")
        arrays = {s: to_numpy(raws[s]) for s in self.required_slots}
        for name, arr in arrays.items():
            for req in self.slot_requirements.get(name, []):
                req.validate(name, arr, arrays)
        return self._run(**arrays)

    @abstractmethod
    def _run(self, **arrays):
        pass
