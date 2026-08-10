from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

__all__ = [
    "SlotRequirement",
    "DatatypeRequirement",
    "NDimsRequirement",
    "ShapeMatchRequirement",
    "AxisLengthRequirement",
]


class SlotRequirement(ABC):
    """Abstract base class for slot requirements."""

    def validate(self, name: str, arr: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
        """Validate the slot requirement."""
        self._validate(name, arr, arrays)

    @abstractmethod
    def _validate(self, name: str, arr: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
        pass


class DatatypeRequirement(SlotRequirement):
    """Requires a specific dtype."""

    def __init__(self, dtype: np.dtype | Sequence[np.dtype]) -> None:
        self.dtype = dtype

    def _validate(self, name, arr, arrays) -> None:
        dtypes = self.dtype if isinstance(self.dtype, Sequence) else [self.dtype]
        if not any(np.issubdtype(arr.dtype, dt) for dt in dtypes):
            raise ValueError(f"Slot '{name}' expected dtype in {dtypes}, got {arr.dtype}")


class NDimsRequirement(SlotRequirement):
    """Requires a specific number of dimensions."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    def _validate(self, name, arr, arrays) -> None:
        if arr.ndim != self.ndim:
            raise ValueError(f"Slot '{name}' expected ndim={self.ndim}, got {arr.ndim}")


class ShapeMatchRequirement(SlotRequirement):
    """Requires the shape of the array to match the shape of another array in a specific axis."""

    def __init__(self, other_slot: str, self_axis: int, other_axis: int) -> None:
        self.other_slot = other_slot
        self.self_axis = self_axis
        self.other_axis = other_axis

    def _validate(self, name, arr, arrays) -> None:
        other = self.other_slot
        if arr.shape[self.self_axis] != arrays[other].shape[self.other_axis]:
            raise ValueError(
                f"Slot '{name}' dim {self.self_axis} ({arr.shape[self.self_axis]}) "
                f"must match '{other}' dim {self.other_axis} ({arrays[other].shape[self.other_axis]})"
            )


class AxisLengthRequirement(SlotRequirement):
    """Requires the length of an array axis to have a specific length."""

    def __init__(self, *, axis: int, length: int) -> None:
        self.axis = axis
        self.length = length

    def _validate(self, name, arr, arrays) -> None:
        if arr.shape[self.axis] != self.length:
            raise ValueError(
                f"Slot '{name}' axis {self.axis} must have length {self.length}, got {arr.shape[self.axis]}"
            )
