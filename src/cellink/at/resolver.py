from __future__ import annotations

import re
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from cellink._core import HiData
from cellink.at.fetch import fetch_slot

__all__ = ["get_model_matrix"]


def _find_custom_transforms(text: str, func_name: str) -> list[dict[str, str]]:
    """Find all calls to a specific function (e.g. ``group_mean``) in a string, handling nested parens."""
    calls = []
    for match in re.finditer(rf"(?<![\w.]){re.escape(func_name)}\(", text):
        start_index = match.end()
        paren_level = 1
        i = start_index
        while i < len(text) and paren_level > 0:
            if text[i] == "(":
                paren_level += 1
            elif text[i] == ")":
                paren_level -= 1
            i += 1

        if paren_level == 0:
            inner_content = text[start_index : i - 1]
            full_expression = f"{func_name}({inner_content})"
            calls.append({"full": full_expression, "inner": inner_content})

    return calls


def get_model_matrix(
    data: HiData | ad.AnnData | pd.DataFrame,
    formula_str: str,
    target_level: Literal["group", "cell"] | None = None,
) -> pd.DataFrame:
    """Resolve a formula and build a model matrix from a HiData, AnnData, or DataFrame object.

    Parameters
    ----------
    data : HiData | AnnData | pandas.DataFrame
        The input data container.
    formula_str : str
        The formulaic formula string. Group/cell-aggregation functions
        (``group_mean``, ``group_max``, ``group_median``, ``group_first``, ``crepeat``) are only
        valid when ``data`` is a :class:`~cellink._core.HiData`.
    target_level : "group" | "cell" | None, optional
        The target data level. Required only when ``data`` is a HiData.

    Returns
    -------
    pandas.DataFrame
        The model matrix.
    """
    try:
        from formulaic import Formula
        from formulaic.parser import DefaultFormulaParser
    except ImportError as e:
        raise ImportError(
            "Resolving a formula/column-name string requires `formulaic`. Install it with:\n\n    pip install 'cellink-tools[at]'"
        ) from e

    from cellink.at.formula import FamilyOperatorResolver

    GROUP_AGG_FUNCS = {
        "group_mean": "mean",
        "group_max": "max",
        "group_median": "median",
        "group_first": "first",
    }
    CUSTOM_TRANSFORMS = {**{k: lambda x: x for k in GROUP_AGG_FUNCS}, "crepeat": lambda x: x}
    parser = DefaultFormulaParser(operator_resolver=FamilyOperatorResolver())

    if isinstance(data, HiData):
        hd = data
        if target_level is None:
            raise ValueError("`target_level` must be 'group' or 'cell' when using a HiData object.")

        if target_level == "group" and "crepeat(" in formula_str:
            raise ValueError("Cannot use 'crepeat()' in a group-level formula.")
        if target_level == "cell" and any(f"{fname}(" in formula_str for fname in GROUP_AGG_FUNCS):
            raise ValueError("Group aggregation functions (e.g., group_mean) cannot be used in a cell-level formula.")

        formula = Formula(formula_str, _parser=parser)
        required_vars = formula.required_variables - set(CUSTOM_TRANSFORMS.keys())
        fetched_data = {k: fetch_slot(hd, k) for k in required_vars}

        cell_vars = {k: v for k, v in fetched_data.items() if len(v) == hd.C.n_obs}
        group_vars = {k: v for k, v in fetched_data.items() if len(v) == hd.n_groups}
        # A formula referencing only group-level (or only cell-level) variables is
        # the common case, not an edge case: guard against pd.concat's "No
        # objects to concatenate" on an empty dict rather than crash on it.
        cell_df = pd.concat(cell_vars.values(), axis=1) if cell_vars else pd.DataFrame(index=hd.C.obs_names)
        group_df = pd.concat(group_vars.values(), axis=1) if group_vars else pd.DataFrame(index=hd.obs_names)

        rewritten_formula = formula_str
        if target_level == "group":
            final_df = pd.DataFrame(index=hd.obs_names)
            for agg_func_name, pd_method_name in GROUP_AGG_FUNCS.items():
                for call in _find_custom_transforms(rewritten_formula, agg_func_name):
                    original_call_str, inner_formula = call["full"], call["inner"]
                    inner_matrix = Formula(f"{inner_formula} - 1").get_model_matrix(cell_df)
                    new_quoted_names = []
                    for col in inner_matrix.columns:
                        new_col_name = f"{agg_func_name}({col})"
                        new_quoted_names.append(f"`{new_col_name}`")
                        if new_col_name not in final_df:
                            agg_group = inner_matrix[col].groupby(hd.C.obs[hd.group_id], observed=True)
                            aggregated_series = getattr(agg_group, pd_method_name)()
                            final_df[new_col_name] = aggregated_series
                    replacement_str = " + ".join(new_quoted_names)
                    rewritten_formula = rewritten_formula.replace(original_call_str, f"({replacement_str})")
        else:  # cell
            final_df = pd.DataFrame(index=hd.C.obs_names)
            for call in _find_custom_transforms(rewritten_formula, "crepeat"):
                original_call_str, inner_formula = call["full"], call["inner"]
                inner_matrix = Formula(f"{inner_formula} - 1").get_model_matrix(group_df)
                new_quoted_names = []
                for col in inner_matrix.columns:
                    new_col_name = f"crepeat({col})"
                    new_quoted_names.append(f"`{new_col_name}`")
                    if new_col_name not in final_df:
                        # `.map()` on a Categorical group_id column (the common case,
                        # AnnData/scanpy obs columns are routinely Categorical) inherits
                        # Categorical dtype on the result, which formulaic then one-hot
                        # encodes instead of treating as the numeric column it actually
                        # is. Force plain values through to avoid that.
                        group_ids = hd.C.obs[hd.group_id]
                        if isinstance(group_ids.dtype, pd.CategoricalDtype):
                            group_ids = group_ids.astype(group_ids.cat.categories.dtype)
                        expanded_series = group_ids.map(inner_matrix[col])
                        final_df[new_col_name] = np.asarray(expanded_series.to_numpy(), dtype=inner_matrix[col].to_numpy().dtype)
                replacement_str = " + ".join(new_quoted_names)
                rewritten_formula = rewritten_formula.replace(original_call_str, f"({replacement_str})")

        final_df = final_df.join(group_df if target_level == "group" else cell_df)

    elif isinstance(data, ad.AnnData):
        if any(f in formula_str for f in ("group_mean(", "crepeat(")):
            raise ValueError(
                "Multi-level functions ('group_mean', 'crepeat', etc.) are not applicable for a single AnnData input."
            )
        formula = Formula(formula_str, _parser=parser)
        required_vars = formula.required_variables
        fetched_data = {k: fetch_slot(data, k) for k in required_vars}
        final_df = pd.concat(fetched_data.values(), axis=1)
        rewritten_formula = formula_str

    elif isinstance(data, pd.DataFrame):
        if any(f in formula_str for f in ("group_mean(", "crepeat(")):
            raise ValueError(
                "Multi-level functions ('group_mean', 'crepeat', etc.) are not applicable for a pandas DataFrame input."
            )
        final_df = data
        rewritten_formula = formula_str

    else:
        raise TypeError(f"Input data must be a HiData, AnnData, or DataFrame object, not {type(data)}")

    return Formula(rewritten_formula, _parser=parser).get_model_matrix(final_df)
