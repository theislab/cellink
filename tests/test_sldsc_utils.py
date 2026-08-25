from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external._sldsc_utils import (
    _compute_celltype_means,
    _compute_specificity,
    _normalize_chromosome,
    _pick_var_col,
    _safe_filename,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("chr1", "1"),
        ("CHR1", "1"),
        ("1", "1"),
        ("chrX", "X"),
        ("chrY", "Y"),
        ("chrM", "M"),
        ("22", "22"),
    ],
)
def test_normalize_chromosome_single(raw, expected):
    result = _normalize_chromosome(pd.Series([raw]))
    assert result.iloc[0] == expected


def test_normalize_chromosome_mixed_series():
    result = _normalize_chromosome(pd.Series(["chr1", "CHR2", "chrX", "10"]))
    assert list(result) == ["1", "2", "X", "10"]


def test_pick_var_col_prefers_default_when_present():
    adata = AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["g1", "g2"], data={"symbol": ["A", "B"], "gene_name": ["A", "B"]}))
    assert _pick_var_col(adata, candidates=["gene_name"], default="symbol") == "symbol"


def test_pick_var_col_falls_back_to_candidates():
    adata = AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["g1", "g2"], data={"gene_name": ["A", "B"]}))
    assert _pick_var_col(adata, candidates=["gene_name", "symbol"], default="missing_col") == "gene_name"


def test_pick_var_col_returns_default_when_nothing_found():
    adata = AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["g1", "g2"]))
    assert _pick_var_col(adata, candidates=["gene_name"], default="fallback") == "fallback"


def test_compute_celltype_means_basic():
    X = np.array([[1.0, 3.0], [3.0, 1.0], [10.0, 20.0]])
    adata = AnnData(X, obs=pd.DataFrame({"celltype": ["A", "A", "B"]}), var=pd.DataFrame(index=["g1", "g2"]))
    means = _compute_celltype_means(adata, "celltype")
    assert list(means.columns) == ["A", "B"]
    np.testing.assert_allclose(means.loc["g1", "A"], 2.0)
    np.testing.assert_allclose(means.loc["g2", "A"], 2.0)
    np.testing.assert_allclose(means.loc["g1", "B"], 10.0)


def test_compute_celltype_means_empty_celltype_is_nan_not_zero():
    """Regression check: a cell type with 0 cells (e.g. after upstream filtering
    drops every cell of that type) must be reported as NaN, not silently
    scored as 0 expression, which would corrupt the specificity computation."""
    X = np.array([[1.0], [3.0]])
    adata = AnnData(
        X,
        obs=pd.DataFrame({"celltype": pd.Categorical(["A", "A"], categories=["A", "B"])}),
        var=pd.DataFrame(index=["g1"]),
    )
    means = _compute_celltype_means(adata, "celltype")
    assert means.loc["g1", "A"] == 2.0
    assert np.isnan(means.loc["g1", "B"])


def test_compute_specificity_sums_to_one_across_celltypes():
    mean_expr = pd.DataFrame({"A": [2.0, 0.0], "B": [8.0, 0.0]}, index=["g1", "g2"])
    spec = _compute_specificity(mean_expr)
    np.testing.assert_allclose(spec.loc["g1", "A"], 0.2)
    np.testing.assert_allclose(spec.loc["g1", "B"], 0.8)
    np.testing.assert_allclose(spec.loc["g1"].sum(), 1.0)


def test_compute_specificity_zero_total_expression_is_zero_not_nan():
    """A gene expressed nowhere has no well-defined specificity; the
    implementation must return 0, not propagate NaN into downstream LD-score
    annotation files (which would silently corrupt them)."""
    mean_expr = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=["g1"])
    spec = _compute_specificity(mean_expr)
    assert spec.loc["g1", "A"] == 0.0
    assert spec.loc["g1", "B"] == 0.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CD4+ T cell", "CD4+_T_cell"),
        ("Natural Killer (NK)", "Natural_Killer__NK_"),
        ("  leading/trailing  ", "leading_trailing"),
        ("already_safe-1.0", "already_safe-1.0"),
    ],
)
def test_safe_filename(raw, expected):
    assert _safe_filename(raw) == expected
