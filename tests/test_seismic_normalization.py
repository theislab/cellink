from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from cellink.tl.external import _seismic
from cellink.tl.external._preprocessing import _log_normalize_from_values
from cellink.tl.external._seismic import run_seismic


def _counts_adata(n_cells=120, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B"], size=n_cells)}, index=[f"c{i}" for i in range(n_cells)])
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=[str(1000 + i) for i in range(n_genes)]))


def _lognorm(adata):
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def test_counts_with_stale_log1p_flag_are_log_normalized():
    adata = _counts_adata()
    expected = _lognorm(adata).X.toarray()
    adata.uns["log1p"] = {"base": None}

    _log_normalize_from_values(adata, "seismic")

    np.testing.assert_allclose(adata.X.toarray(), expected, rtol=1e-5)
    assert "log1p" in adata.uns


def test_counts_without_flag_are_log_normalized():
    adata = _counts_adata()
    expected = _lognorm(adata).X.toarray()

    _log_normalize_from_values(adata, "seismic")

    np.testing.assert_allclose(adata.X.toarray(), expected, rtol=1e-5)


def test_log_normalized_without_flag_is_not_normalized_again():
    adata = _lognorm(_counts_adata())
    del adata.uns["log1p"]
    before = adata.X.toarray().copy()

    _log_normalize_from_values(adata, "seismic")

    np.testing.assert_array_equal(adata.X.toarray(), before)


def test_log_normalized_dense_with_flag_is_left_unchanged():
    adata = _lognorm(_counts_adata())
    adata.X = adata.X.toarray()
    before = adata.X.copy()

    _log_normalize_from_values(adata, "seismic")

    np.testing.assert_array_equal(adata.X, before)


@pytest.mark.parametrize(
    "make_adata, flag, expected",
    [
        (lambda: _counts_adata(), False, ["integer counts. Normalizing"]),
        (lambda: _counts_adata(), True, ["is set, but the expression values are integer counts"]),
        (lambda: _lognorm(_counts_adata()), True, ["Using them as log-normalized"]),
        (lambda: _lognorm(_counts_adata()), False, ["not integer counts and adata.uns['log1p'] is not set"]),
    ],
)
def test_normalization_decision_is_logged_as_a_warning(make_adata, flag, expected, caplog):
    adata = make_adata()
    adata.uns.pop("log1p", None)
    if flag:
        adata.uns["log1p"] = {"base": None}

    with caplog.at_level(logging.WARNING):
        _log_normalize_from_values(adata, "seismic")

    assert len(caplog.records) == 1
    assert all(text in caplog.text for text in expected)


def test_scaled_and_unlogged_values_are_flagged(caplog):
    scaled = _lognorm(_counts_adata())
    scaled.X = scaled.X.toarray()
    sc.pp.scale(scaled)
    unlogged = _counts_adata()
    sc.pp.normalize_total(unlogged, target_sum=1e4)

    with caplog.at_level(logging.WARNING):
        _log_normalize_from_values(scaled, "seismic")
        _log_normalize_from_values(unlogged, "seismic")

    assert "negative numbers" in caplog.text
    assert "may be normalized but not log-transformed" in caplog.text


def test_run_seismic_exports_log_normalized_values_for_counts_with_stale_flag(tmp_path, monkeypatch):
    adata = _counts_adata(n_genes=300)
    adata.uns["log1p"] = {"base": None}
    magma_file = tmp_path / "magma.genes.out"
    pd.DataFrame({"GENE": adata.var_names, "ZSTAT": np.random.default_rng(1).normal(size=adata.n_vars)}).to_csv(
        magma_file, sep="\t", index=False
    )

    class _StopBeforeR(Exception):
        pass

    def _stop(*args, **kwargs):
        raise _StopBeforeR

    monkeypatch.setattr(_seismic.subprocess, "run", _stop)
    prefix = str(tmp_path / "seismic")
    with pytest.raises(_StopBeforeR):
        run_seismic(
            adata, magma_file=str(magma_file), cell_type_col="cell_type", min_genes=1, min_cells=1, prefix=prefix
        )

    exported = pd.read_csv(f"{prefix}_expression.csv.gz", index_col=0)
    assert exported.to_numpy().max() < 20, "raw counts were passed to seismicGWAS as logcounts"


def test_run_seismic_torch_gives_the_same_result_for_counts_and_log_normalized_input(tmp_path):
    pytest.importorskip("torch")
    from cellink.tl.external._seismic_torch import run_seismic_torch

    counts = _counts_adata()
    magma_file = tmp_path / "magma.genes.out"
    pd.DataFrame({"GENE": counts.var_names, "ZSTAT": np.random.default_rng(1).normal(size=counts.n_vars)}).to_csv(
        magma_file, sep="\t", index=False
    )
    stale = counts.copy()
    stale.uns["log1p"] = {"base": None}
    in_layer = _lognorm(counts)
    in_layer.layers["counts"] = counts.X.copy()

    kwargs = {
        "magma_file": str(magma_file),
        "cell_type_col": "cell_type",
        "min_genes": 1,
        "min_cells": 1,
        "save_results": False,
    }
    reference = run_seismic_torch(_lognorm(counts), **kwargs).set_index("cell_type")["pvalue"]
    for adata, extra in [(counts, {}), (stale, {}), (in_layer, {"layer": "counts"})]:
        result = run_seismic_torch(adata, **kwargs, **extra).set_index("cell_type")["pvalue"]
        np.testing.assert_allclose(result.loc[reference.index], reference, rtol=1e-4)


def test_run_seismic_torch_leaves_the_input_unchanged(tmp_path):
    pytest.importorskip("torch")
    from cellink.tl.external._seismic_torch import run_seismic_torch

    adata = _counts_adata()
    adata.obs.loc["c0", "cell_type"] = None
    adata.uns["log1p"] = {"base": None}
    adata.layers["counts"] = adata.X.copy()
    before = adata.copy()
    magma_file = tmp_path / "magma.genes.out"
    pd.DataFrame({"GENE": adata.var_names, "ZSTAT": np.random.default_rng(1).normal(size=adata.n_vars)}).to_csv(
        magma_file, sep="\t", index=False
    )

    run_seismic_torch(
        adata,
        magma_file=str(magma_file),
        cell_type_col="cell_type",
        min_genes=1,
        min_cells=1,
        layer="counts",
        save_results=False,
    )

    np.testing.assert_array_equal(adata.X.toarray(), before.X.toarray())
    np.testing.assert_array_equal(adata.layers["counts"].toarray(), before.layers["counts"].toarray())
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    assert "log1p" in adata.uns
