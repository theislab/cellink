from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from cellink.tl.external._scdrs import SCDRS_AVAILABLE, run_scdrs

pytestmark = pytest.mark.skipif(not SCDRS_AVAILABLE, reason="scdrs not installed")

GENES = [f"SYM{i}" for i in range(200)]


def _counts_adata(n_cells=200):
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(n_cells, len(GENES))).astype(np.float32)
    return AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]), var=pd.DataFrame(index=GENES))


def _lognorm(adata):
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def _scores(adata):
    out = run_scdrs(
        adata,
        gene_sets={"trait": (GENES[:30], [1.0] * 30)},
        n_ctrl=20,
        min_genes=1,
        min_cells=1,
        save_results=False,
    )
    return out[0] if isinstance(out, tuple) else out


def _numeric(df):
    return df.sort_index().select_dtypes("number").to_numpy()


def test_run_scdrs_scores_do_not_depend_on_how_the_input_is_normalized():
    counts = _counts_adata()
    reference = _numeric(_scores(_lognorm(counts)))

    counts_int = counts.copy()
    counts_int.X = counts_int.X.astype(np.int64)
    stale_flag = counts.copy()
    stale_flag.uns["log1p"] = {"base": None}
    lognorm_no_flag = _lognorm(counts)
    del lognorm_no_flag.uns["log1p"]

    for adata in (counts, counts_int, stale_flag, lognorm_no_flag):
        np.testing.assert_allclose(_numeric(_scores(adata)), reference, rtol=1e-5, atol=1e-8)


def test_run_scdrs_adds_scores_to_obs_and_leaves_the_rest_unchanged():
    adata = _counts_adata()
    adata.X[0] = 0
    adata.uns["log1p"] = {"base": None}
    adata.layers["counts"] = adata.X.copy()
    before = adata.X.copy()

    scores = _scores(adata)

    np.testing.assert_array_equal(adata.X, before)
    np.testing.assert_array_equal(adata.layers["counts"], before)
    assert adata.shape == (200, len(GENES))
    assert "log1p" in adata.uns
    assert list(adata.obs.columns) == ["trait_norm_score", "trait_pval"]
    assert np.isnan(adata.obs["trait_pval"].iloc[0])
    np.testing.assert_array_equal(adata.obs["trait_pval"].iloc[1:], scores["pval"])


def test_run_scdrs_returns_its_own_filtered_copy():
    adata = _counts_adata()
    adata.X[0] = 0

    processed = run_scdrs(
        adata,
        gene_sets={"trait": (GENES[:30], [1.0] * 30)},
        n_ctrl=20,
        min_genes=1,
        min_cells=1,
        save_results=False,
        return_adata=True,
    )

    assert processed.n_obs == adata.n_obs - 1
    assert not processed.layers
    assert not np.shares_memory(processed.X, adata.X)
