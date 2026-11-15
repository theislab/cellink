from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external._scdrs import SCDRS_AVAILABLE, run_scdrs

pytestmark = pytest.mark.skipif(not SCDRS_AVAILABLE, reason="scdrs not installed")


def _make_adata(var_names, n_cells=200):
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(n_cells, len(var_names))).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))


def test_run_scdrs_gene_sets_dict_raises_on_id_mismatch():
    var_names = [f"SYM{i}" for i in range(200)]
    adata = _make_adata(var_names)
    # simulates Ensembl-vs-symbol mismatch
    mismatched_genes = [f"ENSG{i:011d}" for i in range(50)]

    with pytest.raises(ValueError, match="gene-ID scheme mismatch"):
        run_scdrs(
            adata,
            gene_sets={"trait": (mismatched_genes, [1.0] * 50)},
            min_genes=1,
            min_cells=1,
            save_results=False,
        )


def test_run_scdrs_gene_sets_dict_proceeds_on_matching_ids():
    var_names = [f"SYM{i}" for i in range(200)]
    adata = _make_adata(var_names)
    matching_genes = var_names[:30]

    try:
        run_scdrs(
            adata,
            gene_sets={"trait": (matching_genes, [1.0] * 30)},
            min_genes=1,
            min_cells=1,
            save_results=False,
        )
    except ValueError as e:
        assert "gene-ID scheme mismatch" not in str(e)


def test_run_scdrs_gs_file_raises_on_id_mismatch_not_neutered_by_pre_intersection(tmp_path):
    var_names = [f"SYM{i}" for i in range(200)]
    adata = _make_adata(var_names)
    real_genes = var_names[:4]
    fake_genes = [f"ENSG{i:011d}" for i in range(196)]
    gs_path = tmp_path / "trait.gs"
    gs_path.write_text("TRAIT\tGENESET\n" + f"trait\t{','.join(real_genes + fake_genes)}\n")

    with pytest.raises(ValueError, match="gene-ID scheme mismatch"):
        run_scdrs(
            adata,
            gs_file=str(gs_path),
            min_genes=1,
            min_cells=1,
            min_overlap_genes=10,
            save_results=False,
        )


def test_run_scdrs_nan_sex_does_not_become_fake_third_category():
    var_names = [f"SYM{i}" for i in range(200)]
    adata = _make_adata(var_names, n_cells=200)
    adata.obs["sex"] = ["M"] * 100 + ["F"] * 100
    adata.obs.loc[adata.obs_names[:10], "sex"] = np.nan
    matching_genes = var_names[:30]

    run_scdrs(
        adata,
        gene_sets={"trait": (matching_genes, [1.0] * 30)},
        min_genes=1,
        min_cells=1,
        encode_sex=True,
        encode_age=False,
        save_results=False,
    )


def test_run_scdrs_nan_age_does_not_drop_entire_age_covariate():
    var_names = [f"SYM{i}" for i in range(200)]
    adata = _make_adata(var_names, n_cells=200)
    rng = np.random.default_rng(1)
    adata.obs["age"] = rng.normal(50, 10, adata.n_obs)
    adata.obs.loc[adata.obs_names[:10], "age"] = np.nan
    matching_genes = var_names[:30]

    run_scdrs(
        adata,
        gene_sets={"trait": (matching_genes, [1.0] * 30)},
        min_genes=1,
        min_cells=1,
        encode_sex=False,
        encode_age=True,
        save_results=False,
    )


def test_scdrs_heterogeneity_test_runs_with_compat_shims():
    """scDRS's Geary's C test runs inside _scdrs_compat, which restores NumPy and SciPy afterwards."""
    import scdrs
    import scipy.sparse as sp
    import scipy.stats

    from cellink.tl.external._scdrs import _scdrs_compat

    had_alias, rankdata = hasattr(np, "float_"), scipy.stats.rankdata
    rng = np.random.default_rng(0)
    adata = _make_adata([f"G{i}" for i in range(5)], n_cells=12)
    adata.obs["group"] = np.repeat(["a", "b"], 6)
    adata.obsp["connectivities"] = sp.csr_matrix(np.ones((12, 12)) - np.eye(12))
    df_score = pd.DataFrame(
        {"norm_score": rng.normal(size=12), **{f"ctrl_norm_score_{i}": rng.normal(size=12) for i in range(5)}},
        index=adata.obs_names,
    )
    with _scdrs_compat():
        res = scdrs.method.test_gearysc(adata, df_score, groupby="group")
    assert set(res.index) == {"a", "b"} and np.isfinite(res["pval"]).all()
    assert hasattr(np, "float_") == had_alias and scipy.stats.rankdata is rankdata
