from __future__ import annotations

import anndata
import dask.array as da
import numpy as np
import pytest
from scipy import stats

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.pp import (
    cell_level_obs_filter,
    donor_level_obs_filter,
    donor_level_var_filter,
    low_abundance_filter,
    missing_values_filter,
    variant_qc,
)
from cellink.pp._variant_qc import _hwe_chisq_pvalue

ABUNDANCE = np.array([[1e-5, 1.5, 0.2], [2e-5, 1.0, 0.3], [3e-5, 2.0, 0.4]])
MISSING = np.array([[1.0, np.nan, 0.3], [0.5, 0.8, np.nan], [0.7, 0.9, 0.4]])


@pytest.mark.parametrize("wrap", [np.asarray, da.from_array])
def test_low_abundance_filter_inplace_filters_caller(wrap):
    adata = anndata.AnnData(X=wrap(ABUNDANCE))
    assert low_abundance_filter(adata, abundance_threshold=0.5) is None
    assert adata.n_vars == 1


@pytest.mark.parametrize("wrap", [np.asarray, da.from_array])
def test_missing_values_filter_inplace_filters_caller(wrap):
    adata = anndata.AnnData(X=wrap(MISSING))
    assert missing_values_filter(adata, max_missing_ratio=0.3) is None
    assert adata.n_vars == 1


def test_filters_with_inplace_false_leave_input_and_honour_copy():
    adata = anndata.AnnData(X=ABUNDANCE.copy())
    view = low_abundance_filter(adata, abundance_threshold=0.5, inplace=False, copy=False)
    copied = low_abundance_filter(adata, abundance_threshold=0.5, inplace=False, copy=True)
    assert adata.n_vars == 3
    assert view.is_view and view.n_vars == 1
    assert not copied.is_view and copied.n_vars == 1


def _genotypes():
    # variant 0 monomorphic, variant 1 all heterozygous (far from HWE), variant 2 at HWE (50/100/50)
    n = 200
    X = np.zeros((n, 3))
    X[:, 1] = 1
    X[:, 2] = np.repeat([0, 1, 2], [50, 100, 50])
    return anndata.AnnData(X=da.from_array(X))


def test_hwe_chisq_pvalue_matches_formula():
    # counts 30/40/30: p = 0.5, expected 25/50/25, chi-square = 1 + 2 + 1 = 4
    np.testing.assert_allclose(_hwe_chisq_pvalue(np.array([[30], [40], [30]])), stats.chi2.sf(4, df=1))
    np.testing.assert_allclose(_hwe_chisq_pvalue(np.array([[0], [0], [0]])), 1.0)  # uncalled variant passes


def test_variant_qc_applies_hwe_threshold():
    out = variant_qc(_genotypes(), maf_threshold=0.01, hwe_pval_threshold=1e-6, inplace=False)
    assert out.n_vars == 1
    assert out.var_names[0] == "2"
    kept_without_hwe = variant_qc(_genotypes(), maf_threshold=0.01, hwe_pval_threshold=0.0, inplace=False)
    assert kept_without_hwe.n_vars == 2


def test_variant_qc_inplace_filters_caller():
    gdata = _genotypes()
    assert variant_qc(gdata, maf_threshold=0.01, hwe_pval_threshold=1e-6) is None
    assert list(gdata.var_names) == ["2"]


def test_donordata_filters_inplace_filter_caller():
    dd = DonorData(G=sim_gdata(), C=sim_adata())
    assert cell_level_obs_filter(dd, "celltype", "CD8") is None
    assert set(dd.C.obs["celltype"]) == {"CD8"}

    dd = DonorData(G=sim_gdata(), C=sim_adata())
    dd.G.obs["group"] = ["a" if i % 2 else "b" for i in range(dd.G.n_obs)]
    donor_level_obs_filter(dd, "group", "a")
    assert set(dd.G.obs["group"]) == {"a"}
    assert set(dd.C.obs[dd.donor_id]) == set(dd.G.obs_names)

    dd = DonorData(G=sim_gdata(), C=sim_adata())
    dd.G.var["kind"] = ["x", "y"] + ["x"] * (dd.G.n_vars - 2)
    donor_level_var_filter(dd, donor_level_key="kind", donor_level_values="y")
    assert dd.G.n_vars == 1


def test_donordata_filter_inplace_false_leaves_input():
    dd = DonorData(G=sim_gdata(), C=sim_adata())
    n_cells = dd.C.n_obs
    out = cell_level_obs_filter(dd, "celltype", ["CD8"], inplace=False)
    assert dd.C.n_obs == n_cells
    assert set(out.C.obs["celltype"]) == {"CD8"}
