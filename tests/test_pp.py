from pathlib import Path

import anndata
import dask.array as da
import numpy as np
import pytest
from scipy import stats

from cellink.io import read_sgkit_zarr
from cellink.pp import (
    log_transform,
    low_abundance_filter,
    missing_values_filter,
    normalize,
    variant_qc,
)
from cellink.pp._variant_qc import _hwe_chisq_pvalue

DATA = Path("tests/data")


def test_variant_qc():
    pytest.importorskip("sgkit")
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")

    gdata_filt = variant_qc(gdata, maf_threshold=0.47, inplace=False)

    assert gdata_filt.shape[1] == 1989


X = da.from_array(
    np.array(
        [
            [1e-5, 1.5, 0.2],
            [2e-5, 1.0, 0.3],
            [3e-5, 2.0, 0.4],
        ]
    )
)


@pytest.mark.parametrize("X", [X])
def test_low_abundance_filter(X):
    adata = anndata.AnnData(X=X)

    filtered_mean = low_abundance_filter(adata, abundance_threshold=0.5, method="mean", inplace=False)
    assert filtered_mean.n_vars == 1

    filtered_median = low_abundance_filter(adata, abundance_threshold=0.3, method="median", inplace=False)
    assert filtered_median.n_vars == 2


X = da.from_array(
    np.array(
        [
            [1.0, np.nan, 0.3],
            [0.5, 0.8, np.nan],
            [0.7, 0.9, 0.4],
        ]
    )
)


@pytest.mark.parametrize("X", [X])
def test_missing_values_filter(X):
    adata = anndata.AnnData(X=X)

    filtered = missing_values_filter(adata, max_missing_ratio=0.3, inplace=False)
    assert filtered.n_vars == 1


X = da.from_array(
    np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
)


@pytest.mark.parametrize("X", [X])
def test_log_transform(X):
    adata = anndata.AnnData(X=X)
    X_np = np.asarray(X)

    log_transformed = log_transform(adata, base=2, inplace=False)
    np.testing.assert_allclose(np.asarray(log_transformed.X), np.log1p(X_np) / np.log(2))

    log_transformed_base_10 = log_transform(adata, base=10, inplace=False)
    np.testing.assert_allclose(np.asarray(log_transformed_base_10.X), np.log1p(X_np) / np.log(10))


@pytest.mark.parametrize("X", [X])
def test_normalize(X):
    adata = anndata.AnnData(X=X)

    normalized_zscore = normalize(adata, method="zscore", inplace=False)
    X_zscore = np.asarray(normalized_zscore.X)
    np.testing.assert_allclose(X_zscore.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_zscore.std(axis=0), 1, atol=1e-10)

    normalized_minmax = normalize(adata, method="minmax", inplace=False)
    X_minmax = np.asarray(normalized_minmax.X)
    np.testing.assert_allclose(X_minmax.min(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_minmax.max(axis=0), 1, atol=1e-10)

    normalized_median = normalize(adata, method="median", inplace=False)
    X_median = np.asarray(normalized_median.X)
    np.testing.assert_allclose(np.median(X_median, axis=0), 0, atol=1e-10)


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


def test_variant_qc_maf_ignores_missing_calls():
    # missing calls (nan, as the PLINK/PGEN readers return them) should not drop common variants
    # or skew the denominator of the allele frequency
    X = _genotypes().X.compute()
    X[0, 2] = np.nan
    out = variant_qc(anndata.AnnData(X=da.from_array(X)), maf_threshold=0.01, hwe_pval_threshold=1e-6, inplace=False)
    assert list(out.var_names) == ["2"]

    # 200 donors, 3 variants with missing calls:
    # 0: 100 nan + 90 ref + 10 het -> called MAF = 10 / (2 * 100) = 0.05 (0.025 if divided by 2 * n_obs)
    # 1: 100 nan + 100 hom-alt -> called AF = 1.0, MAF = 0.0 (0.5 if divided by 2 * n_obs)
    # 2: all nan -> uncalled, dropped at maf_threshold > 0
    X_miss = np.full((200, 3), np.nan)
    X_miss[:90, 0] = 0.0
    X_miss[90:100, 0] = 1.0
    X_miss[:100, 1] = 2.0
    out_miss = variant_qc(
        anndata.AnnData(X=da.from_array(X_miss)), maf_threshold=0.04, hwe_pval_threshold=1e-6, inplace=False
    )
    assert list(out_miss.var_names) == ["0"]
