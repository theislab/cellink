from pathlib import Path

import pytest

import anndata
import dask.array as da
import numpy as np
from cellink.io import read_sgkit_zarr
from cellink.pp import variant_qc, low_abundance_filter, missing_values_filter, log_transform, normalize

DATA = Path("tests/data")


@pytest.mark.slow
def test_variant_qc():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")

    gdata_filt = variant_qc(gdata, maf_threshold=0.47, inplace=False)

    assert gdata_filt.shape[1] == 1989

@pytest.mark.slow
def test_low_abundance_filter():
    X = da.from_array(np.array([
        [1e-5, 1.5, 0.2],
        [2e-5, 1.0, 0.3],
        [3e-5, 2.0, 0.4],
    ]))
    adata = anndata.AnnData(X=X)

    filtered_mean = low_abundance_filter(adata, abundance_threshold=0.5, method="mean", inplace=False)
    assert filtered_mean.n_vars == 1

    filtered_median = low_abundance_filter(adata, abundance_threshold=0.3, method="median", inplace=False)
    assert filtered_median.n_vars == 2

@pytest.mark.slow
def test_missing_values_filter():
    X = da.from_array(np.array([
        [1.0, np.nan, 0.3],
        [0.5, 0.8, np.nan],
        [0.7, 0.9, 0.4],
    ]))
    adata = anndata.AnnData(X=X)

    filtered = missing_values_filter(adata, max_missing_ratio=0.3, inplace=False)
    assert filtered.n_vars == 1

@pytest.mark.slow
def test_log_transform():
    X = da.from_array(np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]))
    adata = anndata.AnnData(X=X)

    log_transformed = log_transform(adata, base=2, inplace=False)

    log_transformed_base_10 = log_transform(adata, base=10, inplace=False)

@pytest.mark.slow
def test_normalize():
    X = da.from_array(np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]))
    adata = anndata.AnnData(X=X)

    normalized_zscore = normalize(adata, method="zscore", inplace=False)

    normalized_minmax = normalize(adata, method="minmax", inplace=False)

    normalized_median = normalize(adata, method="median", inplace=False)

