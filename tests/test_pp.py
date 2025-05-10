from pathlib import Path

import anndata
import dask.array as da
import numpy as np
import pytest

from cellink.io import read_sgkit_zarr
from cellink.pp import log_transform, low_abundance_filter, missing_values_filter, normalize, variant_qc

DATA = Path("tests/data")


def test_variant_qc():
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

    log_transformed = log_transform(adata, base=2, inplace=False)

    log_transformed_base_10 = log_transform(adata, base=10, inplace=False)

@pytest.mark.parametrize("X", [X])
def test_normalize(X):
    adata = anndata.AnnData(X=X)

    normalized_zscore = normalize(adata, method="zscore", inplace=False)

    normalized_minmax = normalize(adata, method="minmax", inplace=False)

    normalized_median = normalize(adata, method="median", inplace=False)
