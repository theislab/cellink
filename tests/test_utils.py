from pathlib import Path

import numpy as np
import pytest

from cellink.io import read_sgkit_zarr
from cellink.utils import (
    column_normalize,
    dosage_per_strand,
    gaussianize,
    one_hot_encode_genotypes,
)

DATA = Path("tests/data")
CONFIG = Path("configs")


def test_one_hot_encode_genotypes():
    pytest.importorskip("sgkit")
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    one_hot_encode_gdata = one_hot_encode_genotypes(gdata[:100, :100])
    one_hot_encode_gdata_load = np.load(DATA / "simulated_genotype_calls_one_hot_encoded.npy")
    assert np.allclose(one_hot_encode_gdata, one_hot_encode_gdata_load)


def test_dosage_per_strand():
    pytest.importorskip("sgkit")
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    dosage_per_strand_gdata = dosage_per_strand(gdata[:100, :100])
    dosage_per_strand_gdata_load = np.load(DATA / "simulated_genotype_calls_dosage_per_strand.npy")
    assert np.allclose(dosage_per_strand_gdata, dosage_per_strand_gdata_load)


def test_column_normalize():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(200, 4))

    X_norm = column_normalize(X)

    assert X_norm.shape == X.shape
    np.testing.assert_allclose(X_norm.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_norm.std(axis=0), 1 / np.sqrt(X.shape[1]))


def test_gaussianize():
    rng = np.random.default_rng(0)
    Y = rng.exponential(scale=3.0, size=(500, 2))

    Y_gauss = gaussianize(Y)

    assert Y_gauss.shape == Y.shape
    # rank order within each column must be preserved
    for col in range(Y.shape[1]):
        assert (np.argsort(Y[:, col]) == np.argsort(Y_gauss[:, col])).all()
    np.testing.assert_allclose(Y_gauss.mean(axis=0), 0, atol=1e-1)
    np.testing.assert_allclose(Y_gauss.std(axis=0), 1, rtol=1e-1)
