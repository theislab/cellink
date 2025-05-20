from pathlib import Path

import numpy as np

from cellink.io import read_sgkit_zarr
from cellink.tl import (
    dosage_per_strand,
    one_hot_encode_genotypes,
)

DATA = Path("tests/data")


def test_one_hot_encode_genotypes():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    one_hot_encode_gdata = one_hot_encode_genotypes(gdata[:100, :100])
    one_hot_encode_gdata_load = np.load(DATA / "simulated_genotype_calls_one_hot_encoded.npy")
    assert np.allclose(one_hot_encode_gdata, one_hot_encode_gdata_load)


def test_dosage_per_strand():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    dosage_per_strand_gdata = dosage_per_strand(gdata[:100, :100])
    dosage_per_strand_gdata_load = np.load(DATA / "simulated_genotype_calls_dosage_per_strand.npy")
    assert np.allclose(dosage_per_strand_gdata, dosage_per_strand_gdata_load)
