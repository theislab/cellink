from pathlib import Path

import pytest

from cellink.io import read_sgkit_zarr
from cellink.tl import simulate_genotype_data_msprime, simulate_genotype_data_numpy, one_hot_encode_genotypes, dosage_per_strand
import numpy as np

DATA = Path("tests/data")

def test_simulate_genotype_data_msprime():
    adata = simulate_genotype_data_msprime(100, 100)
    assert adata.shape == (100, 100)

def test_simulate_genotype_data_numpy():
    adata = simulate_genotype_data_numpy(100, 100)
    assert adata.shape == (100, 100)

def test_one_hot_encode_genotypes():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    one_hot_encode_gdata = one_hot_encode_genotypes(gdata[:100, :100])
    one_hot_encode_gdata_load = np.load(DATA / "simulated_genotype_calls_one_hot_encoded.npy")
    assert np.all(one_hot_encode_gdata == one_hot_encode_gdata_load) == True

def test_dosage_per_strand():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    dosage_per_strand_gdata = dosage_per_strand(gdata[:100, :100])
    dosage_per_strand_gdata_load = np.load(DATA / "simulated_genotype_calls_dosage_per_strand.npy")
    assert np.all(dosage_per_strand_gdata == dosage_per_strand_gdata_load) == True
