from pathlib import Path

import pytest
import os
from sgkit.io.plink import read_plink as sg_read_plink

from cellink._core.donordata import DonorData
from cellink.io import from_sgkit_dataset, read_plink, read_sgkit_zarr, to_plink, read_donordata_objects
import scanpy as sc
import shutil

DATA = Path("tests/data")


@pytest.mark.slow
def test_read_plink():
    read_plink(DATA / "simulated_genotype_calls")

@pytest.mark.slow
def test_read_sgkit_zarr():
    read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")

@pytest.mark.slow
def test_from_sgkit_dataset():
    sgkit_dataset = sg_read_plink(path=DATA / "simulated_genotype_calls")
    from_sgkit_dataset(sgkit_dataset)

@pytest.mark.slow
def test_export():

    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata = gdata[:, :1996]
    # gdata.obs = gdata.obs.set_index("id")
    os.makedirs("tests/temp")
    to_plink(gdata, output_prefix="tests/temp", num_patients_chunk=100)
    shutil.rmtree("tests/temp")

@pytest.mark.slow
def test_read_donordata_objects(adata, gdata):
    os.makedirs("tests/temp")
    adata.write_h5ad("tests/temp/C.h5ad")
    gdata.write_h5ad("tests/temp/G.h5ad")
    dd = read_donordata_objects("tests/temp/C.h5ad", "tests/temp/G.h5ad")
    assert dd.C.shape == adata.shape
    assert dd.G.shape == gdata.shape
    shutil.rmtree("tests/temp")