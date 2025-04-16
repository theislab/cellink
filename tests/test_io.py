import os
import shutil
from pathlib import Path

import pytest
from sgkit.io.plink import read_plink as sg_read_plink

from cellink import DonorData
from cellink.io import from_sgkit_dataset, read_donordata_object, read_plink, read_sgkit_zarr, to_plink

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
def test_read_donordata_object(adata, gdata):
    os.makedirs("tests/temp")
    dd = DonorData(G=gdata, C=adata)
    dd.write_donordata_object("tests/temp/donordata.dd.h5")
    dd_loaded = read_donordata_object("tests/temp/donordata.dd.h5")
    assert dd.C.shape == dd.C.shape
    assert dd.G.shape == dd.G.shape
    shutil.rmtree("tests/temp")
