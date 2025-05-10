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
def test_export(tmp_path):
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata = gdata[:, :1996]

    to_plink(gdata, output_prefix=str(tmp_path), num_patients_chunk=100)


@pytest.mark.slow
def test_read_donordata_object(tmp_path, adata, gdata):
    output_path = tmp_path / "donordata.dd.h5"

    dd = DonorData(G=gdata, C=adata)
    dd.write_donordata_object(str(output_path))

    dd_loaded = read_donordata_object(str(output_path))

    assert dd_loaded.C.shape == dd.C.shape
    assert dd_loaded.G.shape == dd.G.shape
