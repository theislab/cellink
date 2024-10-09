from pathlib import Path

import pytest
from sgkit.io.plink import read_plink as sg_read_plink

from cellink.io import from_sgkit_dataset, read_plink, read_sgkit_zarr

DATA = Path("tests/data")


@pytest.mark.slow
def test_read_plink():
    read_plink(DATA / "chr22.dose.filtered.R2_0.8")


@pytest.mark.slow
def test_read_sgkit_zarr():
    read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")


@pytest.mark.slow
def test_from_sgkit_dataset():
    sgkit_dataset = sg_read_plink(path=DATA / "chr22.dose.filtered.R2_0.8")
    from_sgkit_dataset(sgkit_dataset)
