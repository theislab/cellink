from pathlib import Path

import pytest

from cellink.io import read_sgkit_zarr
from cellink.pp import variant_qc

DATA = Path("tests/data")


@pytest.mark.slow
def test_variant_qc():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")

    gdata_filt = variant_qc(gdata, maf_threshold=0.47, inplace=False)

    assert gdata_filt.shape[1] == 1989
