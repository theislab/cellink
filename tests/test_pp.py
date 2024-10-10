from pathlib import Path

import pytest

DATA = Path("tests/data")

@pytest.mark.slow
def test_variant_qc():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")

    variant_qc(gdata)

