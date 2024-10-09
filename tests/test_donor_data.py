from pathlib import Path

import pytest
from anndata import read_h5ad

from cellink._core.donordata import DonorData
from cellink.io import read_sgkit_zarr

DATA = Path("tests/data")


@pytest.mark.slow
def test_donordata():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(
        DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad"
    )
    dd = DonorData(adata, gdata, "individual")
    print(dd)
    assert False


if __name__ == "__main__":
    test_donordata()
