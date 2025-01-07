from pathlib import Path

import pytest
from sgkit.io.plink import read_plink as sg_read_plink

from anndata import read_h5ad
from cellink._core.donordata import DonorData
from cellink.io import from_sgkit_dataset, read_plink, read_sgkit_zarr, generate_bim_fam, to_plink

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
def test_generate_bim_fam():

    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    # gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")


    print("sdgagsh")
    generate_bim_fam()

@pytest.mark.slow
def test_to_plink():

    print("ksdgfaasgd")
    to_plink()
