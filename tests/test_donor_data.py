from pathlib import Path

import pytest
from anndata import read_h5ad

from cellink._core.donordata import DonorData
from cellink.io import read_sgkit_zarr

DATA = Path("tests/data")


@pytest.mark.slow
def test_donordata_init():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad")
    dd = DonorData(adata, gdata, "individual")
    print(dd)


@pytest.mark.slow
def test_donordata_aggregate():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad")
    dd = DonorData(adata, gdata, "individual")
    dd.aggregate("X", "Gex")
    assert "Gex" in dd.gdata.obsm

    previous_adata_shape = dd.adata.shape
    dd.aggregate("X", "Gex_CD4NC", filter_key="cell_label", filter_value="CD4 NC")
    assert "Gex_CD4NC" in dd.gdata.obsm
    assert dd.adata.shape == previous_adata_shape

    dd.aggregate("age", "age")
    assert "age" in dd.gdata.obs


@pytest.mark.slow
def test_donordata_slice_genomic_region():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad")
    dd = DonorData(adata, gdata, "individual")

    # Slice the genomic region
    sliced_dd = dd.slice_genomic_region(chrom="22", start=20000000, end=21000000, cis_window=10000)

    # Check that the sliced data contains only the specified region
    assert sliced_dd.gdata.n_vars == 2445
    assert all(sliced_dd.gdata.var.chrom == "22")
    assert all(sliced_dd.gdata.var.pos >= 19990000)  # start - cis_window
    assert all(sliced_dd.gdata.var.pos <= 21010000)  # end + cis_window

    # Check that the single-cell data remains unchanged
    assert sliced_dd.adata.shape == dd.adata.shape

    # Check that the number of variants has been reduced
    assert sliced_dd.gdata.shape[1] < dd.gdata.shape[1]


if __name__ == "__main__":
    test_donordata_aggregate()
    test_donordata_slice_genomic_region()
