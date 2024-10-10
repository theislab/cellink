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
    sliced_dd = dd._slice_genomic_region(chrom="22", start=20000000, end=21000000, window_size=10000)

    # Check that the sliced data contains only the specified region
    assert sliced_dd.gdata.n_vars == 2445
    assert all(sliced_dd.gdata.var.chrom == "22")
    assert all(sliced_dd.gdata.var.pos >= 19990000)  # start - window_size
    assert all(sliced_dd.gdata.var.pos <= 21010000)  # end + window_size

    # Check that the single-cell data remains unchanged
    assert sliced_dd.adata.shape == dd.adata.shape

    # Check that the number of variants has been reduced
    assert sliced_dd.gdata.shape[1] < dd.gdata.shape[1]


@pytest.mark.slow
def test_donordata_slice_genomic_region_by_gene():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad")
    dd = DonorData(adata, gdata, "individual")

    # Slice the genomic region by gene (replace with an actual Ensembl ID from your data)
    ensembl_id = "ENSG00000100354"  # Example Ensembl ID, replace with a real one from your data
    sliced_dd = dd.slice_genomic_region(ensembl_id, window_size=1e5, tss=True)

    # Check that the sliced data contains only the specified region
    assert all(sliced_dd.gdata.var.chrom == dd.gene_annotation.loc[ensembl_id, "chromosome_name"])

    gene_start = dd.gene_annotation.loc[ensembl_id, "start_position"]
    gene_end = dd.gene_annotation.loc[ensembl_id, "end_position"]
    tss = gene_start if dd.gene_annotation.loc[ensembl_id, "strand"] == 1 else gene_end

    assert all(sliced_dd.gdata.var.pos >= tss - 1e5)
    assert all(sliced_dd.gdata.var.pos <= tss + 1e5)

    # Check that the single-cell data remains unchanged
    assert sliced_dd.adata.shape == dd.adata.shape

    # Check that the number of variants has been reduced
    assert sliced_dd.gdata.shape[1] < dd.gdata.shape[1]

    # Test with tss=False
    sliced_dd_gene_body = dd.slice_genomic_region(ensembl_id, window_size=1e5, tss=False)
    assert all(sliced_dd_gene_body.gdata.var.pos >= gene_start - 1e5)
    assert all(sliced_dd_gene_body.gdata.var.pos <= gene_end + 1e5)


if __name__ == "__main__":
    test_donordata_aggregate()
    test_donordata_slice_genomic_region()
    test_donordata_slice_genomic_region_by_gene()
