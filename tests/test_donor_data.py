from pathlib import Path

import pytest
from anndata import read_h5ad

from cellink._core.donordata import DonorData
from cellink.io import read_sgkit_zarr
from cellink.pp import annotate_genes

DATA = Path("tests/data")


@pytest.fixture
def gdata():
    gdata = read_sgkit_zarr(DATA / "chr22.dose.filtered.R2_0.8.vcz")
    gdata.obs = gdata.obs.set_index("id")
    return gdata


@pytest.fixture
def annotated_adata():
    # Load the original AnnData object
    adata = read_h5ad(DATA / "debug_OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad")

    # Path to the gene annotation file
    gene_annotation_path = DATA / "gene_annotation.csv"

    # Annotate genes
    annotated_adata = annotate_genes(adata, str(gene_annotation_path))

    return annotated_adata


def test_donordata_init(annotated_adata, gdata):
    dd = DonorData(annotated_adata, gdata, "individual")
    print(dd)


def test_donordata_aggregate(annotated_adata, gdata):
    dd = DonorData(annotated_adata, gdata, "individual")
    dd.aggregate("X", "Gex")
    assert "Gex" in dd.gdata.obsm

    previous_adata_shape = dd.adata.shape
    dd.aggregate("X", "Gex_CD4NC", filter_key="cell_label", filter_value="CD4 NC")
    assert "Gex_CD4NC" in dd.gdata.obsm
    assert dd.adata.shape == previous_adata_shape

    dd.aggregate("age", "age")
    assert "age" in dd.gdata.obs


@pytest.mark.parametrize(
    "ensembl_id, window_size, tss",
    [
        ("ENSG00000100354", 1e5, True),
        ("ENSG00000100354", 5e4, True),
        ("ENSG00000100354", 1e5, False),
        ("ENSG00000183486", 2e5, True),  # Add another gene ID for more diverse testing
    ],
)
def test_donordata_slice_genomic_region_by_gene(annotated_adata, gdata, ensembl_id, window_size, tss):
    dd = DonorData(annotated_adata, gdata, "individual")

    sliced_dd = dd.slice_genomic_region(ensembl_id, window_size=window_size, tss=tss)

    # Check that the sliced data contains only the specified region
    assert all(sliced_dd.gdata.var.chrom == dd.adata.var.loc[ensembl_id, "chrom"])

    gene_start = dd.adata.var.loc[ensembl_id, "start_position"]
    gene_end = dd.adata.var.loc[ensembl_id, "end_position"]

    if tss:
        tss_pos = gene_start if dd.adata.var.loc[ensembl_id, "strand"] == 1 else gene_end
        assert all(sliced_dd.gdata.var.pos >= tss_pos - window_size)
        assert all(sliced_dd.gdata.var.pos <= tss_pos + window_size)
    else:
        assert all(sliced_dd.gdata.var.pos >= gene_start - window_size)
        assert all(sliced_dd.gdata.var.pos <= gene_end + window_size)

    # Check that the single-cell data remains unchanged
    assert sliced_dd.adata.shape == dd.adata.shape

    # Check that the number of variants has been reduced
    assert sliced_dd.gdata.shape[1] < dd.gdata.shape[1]


@pytest.mark.parametrize(
    "chrom, start, end, window_size, expected_n_vars",
    [
        ("22", 20000000, 21000000, 10000, 2445),
        ("22", 30000000, 31000000, 5000, 4157),
    ],
)
def test_donordata_slice_genomic_region(annotated_adata, gdata, chrom, start, end, window_size, expected_n_vars):
    dd = DonorData(annotated_adata, gdata, "individual")

    # Slice the genomic region
    sliced_dd = dd._slice_genomic_region(chrom=chrom, start=start, end=end, window_size=window_size)

    # Check that the sliced data contains only the specified region
    assert sliced_dd.gdata.n_vars == expected_n_vars
    assert all(sliced_dd.gdata.var.chrom == chrom)
    assert all(sliced_dd.gdata.var.pos >= start - window_size)
    assert all(sliced_dd.gdata.var.pos <= end + window_size)

    # Check that the single-cell data remains unchanged
    assert sliced_dd.adata.shape == dd.adata.shape

    # Check that the number of variants has been reduced
    assert sliced_dd.gdata.shape[1] < dd.gdata.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])
