from pathlib import Path

import mudata as md

from cellink._core.data_fields import CAnn, DAnn
from cellink._core.donordata import DonorData

md.set_options(pull_on_update=False)

DATA = Path("tests/data")


def test_donordata_init(adata, gdata):
    DonorData(G=gdata, C=adata)


def test_slice_donors(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    keep_donors = ["D0", "D1"]
    dd = dd[keep_donors]
    sub_adata = adata[adata.obs[DAnn.donor].isin(keep_donors)]
    assert dd.shape[0] == len(keep_donors)
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == sub_adata.shape[0]
    assert dd.shape[3] == sub_adata.shape[1]


def test_slice_variants(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    keep_variants = ["SNP0", "SNP1"]
    dd = dd[:, keep_variants]
    assert dd.shape[0] == gdata.shape[0]
    assert dd.shape[1] == len(keep_variants)
    assert dd.shape[2] == adata.shape[0]
    assert dd.shape[3] == adata.shape[1]


def test_slice_cells(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    keep_cells = ["C0", "C1"]
    dd = dd[:, :, keep_cells]
    keep_individuals = adata[keep_cells].obs[DAnn.donor].unique()
    assert dd.shape[0] == len(keep_individuals)
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == len(keep_cells)
    assert dd.shape[3] == adata.shape[1]


def test_slice_genes(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    keep_genes = ["G1", "G2"]
    print(adata.var.index)
    dd = dd[:, :, :, keep_genes]
    assert dd.shape[0] == gdata.shape[0]
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == adata.shape[0]
    assert dd.shape[3] == len(keep_genes)


def test_slice_all(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    dd = dd["D0", "SNP1", "C0", "G1"]
    assert dd.shape == (1, 1, 1, 1)


def test_donordata_aggregate(adata, gdata, dummy_covariates):
    previous_adata_shape = adata.shape

    dd = DonorData(G=gdata, C=adata)

    dd.aggregate(key_added="Gex")
    assert "Gex" in dd.G.obsm
    assert dd.C.shape == previous_adata_shape

    for celltype in adata.obs[CAnn.celltype].unique():
        celltype_key = f"Gex_{celltype}"
        dd.aggregate(key_added=celltype_key, filter_key=CAnn.celltype, filter_value=celltype)
        assert celltype_key in dd.G.obsm
        assert dd.C.shape == previous_adata_shape

    dd.aggregate(obs=dummy_covariates, func="first", add_to_obs=True)
    assert all(col in dd.G.obs.columns for col in dummy_covariates)


def test_sel_dict_indexing(adata, gdata):
    """Test the __getitem__ method using dictionary indexing."""
    dd = DonorData(G=gdata, C=adata)
    idx_dict = {
        "G_obs": slice(0, 1),
        "G_var": slice(0, 2),
        "C_obs": slice(0, 3),
        "C_var": slice(0, 4),
    }
    dd_dict = dd.sel(**idx_dict)
    # Check that the returned object is a DonorData instance
    assert isinstance(dd_dict, DonorData)
    # Expected shape: (number of donors, number of variants, number of cells, number of genes)
    # Here, slice(0,1) gives 1 donor, slice(0,2) gives 2 variants, slice(0,3) gives 3 cells, and slice(0,4) gives 4 genes
    expected_shape = (1, 2, 3, 4)
    assert dd_dict.shape == expected_shape


def test_mudata_getitem(adata, gdata):
    """Test that DonorData functionality works when using MuData objects for G and C."""

    mu_gdata = md.MuData({"genotype": gdata})
    mu_adata = md.MuData({"RNA": adata, "atac": adata[:, :5].copy()}, axis=-1)
    mu_adata.obs[DAnn.donor] = mu_adata.mod["RNA"].obs[DAnn.donor]
    dd = DonorData(G=mu_gdata, C=mu_adata)

    # Test tuple indexing with MuData objects
    dd_tuple = dd["D0", "SNP1", "C0", "G1"]
    assert dd_tuple.shape == (1, 1, 1, 1)


def test_mudata_sel_dict_indexing(adata, gdata):
    # Also test dictionary indexing with MuData objects
    mu_gdata = md.MuData({"genotype": gdata})
    mu_adata = md.MuData({"RNA": adata, "atac": adata[:, :5].copy()}, axis=-1)
    mu_adata.obs[DAnn.donor] = mu_adata.mod["RNA"].obs[DAnn.donor]
    dd = DonorData(G=mu_gdata, C=mu_adata)

    idx_dict = {
        "G_obs": slice(0, 2),
        "G_var": slice(1, 3),
        "C_obs": slice(1, 4),
        "C_var": slice(2, 5),
    }
    dd_dict = dd.sel(**idx_dict)
    assert isinstance(dd_dict, DonorData)
    # Expected shape: 1 donor, 2 variants, 3 cells, 3 genes
    expected_shape = (1, 2, 3, 3)
    assert dd_dict.shape == expected_shape


def test_ellipsis_indexing(adata, gdata):
    """Test __getitem__ handling of ellipsis by ensuring dd[Ellipsis] returns a DonorData with the same shape as dd."""
    dd = DonorData(G=gdata, C=adata)
    dd_ellipsis = dd[Ellipsis]
    assert isinstance(dd_ellipsis, DonorData)
    # Using ellipsis should return a DonorData with full selection, matching the original shape
    assert dd_ellipsis.shape == dd.shape


def test_ellipsis_with_final_index(adata, gdata):
    """Test __getitem__ with ellipsis at the beginning and a final index.

    This should expand dd[..., 'G1'] into (slice(None), slice(None), slice(None), 'G1'),
    resulting in a shape with the last dimension being of size 1."""
    dd = DonorData(G=gdata, C=adata)
    dd_res = dd[..., "G1"]
    # Expected shape: (num_donors, num_G_vars, num_cells, 1)
    num_donors, num_G_vars = gdata.shape
    num_cells, num_genes = adata.shape
    expected_shape = (num_donors, num_G_vars, num_cells, 1)
    assert dd_res.shape == expected_shape


def test_ellipsis_middle_indexing(adata, gdata):
    """Test __getitem__ with ellipsis in the middle.

    For example, dd['D0', ..., 'G1'] should expand to
    ('D0', slice(None), slice(None), 'G1') and select a single donor and gene."""
    dd = DonorData(G=gdata, C=adata)
    dd_res = dd["D0", Ellipsis, "G1"]
    # Expected shape: (1, num_G_vars, num_cells, 1)
    _, num_G_vars = gdata.shape
    num_cells = adata[adata.obs[DAnn.donor] == "D0"].shape[0]
    expected_shape = (1, num_G_vars, num_cells, 1)
    assert dd_res.shape == expected_shape


if __name__ == "__main__":
    import pytest

    pytest.main()
