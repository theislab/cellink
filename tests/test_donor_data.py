from pathlib import Path

from cellink._core.annotation import CAnn, DAnn
from cellink._core.donordata import DonorData

DATA = Path("tests/data")


def test_donordata_init(adata, gdata):
    DonorData(adata, gdata)


def test_slice_donors(adata, gdata):
    dd = DonorData(adata, gdata)
    keep_donors = ["D0", "D1"]
    dd = dd[keep_donors]
    sub_adata = adata[adata.obs[DAnn.donor].isin(keep_donors)]
    assert dd.shape[0] == len(keep_donors)
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == sub_adata.shape[0]
    assert dd.shape[3] == sub_adata.shape[1]


def test_slice_variants(adata, gdata):
    dd = DonorData(adata, gdata)
    keep_variants = ["SNP0", "SNP1"]
    dd = dd[:, keep_variants]
    assert dd.shape[0] == gdata.shape[0]
    assert dd.shape[1] == len(keep_variants)
    assert dd.shape[2] == adata.shape[0]
    assert dd.shape[3] == adata.shape[1]


def test_slice_cells(adata, gdata):
    dd = DonorData(adata, gdata)
    keep_cells = ["C0", "C1"]
    dd = dd[:, :, keep_cells]
    keep_individuals = adata[keep_cells].obs[DAnn.donor].unique()
    assert dd.shape[0] == len(keep_individuals)
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == len(keep_cells)
    assert dd.shape[3] == adata.shape[1]


def test_slice_genes(adata, gdata):
    dd = DonorData(adata, gdata)
    keep_genes = ["G1", "G2"]
    print(adata.var.index)
    dd = dd[:, :, :, keep_genes]
    assert dd.shape[0] == gdata.shape[0]
    assert dd.shape[1] == gdata.shape[1]
    assert dd.shape[2] == adata.shape[0]
    assert dd.shape[3] == len(keep_genes)


def test_slice_all(adata, gdata):
    dd = DonorData(adata, gdata)
    dd = dd["D0", "SNP1", "C0", "G1"]
    assert dd.shape == (1, 1, 1, 1)


def test_donordata_aggregate(adata, gdata, dummy_covariates):
    previous_adata_shape = adata.shape

    dd = DonorData(adata, gdata)

    dd.aggregate(key_added="Gex")
    assert "Gex" in dd.D.obsm
    assert dd.C.shape == previous_adata_shape

    for celltype in adata.obs[CAnn.celltype].unique():
        celltype_key = f"Gex_{celltype}"
        dd.aggregate(key_added=celltype_key, filter_key=CAnn.celltype, filter_value=celltype)
        assert celltype_key in dd.D.obsm
        assert dd.C.shape == previous_adata_shape

    dd.aggregate(obs=dummy_covariates, func="first", add_to_obs=True)
    assert all(col in dd.D.obs.columns for col in dummy_covariates)
