from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from cellink import DonorData, DonorDataSchemaError, validate
from cellink._core.data_fields import CAnn, DAnn
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.io import read_zarr_dd

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)  # mudata <0.4
else:
    md.settings.pull_on_update = False  # mudata >=0.4 (Python >=3.12): set_options -> settings

DATA = Path("tests/data")


def test_donordata_init(adata, gdata):
    DonorData(G=gdata, C=adata)


def test_donordata_keeps_cell_order_within_donors(adata, gdata):
    """Cells are grouped in G's donor order; within a donor they keep their input order."""
    shuffled = adata[np.random.default_rng(0).permutation(adata.n_obs)].copy()
    dd = DonorData(G=gdata, C=shuffled)
    donors = dd.C.obs[DAnn.donor].astype(str).to_numpy()
    rank = pd.Index(dd.G.obs_names).get_indexer(donors)
    assert (np.diff(rank) >= 0).all()
    for donor in dd.G.obs_names:
        expected = shuffled.obs_names[shuffled.obs[DAnn.donor].astype(str) == donor]
        assert list(dd.C.obs_names[donors == donor]) == list(expected)


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


def test_copy_returns_independent_object(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    dd_copy = dd.copy()
    assert dd_copy is not dd
    assert dd_copy.G is not dd.G
    assert dd_copy.C is not dd.C
    dd_copy.uns["marker"] = "mutated_via_copy"
    dd_copy.G.obs["new_col"] = 0
    assert "marker" not in dd.uns
    assert "new_col" not in dd.G.obs


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


def test_donordata_aggregate_first_add_to_obs(adata, gdata, dummy_covariates):
    dd = DonorData(G=gdata, C=adata).copy()

    dd.C.obs[dummy_covariates[0]] = dd.C.obs[dummy_covariates[0]].astype("category")
    dd.C.obs[dummy_covariates[1]] = dd.C.obs[dummy_covariates[1]].astype("category").cat.codes
    dd.aggregate(obs=dummy_covariates, func="first", add_to_obs=True)


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


@pytest.mark.slow
def test_write_h5_dd(tmp_path, adata, gdata):
    output_path = tmp_path / "donordata.dd.h5"

    dd = DonorData(G=gdata, C=adata)

    dd.write_h5_dd(str(output_path))

    assert output_path.exists()


@pytest.mark.slow
def test_write_zarr_dd(tmp_path, adata, gdata):
    output_path = tmp_path / "donordata.dd.zarr"

    dd = DonorData(G=gdata, C=adata)

    dd.write_zarr_dd(str(output_path))

    assert output_path.exists()


def test_write_zarr_dd_dense_x_keeps_layers(tmp_path, adata, gdata):
    # anndata >= 0.13 lists X as layers[None]; the chunked dense-X writer must not store it again
    gdata.X = np.asarray(gdata.X.todense() if hasattr(gdata.X, "todense") else gdata.X, dtype=np.float32)
    gdata.layers["dosage"] = gdata.X + 1
    output_path = tmp_path / "donordata.dd.zarr"
    DonorData(G=gdata, C=adata).write_zarr_dd(str(output_path))

    reloaded = read_zarr_dd(str(output_path))
    assert [k for k in reloaded.G.layers.keys() if k is not None] == ["dosage"]
    np.testing.assert_array_equal(np.asarray(reloaded.G.X), gdata.X)
    np.testing.assert_array_equal(np.asarray(reloaded.G.layers["dosage"]), gdata.X + 1)


def _small_dd():
    G = ad.AnnData(X=np.arange(4, dtype=float).reshape(2, 2), obs=pd.DataFrame(index=["a", "b"]))
    G.obs.index.name = "donor_id"
    C = ad.AnnData(X=np.ones((3, 2)), obs=pd.DataFrame({"donor_id": ["b", "a", "b"]}, index=["c0", "c1", "c2"]))
    return C, DonorData(G=G, C=C)


def test_sel_and_getitem_align_boolean_series_by_label():
    C, dd = _small_dd()
    assert list(dd.C.obs_names) == ["c1", "c0", "c2"]  # the constructor sorts cells by donor
    keep = pd.Series([True, False, False], index=C.obs_names)  # built on the AnnData passed in
    assert list(dd.sel(C_obs=keep).C.obs_names) == ["c0"]
    assert list(dd[:, :, keep].C.obs_names) == ["c0"]


def test_boolean_series_aligned_to_dd_and_numpy_masks_unchanged():
    _, dd = _small_dd()
    mask = dd.C.obs["donor_id"] == "a"
    assert list(dd.sel(C_obs=mask).C.obs_names) == ["c1"]
    assert list(dd[:, :, mask.to_numpy()].C.obs_names) == ["c1"]
    donor_mask = pd.Series([False, True], index=["a", "b"])
    assert list(dd.sel(G_obs=donor_mask).G.obs_names) == ["b"]


def test_boolean_series_missing_labels_raises():
    _, dd = _small_dd()
    with pytest.raises(KeyError, match="cells"):
        dd.sel(C_obs=pd.Series([True, False], index=["c0", "c1"]))


def test_validate_mudata_donor_level_checks_genotype_modality():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    dd = DonorData(G=md.MuData({"genotype": gdata}), C=adata)
    assert validate(dd)

    gdata_bad = gdata.copy()
    gdata_bad.var = gdata_bad.var.drop(columns=["a1"])
    with pytest.raises(DonorDataSchemaError, match="genotype"):
        validate(DonorData(G=md.MuData({"genotype": gdata_bad}), C=adata))

    gdata_none = gdata.copy()
    gdata_none.var = gdata_none.var.drop(columns=["chrom"])
    with pytest.raises(DonorDataSchemaError, match="No modality"):
        validate(DonorData(G=md.MuData({"genotype": gdata_none}), C=adata))


if __name__ == "__main__":
    import pytest

    pytest.main()
