import numpy as np
import pandas as pd
import pytest

import cellink as cl
from cellink import DonorData
from cellink._core.data_fields import CAnn, DAnn
from cellink._core.dummy_data import sim_adata
from cellink.at.fetch import fetch_slot


@pytest.fixture
def dd_table():
    adata = sim_adata()
    donors = adata.obs[DAnn.donor].unique()
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "age": rng.integers(20, 80, len(donors)).astype(float),
            "sex": pd.Categorical(np.resize(["F", "M"], len(donors))),
            "phenotype": rng.standard_normal(len(donors)),
        },
        index=pd.Index(donors, name=DAnn.donor),
    )
    return DonorData(D=table, C=adata)


def test_cellink_reexports_donordata():
    import donordata

    from cellink._core.donordata import DonorData as LegacyDonorData

    assert cl.DonorData is donordata.DonorData
    assert LegacyDonorData is donordata.DonorData
    assert cl.io.read_dd is donordata.read_dd
    assert cl.pp.donor_level_obs_filter is donordata.pp.donor_level_obs_filter
    assert DAnn is donordata.DAnn and CAnn is donordata.CAnn
    for name in ("read_dd", "read_h5_dd", "read_zarr_dd"):
        assert getattr(cl.io, name) is getattr(donordata.io, name)
        assert not hasattr(cl, name)
    for name in ("concat", "register_donordata_namespace"):
        assert getattr(cl, name) is getattr(donordata, name)


def test_cellink_reexports_mil_dataset():
    pytest.importorskip("torch")
    import donordata.ml

    assert cl.ml.MILDataset is donordata.ml.MILDataset
    assert cl.ml.mil_collate_fn is donordata.ml.mil_collate_fn


def test_validate_skips_variant_schema_for_table(dd_table):
    assert dd_table.D_type == "table"
    assert cl.validate(dd_table)


def test_pp_filters_on_table(dd_table):
    out = cl.pp.donor_level_obs_filter(dd_table, "sex", "F", inplace=False)
    assert set(out.obs["sex"]) == {"F"}
    assert set(out.C.obs[DAnn.donor]) == set(out.obs_names)
    cl.pp.cell_level_obs_filter(dd_table, CAnn.celltype, ["CD4", "CD8"])
    assert set(dd_table.C.obs[CAnn.celltype]) <= {"CD4", "CD8"}


def test_io_roundtrip_table(tmp_path, dd_table):
    dd_table.aggregate(key_added="PB", sync_var=True)
    path = tmp_path / "table.dd.h5"
    dd_table.write_dd(str(path))
    loaded = cl.io.read_dd(str(path))
    assert loaded.D_type == "table"
    pd.testing.assert_frame_equal(loaded.obs, dd_table.obs)
    pd.testing.assert_frame_equal(loaded.obsm["PB"], dd_table.obsm["PB"], check_names=False)


def test_fetch_slot_reads_table_and_obsm(dd_table):
    dd_table.obsm["X_emb"] = pd.DataFrame(
        np.arange(dd_table.n_donors * 2.0).reshape(-1, 2), index=dd_table.obs_names, columns=["emb_a", "emb_b"]
    )
    age = fetch_slot(dd_table, "age")
    assert age.index.equals(dd_table.obs_names)
    np.testing.assert_allclose(age.to_numpy(), dd_table.obs["age"].to_numpy())
    np.testing.assert_allclose(fetch_slot(dd_table, "emb_b").to_numpy(), dd_table.obsm["X_emb"]["emb_b"].to_numpy())


def test_model_matrix_from_table(dd_table):
    pytest.importorskip("formulaic")
    from cellink.at.resolver import get_model_matrix

    y = get_model_matrix(dd_table, "phenotype - 1", target_level="donor")
    np.testing.assert_allclose(y.to_numpy().ravel(), dd_table.obs["phenotype"].to_numpy())
    X = get_model_matrix(dd_table, "age + dmean(G0)", target_level="donor")
    assert X.shape == (dd_table.n_donors, 3)
    crepeat = get_model_matrix(dd_table, "crepeat(age)", target_level="cell")
    expected = dd_table.obs["age"].to_numpy()[dd_table.donor_codes]
    np.testing.assert_allclose(crepeat["crepeat(age)"].to_numpy(), expected)


@pytest.mark.slow
def test_mil_model_trains_on_table_with_donor_embedding(dd_table):
    pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader

    from cellink.ml import DonorMILModel, MILDataset, mil_collate_fn

    dd_table.aggregate(key_added="PB")
    dataset = MILDataset(dd_table, donor_layer="PB", donor_labels_key="phenotype", cell_batch_key="cov1")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=mil_collate_fn)
    model = DonorMILModel(n_input_donor=dd_table.obsm["PB"].shape[1], n_input_cell=dd_table.C.n_vars)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model, loader)
    assert trainer.state.finished
