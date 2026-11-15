from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.at.fetch import fetch_slot

pytest.importorskip("formulaic")
from cellink.at.resolver import get_model_matrix


@pytest.fixture
def adata():
    X = np.arange(6, dtype=float).reshape(3, 2)
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1", "c2"]), var=pd.DataFrame(index=["g0", "g1"]))
    a.layers["counts"] = X * 10
    return a


@pytest.fixture
def dd():
    donordata = DonorData(G=sim_gdata(), C=sim_adata())
    donordata.G.obs["age"] = np.arange(donordata.G.n_obs, dtype=float)
    return donordata


# fetch_slot


def test_fetch_slot_missing_layer_raises(adata):
    with pytest.raises(KeyError, match="countz"):
        fetch_slot(adata, "g0", layer="countz")
    np.testing.assert_allclose(fetch_slot(adata, "g0", layer="counts").to_numpy(), [0.0, 20.0, 40.0])
    np.testing.assert_allclose(fetch_slot(adata, "g0").to_numpy(), [0.0, 2.0, 4.0])


@pytest.mark.parametrize("to_sparse", [scipy.sparse.csr_matrix, scipy.sparse.csr_array, np.asarray])
def test_fetch_slot_indexed_key_on_any_embedding(adata, to_sparse):
    emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adata.obsm["X_sp"] = to_sparse(emb)
    out = fetch_slot(adata, "X_sp_1")
    assert list(out.index) == ["c0", "c1", "c2"]
    np.testing.assert_allclose(out.to_numpy(), emb[:, 1])


# get_model_matrix


def test_get_model_matrix_rejects_unknown_target_level(dd):
    with pytest.raises(ValueError, match="sample"):
        get_model_matrix(dd, "dmean(G0)", target_level="sample")


def test_get_model_matrix_intercept_only_on_anndata(adata):
    mm = get_model_matrix(adata, "1")
    assert mm.shape == (3, 1)
    assert list(mm.index) == ["c0", "c1", "c2"]
    np.testing.assert_allclose(mm.to_numpy(), 1.0)


def test_get_model_matrix_maps_cells_to_donors_for_dmean_and_crepeat(dd):
    assert isinstance(dd.C.obs[dd.donor_id].dtype, pd.CategoricalDtype)
    donors = dd.C.obs[dd.donor_id].astype(str)

    X = get_model_matrix(dd, "dmean(G0) - 1", target_level="donor")
    expected = pd.Series(np.asarray(dd.C[:, "G0"].X).ravel(), index=dd.C.obs_names).groupby(donors).mean()
    np.testing.assert_allclose(X["dmean(G0)"].to_numpy(), expected.reindex(X.index).to_numpy())

    X = get_model_matrix(dd, "crepeat(age) - 1", target_level="cell")
    assert X["crepeat(age)"].dtype == float
    np.testing.assert_allclose(X["crepeat(age)"].to_numpy(), donors.map(dd.G.obs["age"]).to_numpy())
