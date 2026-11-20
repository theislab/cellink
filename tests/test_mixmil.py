import numpy as np
import pytest

pytest.importorskip("torch")

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.tl.external._mixmil import _bags


def test_bags_follow_the_donor_order_of_d():
    adata = sim_adata()
    adata.obsm["X_pca"] = np.random.default_rng(0).standard_normal((adata.n_obs, 4))
    gdata = sim_gdata(adata=adata)
    dd = DonorData(D=gdata[gdata.obs_names[::-1]].copy(), C=adata)
    bags = _bags(dd, "X_pca", "float32")
    assert len(bags) == dd.n_donors
    for donor, bag in zip(dd.obs_names, bags, strict=True):
        expected = adata.obsm["X_pca"][(adata.obs["donor_id"] == donor).to_numpy()]
        np.testing.assert_allclose(bag.numpy(), expected.astype(np.float32))
