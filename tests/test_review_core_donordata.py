from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from cellink import DonorData, DonorDataSchemaError, validate
from cellink._core.dummy_data import sim_adata, sim_gdata


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
