from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from cellink._core.dummy_data import sim_adata
from cellink.at.fetch import fetch_slot


def test_fetch_slot_obs_column():
    adata = sim_adata()
    adata.obs["pheno"] = np.arange(adata.n_obs)
    result = fetch_slot(adata, "pheno")
    assert result.attrs["axis"] == "obs"
    np.testing.assert_array_equal(result.to_numpy(), adata.obs["pheno"].to_numpy())


def test_fetch_slot_var_column():
    adata = sim_adata()
    adata.var["length"] = np.arange(adata.n_vars)
    result = fetch_slot(adata, "length", axis="var")
    assert result.attrs["axis"] == "var"
    np.testing.assert_array_equal(result.to_numpy(), adata.var["length"].to_numpy())


def test_fetch_slot_gene_expression_from_x():
    adata = sim_adata()
    gene = adata.var_names[0]
    result = fetch_slot(adata, gene)
    np.testing.assert_allclose(result.to_numpy(), np.asarray(adata[:, gene].X).ravel())


def test_fetch_slot_layer():
    adata = sim_adata()
    adata.layers["scaled"] = adata.X * 2
    gene = adata.var_names[0]
    result = fetch_slot(adata, gene, layer="scaled")
    np.testing.assert_allclose(result.to_numpy(), np.asarray(adata[:, gene].layers["scaled"]).ravel())


def test_fetch_slot_obsm_whole_embedding_array():
    adata = sim_adata()
    rng = np.random.default_rng(0)
    adata.obsm["X_pca"] = rng.standard_normal((adata.n_obs, 3))
    result = fetch_slot(adata, "X_pca")
    assert result.shape == (adata.n_obs, 3)
    assert list(result.columns) == ["X_pca_1", "X_pca_2", "X_pca_3"]
    np.testing.assert_allclose(result.to_numpy(), adata.obsm["X_pca"])


def test_fetch_slot_obsm_dataframe_embedding_and_named_column():
    adata = sim_adata()
    adata.obsm["emb"] = pd.DataFrame({"a": np.arange(adata.n_obs), "b": np.arange(adata.n_obs) * 2}, index=adata.obs_names)
    whole = fetch_slot(adata, "emb")
    np.testing.assert_array_equal(whole["a"].to_numpy(), np.arange(adata.n_obs))

    single_col = fetch_slot(adata, "a")
    np.testing.assert_array_equal(single_col.to_numpy(), np.arange(adata.n_obs))


def test_fetch_slot_indexed_embedding_column():
    adata = sim_adata()
    adata.obsm["X_pca"] = np.arange(adata.n_obs * 3).reshape(adata.n_obs, 3).astype(float)
    result = fetch_slot(adata, "X_pca_2")
    np.testing.assert_allclose(result.to_numpy(), adata.obsm["X_pca"][:, 2])


def test_fetch_slot_key_not_found_raises_keyerror():
    adata = sim_adata()
    with pytest.raises(KeyError, match="not found"):
        fetch_slot(adata, "nonexistent_key_xyz")


def test_fetch_slot_ambiguous_key_raises_valueerror():
    adata = sim_adata()
    adata.obs["dup"] = 1
    adata.obsm["dupemb"] = pd.DataFrame({"dup": np.arange(adata.n_obs)}, index=adata.obs_names)
    with pytest.raises(ValueError, match="is not unique"):
        fetch_slot(adata, "dup")

    result = fetch_slot(adata, "dup", raise_on_multiple=False)
    assert result.attrs["axis"] == "obs"


def test_fetch_slot_plain_mudata():
    n = 5
    rna = ad.AnnData(X=np.zeros((n, 2)), obs=pd.DataFrame({"pheno": np.arange(n)}, index=[f"c{i}" for i in range(n)]))
    atac = ad.AnnData(X=np.zeros((n, 2)), obs=pd.DataFrame(index=rna.obs_names))
    mdata = md.MuData({"rna": rna, "atac": atac})

    result = fetch_slot(mdata, "pheno")
    np.testing.assert_array_equal(result.to_numpy(), np.arange(n))
