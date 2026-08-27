from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

pytest.importorskip("torch", reason="run_seismic_torch tests need torch")

from cellink.tl.external._seismic_torch import run_seismic_torch  # noqa: E402


def _make_adata(n_genes=300, n_cells=120, seed=0):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    var_names = [str(1000 + i) for i in range(n_genes)]  # Entrez-style numeric-string IDs
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B"], size=n_cells)}, index=[f"c{i}" for i in range(n_cells)])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.uns["log1p"] = {}
    return adata


def _make_magma(n_genes=300, int_gene_col=True):
    genes = list(range(1000, 1000 + n_genes)) if int_gene_col else [str(1000 + i) for i in range(n_genes)]
    rng = np.random.default_rng(1)
    return pd.DataFrame({"GENE": genes, "ZSTAT": rng.normal(size=n_genes)})


def test_run_seismic_torch_matches_int64_magma_gene_ids(tmp_path):
    adata = _make_adata()
    magma_file = tmp_path / "magma.genes.out"
    _make_magma(int_gene_col=True).to_csv(magma_file, sep="\t", index=False)

    result = run_seismic_torch(adata, magma_file=str(magma_file), cell_type_col="cell_type", min_genes=1, min_cells=1, save_results=False)
    assert len(result) == 2
    assert not result["pvalue"].isna().any()


def test_run_seismic_torch_raises_on_duplicated_magma_gene_ids(tmp_path):
    adata = _make_adata()
    magma_file = tmp_path / "magma.genes.out"
    df = _make_magma(int_gene_col=True)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate one GENE id
    df.to_csv(magma_file, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicated"):
        run_seismic_torch(adata, magma_file=str(magma_file), cell_type_col="cell_type", min_genes=1, min_cells=1, save_results=False)


def test_run_seismic_torch_raises_on_duplicated_var_names(tmp_path):
    adata = _make_adata()
    adata.var_names = pd.Index(["1000"] * adata.n_vars)  # force duplicates
    magma_file = tmp_path / "magma.genes.out"
    _make_magma(int_gene_col=True).to_csv(magma_file, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicated"):
        run_seismic_torch(adata, magma_file=str(magma_file), cell_type_col="cell_type", min_genes=1, min_cells=1, save_results=False)


def test_run_seismic_torch_drops_nan_zstat_without_propagating_nan_pvalues(tmp_path):
    adata = _make_adata()
    magma_file = tmp_path / "magma.genes.out"
    df = _make_magma(int_gene_col=True)
    df.loc[0, "ZSTAT"] = np.nan
    df.to_csv(magma_file, sep="\t", index=False)

    result = run_seismic_torch(adata, magma_file=str(magma_file), cell_type_col="cell_type", min_genes=1, min_cells=1, save_results=False)
    assert not result["pvalue"].isna().any(), "a single NaN z-score must not propagate to every cell type's p-value"
