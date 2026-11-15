from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external._seismic import run_seismic


def _make_adata(var_names, n_cells=60):
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n_cells, len(var_names))).astype(np.float32)
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B"], size=n_cells)}, index=[f"c{i}" for i in range(n_cells)])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.uns["log1p"] = {}
    return adata


def test_run_seismic_raises_before_invoking_r_on_gene_id_mismatch(tmp_path):
    # expression uses HGNC symbols, MAGMA uses Ensembl IDs
    adata = _make_adata([f"GENE{i}" for i in range(300)])
    magma_file = tmp_path / "magma.genes.out"
    magma_df = pd.DataFrame({"GENE": [f"ENSG{i:011d}" for i in range(300)], "ZSTAT": np.random.default_rng(0).normal(size=300)})
    magma_df.to_csv(magma_file, sep="\t", index=False)

    with pytest.raises(ValueError, match="genes shared"):
        run_seismic(adata, magma_file=str(magma_file), cell_type_col="cell_type", species="human")


def test_run_seismic_missing_magma_gene_col_raises(tmp_path):
    adata = _make_adata([f"GENE{i}" for i in range(10)])
    magma_file = tmp_path / "magma.genes.out"
    pd.DataFrame({"NOT_GENE": ["a", "b"], "ZSTAT": [0.1, 0.2]}).to_csv(magma_file, sep="\t", index=False)

    with pytest.raises(ValueError, match="GENE"):
        run_seismic(adata, magma_file=str(magma_file), cell_type_col="cell_type", species="human")


def test_run_seismic_has_no_dead_n_pcs_parameter():
    import inspect

    params = inspect.signature(run_seismic).parameters
    assert "n_pcs" not in params


def test_run_seismic_mouse_all_uppercase_symbols_raise_before_r(tmp_path):
    var_names = ["TRP53", "CD8A", "ACTB", "GAPDH", "MYC", "TP53", "IL6", "TNF", "CD4", "CD3E"]
    adata = _make_adata(var_names)

    with pytest.raises(ValueError, match="ALL-UPPERCASE"):
        run_seismic(
            adata,
            magma_file="/dev/null",
            cell_type_col="cell_type",
            species="mouse",
            mouse_gene_id_type="symbol",
            min_genes=1,
            min_cells=1,
        )


def test_run_seismic_mouse_real_mixed_case_symbols_pass_the_case_check(tmp_path):
    var_names = ["Trp53", "Cd8a", "Actb", "Gapdh", "Myc", "Tp53", "Il6", "Tnf", "Cd4", "Cd3e"]
    adata = _make_adata(var_names)

    try:
        run_seismic(
            adata,
            magma_file="/dev/null",
            cell_type_col="cell_type",
            species="mouse",
            mouse_gene_id_type="symbol",
            min_genes=1,
            min_cells=1,
        )
    except ValueError as e:
        assert "ALL-UPPERCASE" not in str(e)
    except Exception:
        pass 
