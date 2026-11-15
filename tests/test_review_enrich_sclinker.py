from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from cellink.tl.external import _sclinker_utils
from cellink.tl.external._sclinker import (
    _compute_nmf_gene_correlations,
    compute_joint_nmf_programs,
    compute_nmf_programs,
)
from cellink.tl.external._sclinker_utils import (
    _merge_bedgraph_bedtools,
    _merge_bedgraph_python,
    genescores_to_100kb_bedgraph,
)

TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


def _bg():
    return pd.DataFrame(
        {
            "chr": ["chr1", "chr1", "chr1", "chr2"],
            "start": [0, 50, 500, 0],
            "end": [100, 150, 600, 10],
            "score": [0.5, 0.7, 0.2, 0.4],
        }
    )


def test_python_merge_keeps_max_score():
    merged = _merge_bedgraph_python(_bg())
    assert merged.to_dict("list") == {
        "chr": ["chr1", "chr1", "chr2"],
        "start": [0, 500, 0],
        "end": [150, 600, 10],
        "score": [0.7, 0.2, 0.4],
    }


def test_bedtools_merge_uses_max(monkeypatch):
    seen = []

    def fake_run(cmd, input=None, capture_output=True, text=True, check=True):
        seen.append(cmd)
        out = "" if cmd[1] == "sort" else "chr1\t0\t150\t0.7\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(_sclinker_utils.subprocess, "run", fake_run)
    merged = _merge_bedgraph_bedtools(_bg())
    assert seen[1][-2:] == ["-o", "max"]
    assert merged["score"].tolist() == [0.7]


def test_100kb_overlapping_windows_keep_max_score():
    genescores = pd.DataFrame({"prog": [0.3, 0.7]}, index=["GENE_A", "GENE_B"])
    annotation = pd.DataFrame(
        {"GENE": ["GENE_A", "GENE_B"], "CHR": ["1", "1"], "START": [500_000, 550_000], "END": [510_000, 560_000]}
    )
    bgs = genescores_to_100kb_bedgraph(genescores, annotation, use_bedtools_for_merge=False)
    assert len(bgs["prog"]) == 1
    assert np.isclose(bgs["prog"]["score"].max(), 0.7)


def _adata(n_cells, n_genes, seed):
    rng = np.random.default_rng(seed)
    return AnnData(
        X=rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{seed}_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"G{j}" for j in range(n_genes)]),
    )


def test_joint_nmf_accepts_unequal_cell_counts_and_recovers_shared_gene_programs():
    # genes x cells data with two gene programs shared by both conditions and one specific to each
    rng = np.random.default_rng(0)
    shared = rng.gamma(0.5, 1, size=(60, 2))
    Xh = np.hstack([shared, rng.gamma(0.5, 1, size=(60, 1))]) @ rng.gamma(1, 1, size=(3, 40))
    Xd = np.hstack([shared, rng.gamma(0.5, 1, size=(60, 1))]) @ rng.gamma(1, 1, size=(3, 35))

    def to_adata(X, prefix):
        obs = pd.DataFrame(index=[f"{prefix}{i}" for i in range(X.shape[1])])
        return AnnData(X=X.T.astype(np.float32), obs=obs, var=pd.DataFrame(index=[f"G{j}" for j in range(60)]))

    healthy, disease = to_adata(Xh, "h"), to_adata(Xd, "d")
    res = compute_joint_nmf_programs(
        healthy, disease, n_shared=2, n_healthy_specific=1, n_disease_specific=1, save=False
    )
    assert res["Wh"].shape == (40, 3) and list(res["Wh"].index) == list(healthy.obs_names)
    assert res["Wd"].shape == (35, 3) and list(res["Wd"].index) == list(disease.obs_names)
    assert res["Hh"].shape == (60, 3) and list(res["Hh"].index) == list(healthy.var_names)
    assert res["shared_Wh"].shape == (40, 2) and res["unique_Hd"].shape == (60, 1)
    for i in range(2):
        assert np.corrcoef(res["Hh"].iloc[:, i], res["Hd"].iloc[:, i])[0, 1] > 0.99
        assert max(abs(np.corrcoef(res["Hh"].iloc[:, i], shared[:, j])[0, 1]) for j in range(2)) > 0.99
    assert abs(np.corrcoef(res["Hh"].iloc[:, 2], res["Hd"].iloc[:, 2])[0, 1]) < 0.5


def test_nmf_gene_correlations_handle_all_zero_genes():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(50, 6)).astype(np.float32)
    X[:, 2] = 0
    X[:, 5] = 0  # all-zero last gene
    W = rng.random((50, 2))
    corr = _compute_nmf_gene_correlations(X, W, [f"g{i}" for i in range(6)], ["a", "b"]).to_numpy()
    for g in range(6):
        for k in range(2):
            expected = 0.0 if X[:, g].std() == 0 else np.corrcoef(X[:, g], W[:, k])[0, 1]
            assert np.isclose(corr[g, k], expected, atol=1e-6)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_nmf_min_cells_drops_rare_genes_float32_and_leaves_adata_alone(monkeypatch, as_sparse):
    monkeypatch.setitem(sys.modules, "torchnmf", None)  # sklearn backend, quick and deterministic
    monkeypatch.setitem(sys.modules, "torchnmf.nmf", None)
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(60, 12)).astype(np.float64)
    X[:, 3] = 0
    X[:3, 7], X[3:, 7] = 1, 0  # detected in 3 cells
    adata = AnnData(
        X=sparse.csr_matrix(X) if as_sparse else X.copy(),
        obs=pd.DataFrame({"ct": ["a", "b"] * 30}, index=[f"c{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"G{j}" for j in range(12)]),
    )
    W, H, corr = compute_nmf_programs(adata, n_components=3, celltype_col="ct", layer=None, min_cells=5, save=False)
    kept = [f"G{j}" for j in range(12) if j not in (3, 7)]
    assert list(H.index) == kept and list(corr.index) == kept
    assert W.shape == (60, 3)
    X_after = adata.X.toarray() if as_sparse else adata.X
    np.testing.assert_array_equal(X_after, X)  # not normalised in place
    assert adata.n_vars == 12

    _, H_all, corr_all = compute_nmf_programs(adata, n_components=3, celltype_col="ct", layer=None, save=False)
    assert len(H_all) == 12 and np.allclose(corr_all.loc["G3"], 0.0)


def test_sclinker_tutorial_imports_resolve():
    import ast
    import importlib

    nb = json.loads((TUTORIALS / "sclinker.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line for line in "".join(cell["source"]).splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cellink"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"


def test_public_exports():
    import cellink.tl.external as ext

    for name in ("download_magma_references", "get_gene_annotation", "run_scprs"):
        assert name in ext.__all__ and callable(getattr(ext, name))
