from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from cellink.tl.external._sclinker import (
    _compute_nmf_gene_correlations,
    compute_joint_nmf_programs,
    compute_nmf_programs,
)
from cellink.tl.external._sclinker_utils import (
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


def _nmf_adata(as_sparse=False, seed=0):
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.poisson(2.0, size=(60, 12)).astype(np.float64))
    X[:, 3] = 0
    X[:3, 7], X[3:, 7] = 1, 0
    adata = AnnData(
        X=sparse.csr_matrix(X) if as_sparse else X.copy(),
        obs=pd.DataFrame({"ct": ["a", "b"] * 30}, index=[f"c{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"G{j}" for j in range(12)]),
    )
    adata.layers["counts"] = np.expm1(X)
    return adata, X


@pytest.mark.parametrize("as_sparse", [False, True])
def test_nmf_fits_sklearn_random_init_on_log_normalized_expression(as_sparse):
    from sklearn.decomposition import NMF

    adata, X = _nmf_adata(as_sparse)
    W, H, _ = compute_nmf_programs(adata, n_components=3, celltype_col="ct", collapse_threshold=None, save=False)
    model = NMF(n_components=3, init="random", random_state=0)
    np.testing.assert_allclose(W.to_numpy(), model.fit_transform(X / X.max()))
    np.testing.assert_allclose(H.to_numpy(), model.components_.T)
    np.testing.assert_array_equal(adata.X.toarray() if as_sparse else adata.X, X)


def test_nmf_correlations_cover_every_expressed_gene():
    adata, X = _nmf_adata()
    W, _, corr = compute_nmf_programs(adata, n_components=3, celltype_col="ct", collapse_threshold=None, save=False)
    assert list(corr.index) == [f"G{j}" for j in range(12) if j != 3]
    for gene in ("G0", "G7"):
        j = int(gene[1:])
        expected = [np.corrcoef(X[:, j], W[f])[0, 1] for f in W.columns]
        np.testing.assert_allclose(corr.loc[gene].to_numpy(), expected, atol=1e-12)


def test_nmf_fit_genes_come_from_highly_variable_and_min_cells():
    adata, _ = _nmf_adata()
    adata.var["highly_variable"] = [j % 2 == 1 for j in range(12)]
    _, H, corr = compute_nmf_programs(adata, n_components=2, celltype_col="ct", save=False)
    assert list(H.index) == [f"G{j}" for j in range(1, 12, 2)]
    assert len(corr) == 11

    _, H, _ = compute_nmf_programs(
        adata, n_components=2, celltype_col="ct", use_highly_variable=False, min_cells=5, save=False
    )
    assert list(H.index) == [f"G{j}" for j in range(12) if j not in (3, 7)]
    with pytest.raises(ValueError, match="highly_variable"):
        compute_nmf_programs(_nmf_adata()[0], n_components=2, use_highly_variable=True, save=False)


def test_correlated_factors_are_collapsed():
    from cellink.tl.external._sclinker import _collapse_correlated_factors

    rng = np.random.default_rng(0)
    a, b = rng.random(100), rng.random(100)
    W = np.column_stack([a, b, 2 * a + 0.01 * rng.random(100), np.ones(100)])
    assert _collapse_correlated_factors(W, 0.8) == [0, 1, 3]
    assert _collapse_correlated_factors(W, None) == [0, 1, 2, 3]


def test_nmf_genescores_clip_negative_correlations_and_scale_to_one(tmp_path):
    adata, _ = _nmf_adata()
    _, _, corr = compute_nmf_programs(adata, n_components=3, celltype_col="ct", out_dir=tmp_path, save=True)
    scores = pd.read_csv(tmp_path / "nmf_genescores.csv", index_col=0)
    expected = corr.clip(lower=0)
    pd.testing.assert_frame_equal(scores, expected / expected.max(), check_names=False)
    assert (scores.max() == 1).all()


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
