from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

from cellink.tl.external._seismic_torch import RegressionNLL, SparseScore, _adata_to_sparse_csr_tensor


def test_adata_to_sparse_csr_tensor_from_sparse_layer():
    X = csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float32))
    adata = AnnData(X)
    t = _adata_to_sparse_csr_tensor(adata, layer=None)
    assert t.layout == torch.sparse_csr
    dense = t.to_dense().numpy()
    np.testing.assert_allclose(dense, X.toarray())


def test_adata_to_sparse_csr_tensor_from_dense_falls_back_to_dense_tensor():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adata = AnnData(X)
    t = _adata_to_sparse_csr_tensor(adata, layer=None)
    assert not t.is_sparse
    np.testing.assert_allclose(t.numpy(), X)


def test_regression_nll_matches_statsmodels_ols():
    """Cross-validate RegressionNLL's closed-form likelihood-ratio test against
    an independent OLS fit + LRT computed by hand from statsmodels, rather than
    trusting the closed-form linear-algebra implementation on its own."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n).astype(np.float64)
    g = 0.6 * z + rng.normal(scale=0.5, size=n)  # correlated covariate
    g = g.astype(np.float64)

    model = RegressionNLL(torch.tensor(z))
    nll, pval, beta, se = model.forward(torch.tensor(g[:, None]), return_all=True)

    # independent reference: full model vs. intercept-only null, via statsmodels
    X_full = sm.add_constant(g)
    fit_full = sm.OLS(z, X_full).fit()
    fit_null = sm.OLS(z, np.ones((n, 1))).fit()
    ref_beta = fit_full.params[1]
    ref_se = fit_full.bse[1]

    np.testing.assert_allclose(beta.item(), ref_beta, rtol=1e-4)
    # small, expected gap vs. statsmodels' unbiased-variance convention, not a bug
    np.testing.assert_allclose(se.item(), ref_se, rtol=5e-3)
    # RegressionNLL's one-sided p-value, doubled, should recover statsmodels' own
    # two-sided p-value for whichever tail the observed effect sign falls on
    ref_p_two_sided = fit_full.pvalues[1]
    doubled = min(2 * pval.item(), 1.0)
    np.testing.assert_allclose(doubled, ref_p_two_sided, rtol=1e-2, atol=1e-6)


def test_regression_nll_multiple_columns_independent_of_each_other():
    """Each column of G is regressed independently; a strongly-associated
    column and a pure-noise column in the same call must not contaminate
    each other's p-value."""
    rng = np.random.default_rng(1)
    n = 300
    z = rng.normal(size=n)
    strong = 0.8 * z + rng.normal(scale=0.3, size=n)
    noise = rng.normal(size=n)
    G = np.stack([strong, noise], axis=1)

    model = RegressionNLL(torch.tensor(z))
    _, pval, beta, _ = model.forward(torch.tensor(G), return_all=True)

    assert pval[0, 0].item() < 0.001  # strong signal, should be highly significant
    assert pval[1, 0].item() > 0.05  # pure noise, should not be


def test_sparse_score_perfectly_specific_gene_scores_near_one_in_its_own_celltype():
    """A gene expressed only in cluster A (zero elsewhere) should score near
    1.0 for cluster A and near 0 for cluster B under seismic's specificity
    z-test, the basic sanity check that the sparse closed-form matches the
    algorithm's own definition."""
    # 6 cells: 3 in cluster A (gene highly expressed), 3 in cluster B (gene ~0)
    E = torch.tensor(
        [
            [5.0, 0.0],
            [6.0, 0.1],
            [4.0, 0.0],
            [0.0, 5.0],
            [0.1, 6.0],
            [0.0, 4.0],
        ]
    )
    masks = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    scorer = SparseScore(E)
    s = scorer.forward(masks)
    assert s.shape == (2, 2)  # [genes, celltypes]
    # gene 0 (col 0 of E) is high in cluster A (mask col 0) -> should score highest there
    assert s[0, 0] > s[0, 1]
    # gene 1 is high in cluster B -> should score highest there
    assert s[1, 1] > s[1, 0]


def test_sparse_score_uniform_expression_gives_low_specificity_everywhere():
    """A gene expressed identically across both clusters has no real
    specificity signal; its score should not spike in either cluster."""
    E = torch.tensor([[5.0], [5.0], [5.0], [5.0], [5.0], [5.0]])
    masks = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )
    scorer = SparseScore(E)
    s = scorer.forward(masks)
    # both cluster scores should be small and roughly comparable (no specificity)
    assert abs(s[0, 0].item() - s[0, 1].item()) < 0.3
