from __future__ import annotations

import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cellink.at.acat import acat_test, compute_acat
from cellink.tl import build_pca_informed_prior, coloc_abf, coloc_susie, subset_gene, subset_genomic_region
from cellink.tl._coloc import _combine_log_abf, _log_h3, _logsumexp

PP_COLUMNS = ["idx1", "idx2", "nsnps", "hit1", "hit2", "PP0", "PP1", "PP2", "PP3", "PP4"]


# coloc


def test_coloc_susie_defaults_to_r_coloc_susie_p12():
    assert inspect.signature(coloc_susie).parameters["p12"].default == 5e-6
    lbf1 = np.zeros((1, 30))
    lbf1[0, 5] = 20.0
    lbf2 = np.zeros((1, 30))
    lbf2[0, 5] = 15.0
    res = coloc_susie(lbf1, lbf2, cs1_index=[0], cs2_index=[0])
    expected = _combine_log_abf(lbf1[0], lbf2[0], 1e-4, 1e-4, 5e-6)
    for key in ["PP0", "PP1", "PP2", "PP3", "PP4"]:
        assert res.loc[0, key] == pytest.approx(expected[key])


def test_coloc_susie_without_credible_sets_keeps_columns():
    res = coloc_susie(np.zeros((2, 10)), np.zeros((1, 10)), cs1_index=[], cs2_index=[0])
    assert res.empty
    assert list(res.columns) == PP_COLUMNS


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("which", ["beta1", "se1", "beta2", "se2"])
def test_coloc_abf_rejects_non_finite_input(which, bad):
    args = {name: np.array([0.1, 0.2, 0.3]) for name in ["beta1", "se1", "beta2", "se2"]}
    args[which][1] = bad
    with pytest.raises(ValueError, match=which):
        coloc_abf(**args)


def test_log_h3_matches_direct_formula():
    rng = np.random.default_rng(0)
    l1, l2 = rng.normal(0, 2, 40), rng.normal(0, 2, 40)
    lsum1, lsum2 = _logsumexp(l1), _logsumexp(l2)
    direct = np.log(np.exp(lsum1 + lsum2) - np.exp(_logsumexp(l1 + l2)))
    assert _log_h3(l1, l2, lsum1, lsum2) == pytest.approx(direct, rel=1e-12)


def test_coloc_abf_finite_for_strong_distinct_signals():
    n = 50
    beta1, beta2 = np.full(n, 0.001), np.full(n, 0.001)
    se = np.full(n, 0.01)
    beta1[3] = 0.4  # z = 40 in study 1 only
    beta2[30] = 0.4  # z = 40 in study 2 only, at another SNP
    pp = coloc_abf(beta1, se, beta2, se)
    values = np.array(list(pp.values()))
    assert np.all(np.isfinite(values))
    assert values.sum() == pytest.approx(1.0)
    assert pp["PP3"] > 0.99


# ACAT


def test_acat_ignores_nan_pvalues():
    assert acat_test(np.array([0.01, np.nan, 0.5])) == pytest.approx(acat_test(np.array([0.01, 0.5])))
    assert np.isnan(acat_test(np.array([np.nan, np.nan])))
    assert acat_test(np.array([0.2])) == pytest.approx(0.2)


def test_acat_rejects_non_positive_weights():
    with pytest.raises(AssertionError, match="positive"):
        acat_test(np.array([0.01, 0.5]), weights=np.array([-1.0, 2.0]))
    with pytest.raises(AssertionError, match="positive"):
        acat_test(np.array([0.01, 0.5]), weights=np.array([0.0, 1.0]))


def test_acat_drops_weights_of_nan_pvalues_and_normalises():
    weighted = acat_test(np.array([0.01, np.nan, 0.5]), weights=np.array([2.0, 7.0, 2.0]))
    assert weighted == pytest.approx(acat_test(np.array([0.01, 0.5])))
    rows = compute_acat(np.array([[0.01, np.nan, 0.5]]), weights=np.array([1.0, 1.0, 1.0]))
    assert rows[0] == pytest.approx(acat_test(np.array([0.01, 0.5])))


# SKAT weights


def test_run_skat_test_applies_maf_beta_weights(monkeypatch):
    from cellink.tl import _rvat

    seen = {}

    def fake_skat(Y, X, F):
        seen["X"] = X
        return np.array([[0.5]])

    monkeypatch.setattr(_rvat, "_skat_test", fake_skat)
    rng = np.random.default_rng(0)
    X = rng.binomial(2, [0.02, 0.3, 0.9], size=(200, 3)).astype(float)
    out = _rvat.run_skat_test(ad.AnnData(X=X), rng.normal(size=(200, 1)), np.ones((200, 1)), "gene")

    allele_freq = X.mean(0) / 2
    maf = np.minimum(allele_freq, 1 - allele_freq)
    np.testing.assert_allclose(seen["X"], X * np.sqrt(stats.beta.pdf(maf, 1, 25)))
    assert out["weight_col"] == "maf_beta"
    assert out["pv"][0, 0] == 0.5


# PCA-informed prior


def test_pca_prior_missing_scores_count_as_no_evidence():
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.normal(5, 1, size=(50, 3)), columns=["t1", "t2", "t3"])
    scores.iloc[0] = np.nan
    prior = build_pca_informed_prior(scores, {c: "A" for c in scores.columns})
    assert prior.sum() == pytest.approx(1.0)
    assert prior.iloc[0] < prior.median()


def test_pca_prior_single_variant_and_constant_tracks_are_uniform():
    single = build_pca_informed_prior(pd.DataFrame({"t1": [0.3]}, index=["v"]), {"t1": "A"})
    assert single.tolist() == [1.0]
    constant = build_pca_informed_prior(pd.DataFrame({"t1": [2.0] * 4, "t2": [1.0] * 4}), {"t1": "A", "t2": "A"})
    np.testing.assert_allclose(constant.to_numpy(), 0.25)


# region subsetting


def _gdata_with_genes():
    var = pd.DataFrame({"chrom": ["1"] * 4, "pos": [10, 20, 30, 40]}, index=["v0", "v1", "v2", "v3"])
    gdata = ad.AnnData(X=np.zeros((2, 4)), var=var)
    gdata.uns["variant_annotation_vep"] = pd.DataFrame(
        {"gene_id": ["g1", "g1", "g2", "g3"]}, index=pd.Index(["v0", "v1", "v2", "v3"], name="snp_id")
    )
    return gdata


@pytest.mark.parametrize(
    "genes", [["g1", "g2"], ("g1", "g2"), {"g1", "g2"}, np.array(["g1", "g2"]), pd.Index(["g1", "g2"])]
)
def test_subset_gene_accepts_list_likes(genes):
    assert subset_gene(_gdata_with_genes(), genes).n_vars == 3


def test_subset_gene_single_string():
    assert list(subset_gene(_gdata_with_genes(), "g1").var_names) == ["v0", "v1"]


def test_subset_genomic_region_without_variants_returns_empty_copy():
    var = pd.DataFrame({"chrom": pd.Series([], dtype=object), "pos": pd.Series([], dtype=int)})
    out = subset_genomic_region(ad.AnnData(X=np.zeros((3, 0)), var=var), "1", 0, 100)
    assert out.n_vars == 0
    assert not out.is_view
