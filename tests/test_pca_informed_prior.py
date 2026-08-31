from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellink.tl import build_pca_informed_prior

VARIANT_IDS = [f"1:{100 + i}:A:G" for i in range(6)]


def _scores_two_single_track_categories(rng):
    """Each category has exactly one track."""
    trackA = rng.normal(loc=5.0, scale=2.0, size=len(VARIANT_IDS))
    trackB = rng.normal(loc=-3.0, scale=0.5, size=len(VARIANT_IDS))
    scores = pd.DataFrame({"trackA": trackA, "trackB": trackB}, index=VARIANT_IDS)
    track_categories = {"trackA": "A", "trackB": "B"}
    return scores, track_categories, trackA, trackB


def test_build_pca_informed_prior_matches_independently_computed_zscore_l2norm():
    rng = np.random.default_rng(0)
    scores, track_categories, trackA, trackB = _scores_two_single_track_categories(rng)

    prior = build_pca_informed_prior(scores, track_categories)

    zA = (trackA - trackA.mean()) / trackA.std(ddof=0)
    zB = (trackB - trackB.mean()) / trackB.std(ddof=0)
    l2 = np.sqrt(zA**2 + zB**2)
    expected = l2 / l2.sum()

    assert list(prior.index) == VARIANT_IDS
    np.testing.assert_allclose(prior.to_numpy(), expected, rtol=1e-6, atol=1e-8)


def test_build_pca_informed_prior_sums_to_one_and_is_nonnegative():
    rng = np.random.default_rng(1)
    scores, track_categories, _, _ = _scores_two_single_track_categories(rng)

    prior = build_pca_informed_prior(scores, track_categories)

    assert prior.sum() == pytest.approx(1.0)
    assert (prior >= 0).all()


def test_build_pca_informed_prior_ignores_columns_without_a_category():
    rng = np.random.default_rng(2)
    scores, track_categories, trackA, trackB = _scores_two_single_track_categories(rng)
    scores["metadata_col"] = ["not_a_track"] * len(VARIANT_IDS)

    prior = build_pca_informed_prior(scores, track_categories)

    zA = (trackA - trackA.mean()) / trackA.std(ddof=0)
    zB = (trackB - trackB.mean()) / trackB.std(ddof=0)
    l2 = np.sqrt(zA**2 + zB**2)
    expected = l2 / l2.sum()
    np.testing.assert_allclose(prior.to_numpy(), expected, rtol=1e-6, atol=1e-8)


def test_build_pca_informed_prior_accepts_series_or_dict_categories():
    rng = np.random.default_rng(3)
    scores, track_categories, _, _ = _scores_two_single_track_categories(rng)

    prior_from_dict = build_pca_informed_prior(scores, track_categories)
    prior_from_series = build_pca_informed_prior(scores, pd.Series(track_categories))

    np.testing.assert_allclose(prior_from_dict.to_numpy(), prior_from_series.to_numpy())


def test_build_pca_informed_prior_raises_when_no_columns_match():
    rng = np.random.default_rng(4)
    scores, _, _, _ = _scores_two_single_track_categories(rng)

    with pytest.raises(ValueError, match="matching entry"):
        build_pca_informed_prior(scores, {"unrelated_col": "X"})


def test_build_pca_informed_prior_respects_max_pcs_per_category_cap():
    rng = np.random.default_rng(5)
    n = 20
    # One category with 5 correlated-but-not-identical tracks; capping at 1 PC
    # must not raise and must still return a valid, normalized prior.
    base = rng.normal(size=n)
    cols = {f"trackC{i}": base + rng.normal(scale=0.01, size=n) for i in range(5)}
    scores = pd.DataFrame(cols, index=[f"1:{200 + i}:A:G" for i in range(n)])
    track_categories = {c: "C" for c in cols}

    prior = build_pca_informed_prior(scores, track_categories, max_pcs_per_category=1)

    assert len(prior) == n
    assert prior.sum() == pytest.approx(1.0)
