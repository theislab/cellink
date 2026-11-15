from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["build_pca_informed_prior"]


def build_pca_informed_prior(
    scores: pd.DataFrame,
    track_categories: pd.Series | dict[str, str],
    variance_threshold: float = 0.95,
    max_pcs_per_category: int = 20,
    random_state: int | None = 0,
) -> pd.Series:
    """Build a pooled, per-category-PCA fine-mapping prior from a wide,
    per-variant/per-track functional-genomics score matrix.

    Following the category-split + per-category-PCA structure of
    Srivastava et al. 2025 ("Borzoi-informed fine mapping improves causal
    variant prioritization in complex trait GWAS", bioRxiv
    2025.07.09.663936): within each category, every track is standardized
    (zero mean, unit variance across variants) before PCA. PCA is then
    run per category and components are retained up to ``variance_threshold``
    cumulative explained variance, capped at ``max_pcs_per_category``.
    
    Parameters
    ----------
    scores : pandas.DataFrame
        Wide score matrix, one row per variant (any index, typically
        ``variant_id``) and one column per track. May contain columns not
        present in ``track_categories`` (e.g. metadata columns) -- only
        columns with a known category are used. NaNs within used columns
        are treated as 0 (no evidence) after standardization.
    track_categories : pandas.Series or dict
        Maps each track's column name (as it appears in ``scores.columns``)
        to a category label (e.g. ``"ATAC"``, ``"RNA_GTEx"``). Only the
        intersection with ``scores.columns`` is used; entries for columns
        absent from ``scores`` are ignored.
    variance_threshold : float, default=0.95
        Cumulative explained-variance fraction at which to stop retaining
        components within a category.
    max_pcs_per_category : int, default=20
        Upper bound on the number of principal components retained per
        category, regardless of ``variance_threshold``.
    random_state : int or None, default=0
        Passed to ``sklearn.decomposition.PCA`` for reproducibility.

    Returns
    -------
    pandas.Series
        One combined prior weight per row of ``scores`` (same index),
        non-negative, summing to 1 across all variants.

    Raises
    ------
    ValueError
        If no column of ``scores`` has a matching entry in
        ``track_categories``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cellink.tl import build_pca_informed_prior
    >>> rng = np.random.default_rng(0)
    >>> scores = pd.DataFrame(
    ...     {"atac_1": rng.normal(size=6), "rna_1": rng.normal(size=6)},
    ...     index=[f"1:{100 + i}:A:G" for i in range(6)],
    ... )
    >>> track_categories = {"atac_1": "ATAC", "rna_1": "RNA"}
    >>> prior = build_pca_informed_prior(scores, track_categories)
    >>> len(prior) == len(scores)
    True
    >>> round(float(prior.sum()), 8)
    1.0
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if not isinstance(track_categories, pd.Series):
        track_categories = pd.Series(track_categories)

    used_cols = [c for c in scores.columns if c in track_categories.index]
    if not used_cols:
        raise ValueError(
            "None of `scores.columns` has a matching entry in `track_categories` -- "
            "nothing to build a prior from."
        )

    n_variants = len(scores)
    pooled_pcs: list[np.ndarray] = []

    for category in sorted(track_categories.loc[used_cols].unique()):
        cat_cols = [c for c in used_cols if track_categories[c] == category]
        X = scores[cat_cols].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        n_features = X.shape[1]
        max_components = min(n_variants, n_features)

        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=max_components, random_state=random_state)
        pc_scores = pca.fit_transform(Xs)
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        n_at_threshold = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_at_threshold = min(n_at_threshold, max_components)
        n_retained = min(n_at_threshold, max_pcs_per_category)

        logger.info(
            f"build_pca_informed_prior: category={category!r}, {n_features} tracks, "
            f"{max_components} PCs available, {n_at_threshold} needed for "
            f"{variance_threshold:.0%} variance, {n_retained} retained "
            f"(cap={max_pcs_per_category})"
        )

        retained = pc_scores[:, :n_retained]
        retained_std = StandardScaler().fit_transform(retained)
        for j in range(n_retained):
            pooled_pcs.append(retained_std[:, j])

    pooled = np.stack(pooled_pcs, axis=1)
    combined_l2 = np.linalg.norm(pooled, axis=1)
    combined_prior = combined_l2 / combined_l2.sum()

    return pd.Series(combined_prior, index=scores.index, name="pca_informed_prior")
