from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["coloc_abf", "coloc_susie", "DEFAULT_PRIOR_VAR", "DEFAULT_PRIOR_VAR_CC"]


DEFAULT_P1 = 1e-4
DEFAULT_P2 = 1e-4
DEFAULT_P12 = 1e-5
DEFAULT_PRIOR_VAR = 0.15**2
DEFAULT_PRIOR_VAR_CC = 0.2**2


def _log_abf(beta: np.ndarray, se: np.ndarray, prior_var: float) -> np.ndarray:
    """Wakefield (2007) approximate log Bayes factor per SNP."""
    z = beta / se
    v = se**2
    r = prior_var / (prior_var + v)
    return 0.5 * (np.log(1 - r) + r * z**2)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _log_h3(l1: np.ndarray, l2: np.ndarray, lsum1: float, lsum2: float) -> float:
    log_all_pairs = lsum1 + lsum2
    log_diag = _logsumexp(l1 + l2)
    diff = np.exp(log_all_pairs - log_diag) - 1
    if diff <= 0:
        return -np.inf
    return log_diag + np.log(diff)


def coloc_abf(
    beta1: np.ndarray | pd.Series,
    se1: np.ndarray | pd.Series,
    beta2: np.ndarray | pd.Series,
    se2: np.ndarray | pd.Series,
    p1: float = DEFAULT_P1,
    p2: float = DEFAULT_P2,
    p12: float = DEFAULT_P12,
    prior_var: float = DEFAULT_PRIOR_VAR,
    prior_var2: float | None = None,
) -> dict[str, float]:
    """Approximate Bayes factor colocalization between two association
    studies sharing the same set of SNPs (Giambartolomei et al. 2014).

    Tests, for one gene/region, whether a cis-eQTL signal and a GWAS
    signal are more consistent with a single shared causal variant (H4)
    than with two distinct causal variants (H3), association with only
    one trait (H1/H2), or no association at all (H0).

    Parameters
    ----------
    beta1
        Effect sizes for study 1 (e.g. cis-eQTL), one per SNP.
    se1
        Standard errors for study 1, same order as ``beta1``.
    beta2
        Effect sizes for study 2 (e.g. GWAS), same SNPs and order as ``beta1``.
    se2
        Standard errors for study 2, same order as ``beta2``.
    p1
        Prior probability a given SNP is causal for study 1 only.
    p2
        Prior probability a given SNP is causal for study 2 only.
    p12
        Prior probability a given SNP is causal for both studies jointly
        (i.e. the shared-causal-variant hypothesis, H4).
    prior_var
        Prior variance of study 1's effect size, on the scale ``beta1``/``se1``
        are given in (the default assumes a standardized, sdY=1 quantitative
        trait; rescale by ``sdY**2`` otherwise, matching R coloc's own
        ``sdY`` convention).
    prior_var2
        Prior variance of study 2's effect size. Defaults to ``prior_var``
        (both studies quantitative, sharing one prior, the common case for
        two eQTL-like studies). Pass ``0.2**2`` when study 2 is a
        case-control GWAS, matching R coloc's own default for ``type="cc"``.

    Returns
    -------
    dict with keys ``PP0``-``PP4``, the posterior probability of each
    hypothesis (summing to 1):

    - PP0: no association with either trait
    - PP1: association with study 1 only
    - PP2: association with study 2 only
    - PP3: both traits associated, but with distinct causal variants
    - PP4: both traits associated, with the same causal variant

    Examples
    --------
    >>> import numpy as np
    >>> from cellink.tl import coloc_abf
    >>> rng = np.random.default_rng(0)
    >>> n = 50
    >>> beta1 = rng.normal(0, 0.02, n)
    >>> se1 = np.full(n, 0.05)
    >>> beta2 = rng.normal(0, 0.02, n)
    >>> se2 = np.full(n, 0.05)
    >>> beta1[0], se1[0] = 0.5, 0.04  # SNP 0: a real shared causal signal
    >>> beta2[0], se2[0] = 0.4, 0.05
    >>> pp = coloc_abf(beta1, se1, beta2, se2)
    >>> bool(pp["PP4"] > 0.99)
    True
    """
    beta1 = np.asarray(beta1, dtype=float)
    se1 = np.asarray(se1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    se2 = np.asarray(se2, dtype=float)
    if not (len(beta1) == len(se1) == len(beta2) == len(se2)):
        raise ValueError("beta1, se1, beta2, se2 must all have the same length (one shared set of SNPs).")

    l1 = _log_abf(beta1, se1, prior_var)
    l2 = _log_abf(beta2, se2, prior_var2 if prior_var2 is not None else prior_var)
    return _combine_log_abf(l1, l2, p1, p2, p12)


def _combine_log_abf(l1: np.ndarray, l2: np.ndarray, p1: float, p2: float, p12: float) -> dict[str, float]:
    """Standard coloc H0-H4 combination (Giambartolomei et al. 2014, matching
    R coloc's own ``combine.abf``) given two arrays of per-SNP log Bayes
    factors already on a shared set of SNPs, the common core of both
    ``coloc_abf`` (l1/l2 from Wakefield ABF on beta/se) and ``coloc_susie``
    (l1/l2 taken directly from SuSiE's own per-effect ``lbf_variable`` rows)."""
    if len(l1) != len(l2):
        raise ValueError("l1 and l2 must have the same length (one shared set of SNPs).")
    lsum1 = _logsumexp(l1)
    lsum2 = _logsumexp(l2)
    lsum12 = _logsumexp(l1 + l2)

    log_h = {
        "PP0": 0.0,
        "PP1": np.log(p1) + lsum1,
        "PP2": np.log(p2) + lsum2,
        "PP3": np.log(p1) + np.log(p2) + _log_h3(l1, l2, lsum1, lsum2),
        "PP4": np.log(p12) + lsum12,
    }
    logs = np.array(list(log_h.values()))
    m = np.max(logs)
    post = np.exp(logs - m)
    post = post / post.sum()
    return dict(zip(log_h.keys(), post))


def coloc_susie(
    lbf1: np.ndarray | pd.DataFrame,
    lbf2: np.ndarray | pd.DataFrame,
    cs1_index: list[int] | np.ndarray,
    cs2_index: list[int] | np.ndarray,
    p1: float = DEFAULT_P1,
    p2: float = DEFAULT_P2,
    p12: float = DEFAULT_P12,
    snp_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Multiple-causal-variant colocalization against SuSiE credible sets
    from both studies (Wallace 2021; R coloc's ``coloc.susie()`` /
    ``coloc.bf_bf()``), rather than ``coloc_abf``'s single-causal-variant-
    per-trait assumption.

    Does not run SuSiE itself, takes each study's own already-fitted
    SuSiE per-single-effect log Bayes factor matrix (``lbf_variable``, a
    real, standard output field of ``susieR::susie()``/``susie_rss()``: one
    row per single effect ``l``, one column per SNP) and, for every pair of
    real credible sets (one from each study), runs the exact same H0-H4
    combination ``coloc_abf`` uses, just on that effect's own per-SNP
    log-BF row directly, rather than a Wakefield ABF recomputed from
    beta/se. This is the identical algorithm R's ``coloc.bf_bf()``
    implements, verified numerically against it directly.

    Parameters
    ----------
    lbf1
        Study 1's SuSiE fit's ``lbf_variable`` matrix, shape
        ``(n_effects1, n_snps)``, same SNP order/columns as ``lbf2``.
    lbf2
        Study 2's SuSiE fit's ``lbf_variable`` matrix, shape
        ``(n_effects2, n_snps)``.
    cs1_index
        Row indices into ``lbf1`` corresponding to study 1's real credible
        sets (SuSiE's own ``fit$sets$cs_index``, i.e. the single effects
        that survived SuSiE's own purity/coverage filtering, not every
        row of ``lbf_variable``, which includes effects SuSiE fit but did
        not resolve into a credible set).
    cs2_index
        Row indices into ``lbf2`` for study 2's real credible sets.
    p1, p2, p12
        Same priors as ``coloc_abf``.
    snp_ids
        Optional SNP labels (length ``n_snps``) for the ``hit1``/``hit2``
        columns (each pair's most likely causal SNP per study). Defaults to
        integer positions if not given.

    Returns
    -------
    pd.DataFrame, one row per (study-1 signal, study-2 signal) pair, with
    columns ``idx1``, ``idx2`` (the row indices from ``cs1_index``/
    ``cs2_index``), ``nsnps``, ``hit1``, ``hit2``, and ``PP0``-``PP4``.

    Examples
    --------
    >>> import numpy as np
    >>> from cellink.tl import coloc_susie
    >>> n = 30
    >>> lbf1 = np.zeros((1, n)); lbf1[0, 5] = 20.0  # effect 0 points at SNP 5
    >>> lbf2 = np.zeros((1, n)); lbf2[0, 5] = 15.0  # effect 0 also points at SNP 5
    >>> res = coloc_susie(lbf1, lbf2, cs1_index=[0], cs2_index=[0])
    >>> bool(res.loc[0, "PP4"] > 0.9)
    True
    """
    lbf1 = np.asarray(lbf1, dtype=float)
    lbf2 = np.asarray(lbf2, dtype=float)
    cs1_index = np.asarray(cs1_index, dtype=int)
    cs2_index = np.asarray(cs2_index, dtype=int)
    if lbf1.shape[1] != lbf2.shape[1]:
        raise ValueError("lbf1 and lbf2 must have the same number of columns (one shared set of SNPs).")
    n_snps = lbf1.shape[1]
    labels = list(snp_ids) if snp_ids is not None else list(range(n_snps))

    rows = []
    for idx1 in cs1_index:
        l1 = lbf1[idx1]
        for idx2 in cs2_index:
            l2 = lbf2[idx2]
            pp = _combine_log_abf(l1, l2, p1, p2, p12)
            rows.append(
                {
                    "idx1": int(idx1),
                    "idx2": int(idx2),
                    "nsnps": n_snps,
                    "hit1": labels[int(np.argmax(l1))],
                    "hit2": labels[int(np.argmax(l2))],
                    **pp,
                }
            )
    return pd.DataFrame(rows)
