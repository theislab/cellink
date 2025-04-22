import numpy as np
import scipy.stats as st


def acat_test(pvalues: np.ndarray, tolerance: float = 1e-16, weights: np.ndarray = None) -> float:
    """
    Perform the Aggregated Cauchy Association Test (ACAT) to combine p-values.

    ACAT is a fast and robust method for combining possibly dependent p-values,
    using properties of the Cauchy distribution. It is especially effective when
    only a subset of the input p-values are small.
    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R

    Parameters
    ----------
    pvalues : np.ndarray or list of float
        The p-values to combine. Must be in the range [0, 1]. NaN values are ignored.

    tolerance : float, optional
        A lower bound for p-values to ensure numerical stability. Values below this threshold
        are clipped to `tolerance`. Default is 1e-16.

    weights : np.ndarray or list of float, optional
        Non-negative weights for each p-value. If None, equal weights are used.

    Returns
    -------
    float
        The ACAT-combined p-value.

    """
    if weights is None:
        weights = [1 / len(pvalues) for i in pvalues]

    assert len(weights) == len(pvalues), "Length of weights and p-values differs."
    assert weights.all() > 0, "All weights must be positive."

    if not any(pvalues < tolerance):
        cct_stat = sum(weights * np.tan((0.5 - pvalues) * np.pi))
    else:
        is_small = [i < (tolerance) for i in pvalues]
        is_large = [i >= (tolerance) for i in pvalues]
        cct_stat = sum((weights[is_small] / pvalues[is_small]) / np.pi)
        cct_stat += sum(weights[is_large] * np.tan((0.5 - pvalues[is_large]) * np.pi))
    if cct_stat > 1e15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - st.cauchy.cdf(cct_stat)
    return pval


def compute_acat(pvs: np.ndarray, tolerance: float = 1e-16, weights: np.ndarray = None) -> np.ndarray:
    """
    Aggregate p-values using the Cauchy combination method (ACAT).

    This function performs p-value aggregation using the ACAT (Aggregated Cauchy Association Test) method,
    which is particularly powerful and robust for combining dependent and sparse p-values. It supports
    missing values (NaNs) and optional weighting.

    Parameters
    ----------
    pvs : np.ndarray
        A 2D array of shape (n_different_tests, n_tests_to_integrate), where each row contains
        the p-values to be aggregated into a single meta p-value. Values must be in the range [0, 1].
        NaNs are allowed and will be ignored during aggregation.

    tolerance : float, optional
        A small threshold to avoid issues with extremely small p-values. Values below `tolerance`
        are capped at `tolerance` to maintain numerical stability. Default is 1e-16.

    weights : np.ndarray, optional
        A 1D array of non-negative weights with shape (n_tests_to_integrate,). If provided,
        it should sum to 1 or will be internally normalized. If None, equal weights are used.

    Returns
    -------
    np.ndarray
        A 1D array of aggregated p-values, one for each row of the input `pvs`.

    Notes
    -----
    The ACAT method transforms each p-value `p` using the tangent function:
        T = sum_i w_i * tan[(0.5 - p_i) * pi]
    Then the aggregated p-value is computed using the Cauchy distribution:
        p_combined = 0.5 - arctan(T) / pi

    References
    ----------
    Liu, et al (2019).
    ACAT: A Fast and Powerful p Value Combination Method for Rare-Variant Analysis in Sequencing Studies
    The American Journal of Human Genetics, 104(3), 410–421.
    https://doi.org/10.1016/j.ajhg.2019.01.002
    """
    RV = np.ones(pvs.shape[0])
    for i in range(pvs.shape[0]):
        _pvs = pvs[i]
        is_nan = np.isnan(_pvs)
        if is_nan.all():
            RV[i] = np.nan
        else:
            _pvs = _pvs[~is_nan]
            _pvs[_pvs == 0] = tolerance
            _pvs[_pvs == 1] = 1 - tolerance
            RV[i] = acat_test(_pvs, tolerance=tolerance, weights=weights)
    return RV
