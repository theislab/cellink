import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform as sk_quantile_transform
from statsmodels.stats.multitest import fdrcorrection

__all__ = ["quantile_transform", "bonferroni_adjustment", "q_value"]

def quantile_transform(x, seed=1):
    """
    Gaussian quantile transform for values in a pandas Series.    :param x: Input pandas Series.
    :type x: pd.Series
    :param seed: Random seed.
    :type seed: int
    :return: Transformed Series.
    :rtype: pd.Series    .. note::
        “nan” values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()
    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)
    x_transform[~is_nan] = sk_quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]
    return x_transform

def bonferroni_adjustment(pv, no_tested_variants):
    return np.clip(pv * no_tested_variants, 0, 1)

def q_value(pv, no_tested_variants):
    bf_pv = bonferroni_adjustment(pv, no_tested_variants)
    return fdrcorrection(bf_pv)[1]
