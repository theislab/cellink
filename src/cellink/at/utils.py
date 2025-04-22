from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
from chiscore._davies import _pvalue_lambda
from numpy import asarray, atleast_1d

from cellink._core import DonorData

# To handle multiple data types
ArrayLike = np.ndarray | pd.Series | pd.DataFrame | list[float] | list[int]
DotPath = str | list[str]
# If passing Y and X directly | If Y and X are in a DataFrame |   If Y and X are in AnnData | If Y and X are in DonorData
DataContainer = None | pd.DataFrame | anndata.AnnData | DonorData


def xgower_factor_(X):
    """
    Computes a scaling factor based on the extended Gower's centered similarity matrix.

    This function calculates a scalar that is useful in standardizing kernel-based similarity
    or distance matrices derived from the input matrix `X`. It is often used to scale genetic
    similarity matrices or relatedness matrices in genome-wide association studies (GWAS) or
    kernel machine methods.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features), where each row is a sample and
        each column is a feature (e.g., SNP dosage or genotype encoding).

    Returns
    -------
    float
        The scaling factor derived from the variance structure of `X`, computed as:

        sqrt( (sum(X^2) - sum(X)^T * sum(X) / n) / (n - 1) )

    Notes
    -----
    This is related to the trace of the Gower-centered Gram matrix and can be used to
    normalize kernels or similarity matrices when comparing across datasets or models.
    """
    a = np.power(X, 2).sum()
    b = X.dot(X.sum(0)).sum()
    return np.sqrt((a - b / X.shape[0]) / (X.shape[0] - 1))


def davies_pvalue(tstats: np.array, weights: np.array, return_info=False) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    Joint significance of statistics derived from chi2-squared distributions.

    Parameters
    ----------
    tstats : float
        Test statistics.
    weights : array_like
        Weights of the linear combination.

    Returns
    -------
    float, dict
        Estimated p-value.
        If return_info is True, also returns a dictionary with additional information.
    """
    tstats = asarray(atleast_1d(tstats), float)
    weights = asarray(weights, float)
    maxstats = tstats.max()
    if maxstats > 0:
        tstats = tstats / maxstats
        weights = weights / maxstats
    re = _pvalue_lambda(weights, tstats)
    if return_info:
        return re["p_value"][0], re
    return re["p_value"][0]
