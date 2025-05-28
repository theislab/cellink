from __future__ import annotations

import logging

import anndata
import numpy as np
import pandas as pd
import scipy as sp
from chiscore._davies import _pvalue_lambda
from numpy import asarray, atleast_1d

from cellink._core import DonorData

__all__ = [
    "ArrayLike",
    "DotPath",
    "DataContainer",
    "xgower_factor_",
    "davies_pvalue",
    "ensure_float64_array",
    "compute_eigenvals",
    "generate_phenotype",
]

logger = logging.getLogger(__name__)

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


def ensure_float64_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure that the input array is of type float64.

    Parameters
    ----------
    arr : ArrayLike
        Input array-like object (e.g., list, numpy array, pandas Series, DataFrame).

    Returns
    -------
    np.ndarray
        A numpy array of type float64.
    """
    return np.asarray(arr, dtype=np.float64)


# util function to compute eigenvalues using numpy
def compute_eigenvals(
    Lambda: np.ndarray,
) -> np.ndarray:
    """Computes eigenvalues of a matrix.

    Parameters
    ----------
    Lambda : np.array
        Matrix to compute eigenvalues for.

    Returns
    -------
    np.array
        Eigenvalues of the matrix.
    """
    Lambdat = Lambda.astype(np.float32)
    try:
        lambdas = np.linalg.eigvalsh(Lambdat)
    except np.linalg.LinAlgError:
        # If the matrix is not symmetric, use scipy's function to compute eigenvalues
        logging.warning("Switching to scipy for eigenvalue computation due to LinAlgError.")
        lambdas = sp.linalg.eigvalsh(Lambdat)
    return lambdas.astype(np.float64)


def generate_phenotype(
    X: np.array,
    vg: float,
    number_causal_variants: int = 10,
    E: np.ndarray = None,
    number_causal_interactions: int = 0,
    percentage_variance_explained_by_interaction: float = 0,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Simulate a quantitative phenotype with additive genetic effects, environmental effects, and optionally gene-environment interactions.

    Parameters
    ----------
        X: np.ndarray
            Genotype matrix (n_samples x n_variants).
        vg: float
            Proportion of phenotypic variance explained by genetic factors (heritability).
        number_causal_variants: int
            Number of variants with non-zero effect.
        E: np.ndarray, optional
            Environmental matrix (n_samples x n_env_vars).
        number_causal_interactions: int
            Number of environmental variables interacting with genetic effects.

    Returns
    -------
        Y: np.ndarray
            Simulated phenotype (n_samples x 1).
        betas: np.ndarray
            Coefficient vector for genetic effects.
        Yg: np.ndarray
            Genetic contribution to phenotype.
        Yn: np.ndarray
            Noise contribution to phenotype.
        Ye: np.ndarray, optional
            Interaction term (only if E is provided).
        gammas: np.ndarray, optional
            Environmental interaction weights (only if E is provided).
    """
    # sanity checks
    assert (
        number_causal_variants <= X.shape[1]
    ), "Number of causal variants should be smaller than the number of variants in the genotype"
    if E is not None:
        assert (
            number_causal_interactions <= E.shape[1]
        ), "Number of causal interactions should be smaller than the number of environmental factors"
        assert percentage_variance_explained_by_interaction < 1.0, "Percentage can't be larger than 1"
        assert percentage_variance_explained_by_interaction > 0.0, "Percentage can't be less than 0"
    else:
        if percentage_variance_explained_by_interaction != 0:
            logger.warning("With no Environmental contribution this parameter must be 0.")
            percentage_variance_explained_by_interaction = 0
    # Generate random coefficients
    betas = np.zeros((X.shape[1], 1))

    idx = np.random.choice(X.shape[1], number_causal_variants, replace=False)

    betas[idx] = np.random.normal(size=(number_causal_variants, 1))

    # Generate phenotype

    Yn = np.sqrt(1 - vg) * np.random.normal(size=(X.shape[0], 1))

    Yg = X @ betas
    Yg = Yg - np.mean(Yg, axis=0)
    Yg = Yg / xgower_factor_(Yg)
    Yg = np.sqrt(vg * (1 - percentage_variance_explained_by_interaction)) * Yg
    Y = Yg + Yn
    if E is not None:
        # Generate random coefficients
        gammas = np.zeros((E.shape[1], 1))

        idx = np.random.choice(E.shape[1], number_causal_interactions, replace=False)

        gammas[idx] = np.random.normal(size=(number_causal_interactions, 1))
        Ye = (X @ betas) * (E @ gammas)
        Ye = Ye - np.mean(Ye, axis=0)
        Ye = Ye / xgower_factor_(Ye)
        Ye = np.sqrt(vg * percentage_variance_explained_by_interaction) * (Ye)
        Y = Y + Ye

    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    if E is not None:
        return Y, betas, Yg, Yn, Ye, gammas
    return Y, betas, Yg, Yn
