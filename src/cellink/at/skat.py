import logging

import anndata
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy
import scipy.stats as st
#from limix_core.covar import FreeFormCov #TODO ARNOLDT
#from limix_core.gp import GP2KronSumLR

from cellink._core import DonorData
from cellink.at.utils import (
    ArrayLike,
    DataContainer,
    DotPath,
    davies_pvalue,
    ensure_float64_array,
    xgower_factor_,
)


def custom_getattr(name):
    if name in scipy.__dict__:
        return getattr(scipy, name)
    return getattr(np, name)


scipy.__getattr__ = custom_getattr


__all__ = ["Skat"]

logger = logging.getLogger(__name__)


def _skat_test(
    y: np.ndarray,
    X: np.ndarray,
    F: np.ndarray | None = None,
    return_info: bool = False,
) -> tuple[float, dict] | float:
    """
    Performs SKAT test for association between y and X.

    Parameters
    ----------
    y : np.array
        Phenotype data
    X : np.array
        Genotype data
    F : np.array, optional
        Covariates data. If not specified, an intercept is assumed.

    Returns
    -------
    float
        p-value of the SKAT test
    dict
        additional information about the fit of the null model

    Notes
    -----
    This method uses a Gaussian process to fit the null model and compute the skat test statistic.
    """
    # fit exact null model
    E = np.ones([X.shape[0], 1])
    if F is None:
        F = np.ones([X.shape[0], 1])

    # type casting
    y = ensure_float64_array(y)
    X = ensure_float64_array(X)
    F = ensure_float64_array(F)

    gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=np.ones((1, 1)))
    gp.covar.Cr.setCovariance(1e-9 * np.ones((1, 1)))
    gp.covar.Cn.setCovariance(np.ones((1, 1)))
    info_opt = gp.optimize(verbose=False)

    def _P(
        X: np.ndarray,
        gp: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the projection matrix P.

        For mathematical details, see the supplementary method of the paper (Lippert et al. 2014):
        https://doi.org/10.1093/bioinformatics/btu504
        """
        KiX = gp.covar.solve(X)
        FtKiX = gp.mean.W.T.dot(KiX)
        Areml_inv = la.inv(gp.mean.W.T.dot(gp.covar.solve(gp.mean.W)))
        KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(Areml_inv.dot(FtKiX)))
        out = KiX - KiFAiFtKiX
        return out

    # make interaction test
    PY = _P(y, gp)
    # score statistics
    XPY = X.T.dot(PY)
    Q = (XPY**2).sum()

    # eigenvalues
    PX = _P(X, gp)
    Lambda = X.T.dot(PX)
    lambdas = la.eigvalsh(Lambda)
    pv = davies_pvalue(Q, lambdas)
    if return_info:
        return np.array([[pv]]), info_opt
    return np.array([[pv]])


class Skat:
    """SKAT test for association between Y and X."""

    def __init__(self, a: int = 1, b: int = 25, min_threshold: int = 10, max_threshold: int = 5000) -> None:
        """
        SKAT test for association between Y and X.

        Parameters
        ----------
        a : float (default=1)
            Parameter alpha of Beta distribution
        b : float  (default=25)
            Parameter beta of Beta distribution
        min_threshold : int (default=10)
            Minimum number of variants to perform the test
        max_threshold : int (default=5000)
            Maximum number of variants to perform the test
        Values for a, b are chosen accordingly to the SKAT original paper (https://doi.org/10.1016/j.ajhg.2011.05.029)
        While values for min_threshold and max_threshold follow Clarke,Holthamp et al. (https://doi.org/10.1038/s41588-024-01919-z)
        """
        assert a > 0, "Parameter alpha of Beta distribution must be > 0"
        assert b > 0, "Parameter beta of Beta distribution must be > 0"
        self.a = a
        self.b = b
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def run_test(
        self,
        data: DataContainer = None,
        Y: ArrayLike | DotPath | None = None,
        X: ArrayLike | DotPath | None = None,
    ) -> float:
        """Run SKAT test for association between Y and X."""
        # when data is None, Y and X must be provided as numpy arrays
        if data is None:
            assert isinstance(Y, np.ndarray), "If data is None, Y must be provided and be a numpy array"
            assert isinstance(X, np.ndarray), "If data is None, X must be provided and be a numpy array"
            return self._run_test(Y=Y, X=X)
        # when data is provided, Y and X must be provided as strings or lists of strings
        # and must be columns in the data
        else:
            assert isinstance(
                data, pd.DataFrame | anndata.AnnData | DonorData
            ), "data must be a pandas DataFrame, anndata.AnnData or DonorData"
            assert isinstance(Y, str), "Y must be a string or a list of strings"
            assert isinstance(X, str | list[str]), "X must be a string or a list of strings"
            if isinstance(X, str):
                X = [X]
            if isinstance(data, pd.DataFrame):
                assert Y in data.columns, "Y must be a column in the DataFrame."
                assert X in data.columns, "X must be a column in the DataFrame."
                Y = data[Y].values
                X = data[X].values
            elif isinstance(data, anndata.AnnData):
                assert Y in data.obs.columns, "Y must be a column in AnnData obs DataFrame."
                assert X in data.obs.columns, "X must be a column in AnnData obs DataFrame."
                Y = data.obs.loc[:, Y].values
                X = data.obs.loc[:, X].values
            elif isinstance(data, DonorData):
                assert Y in data.donor_data.obs.columns, "Y must be a column in the DonorData object."
                assert X in data.donor_data.obs.columns, "X must be a column in the DonorData object."
                Y = data.donor_data[Y].values
                X = data.donor_data[X].values
            return self._run_test(Y, X)

    def _run_test(
        self,
        Y: np.ndarray,
        X: np.ndarray,
    ) -> float:
        """
        Method to perform SKAT test.

        Variants with a Minor Allele Count (MAC) < 10 are collapsed together.
        If the number of variants is < 10, it returns NaN.
        It also returns NaN if the number of variants is > 5000.
        Same approach as Clarke,Holtkamp, et al. (https://doi.org/10.1038/s41588-024-01919-z)

        Parameters
        ----------
        Y : np.array
            Phenotype data
        X : np.array
            Genotype data

        Returns
        -------
        float
            p-value of the SKAT test
        """
        Ilow = X.sum(0) < self.min_threshold

        if (~Ilow).sum() == 0:
            logger.warning(f"No variants with MAC > {self.min_threshold}. Returning NaN.")
            return np.nan
        else:
            _xlow = X[:, Ilow].sum(1)[:, None]

            if np.all(_xlow == 0):
                logger.warning(f"There are no variants with a MAC < {self.min_threshold}.")
                _Xskat = X[:, ~Ilow]
                maf = 0.5 * _Xskat.mean(0)
            else:
                _Xskat = np.concatenate([X[:, ~Ilow], _xlow], axis=1)
                maf = 0.5 * _Xskat.mean(0)
                maf[-1] = 0.5 * X[:, Ilow].mean()

            if _Xskat.shape[1] > self.max_threshold:
                logger.warning(f"Number of variants > {self.max_threshold}. Returning NaN.")
                return np.nan
            else:
                _weights = st.beta.pdf(maf, self.a, self.b)
                _Xskat = (_Xskat - _Xskat.mean(0)) * np.sqrt(_weights)
                _Xskat = _Xskat / xgower_factor_(_Xskat)
        return _skat_test(Y, _Xskat)
