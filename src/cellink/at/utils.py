
from typing import Union
import numpy as np
from numpy import asarray, atleast_1d
from chiscore._davies import _pvalue_lambda
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR
import numpy.linalg as la

def xgower_factor_(X):
    a = np.power(X, 2).sum()
    b = X.dot(X.sum(0)).sum()
    return np.sqrt((a - b / X.shape[0]) / (X.shape[0] - 1))

def davies_pvalue(tstats:np.array, weights:np.array, return_info=False) -> Union[tuple[np.ndarray, dict], np.ndarray]:
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

def skat_test(y, X, F=None,return_info=False)-> Union[tuple[float, dict], float]:
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
        """
        # fit exact null model
        E = np.ones([X.shape[0], 1])
        if F is None:
            F = np.ones([X.shape[0], 1])
        gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(1e-9 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(np.ones((1, 1)))
        info_opt = gp.optimize(verbose=False)

        def _P(X, gp):
            """
            Computes the projection matrix P = I - F (F'F)^-1 F'
            """
            # P = I - F (F'F)^-1 F'
            KiX = gp.covar.solve(X)
            FtKiX = gp.mean.W.T.dot(KiX)
            Areml_inv = la.inv(gp.mean.W.T.dot(gp.covar.solve(gp.mean.W)))
            KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(Areml_inv.dot(FtKiX)))
            out = KiX - KiFAiFtKiX
            return out

        # make interaction test
        PY = _P(y,gp)
        # score statistics
        XPY = X.T.dot(PY)
        Q = (XPY**2).sum()

        # eigenvalues
        PX = _P(X,gp)
        Lambda = X.T.dot(PX)
        lambdas = la.eigvalsh(Lambda)
        if return_info:
            return davies_pvalue(Q, lambdas), info_opt
        return davies_pvalue(Q, lambdas)
        