import logging

import anndata
import numpy as np
import pandas as pd
import scipy.linalg as la
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR
from limix_lmm import LMM
from tqdm import tqdm

from cellink._core import DonorData
from cellink.at.utils import (
    ArrayLike,
    DataContainer,
    DotPath,
    compute_eigenvals,
    davies_pvalue,
    ensure_float64_array,
)

__all__ = ["StructLMM"]

logger = logging.getLogger(__name__)


class StructLMM:
    """Faster version of StructLMM."""

    def __init__(
        self,
        data: DataContainer = None,
        Y: ArrayLike | DotPath | None = None,
        X: ArrayLike | DotPath | None = None,
        E: ArrayLike | DotPath | None = None,
        F: ArrayLike | DotPath | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the OurStructLMM class.

        Parameters
        ----------
        y : np.ndarray
            Phenotype data.
        E : np.ndarray
            Covariance matrix of the variants.
        F : np.ndarray
            Covariates data. If not specified, an intercept is assumed.
        verbose: bool, optional
        """
        if data is None:
            assert isinstance(Y, np.ndarray), "If data is None, Y must be provided and be a numpy array"
            assert isinstance(X, np.ndarray), "If data is None, X must be provided and be a numpy array"
            assert isinstance(F, np.ndarray), "If data is None, F must be provided and be a numpy array"
            assert isinstance(E, np.ndarray), "If data is None, E must be provided and be a numpy array"

        # when data is provided, Y and X must be provided as strings or lists of strings
        # and must be columns in the data
        else:
            assert isinstance(
                data, pd.DataFrame | anndata.AnnData | DonorData
            ), "data must be a pandas DataFrame, anndata.AnnData or DonorData"
            assert isinstance(Y, str), "Y must be a string or a list of strings"
            assert isinstance(X, str | list[str]), "X must be a string or a list of strings"
            assert isinstance(F, str | list[str]), "F must be a string or a list of strings"
            assert isinstance(E, str | list[str]), "E must be a string or a list of strings"
            if isinstance(X, str):
                X = [X]
            if isinstance(F, str):
                F = [F]
            if isinstance(E, str):
                E = [E]
            if isinstance(data, pd.DataFrame):
                assert Y in data.columns, "Y must be a column in the DataFrame."
                assert X in data.columns, "X must be a column in the DataFrame."
                assert F in data.columns, "F must be a column in the DataFrame."
                assert E in data.columns, "E must be a column in the DataFrame."
                Y = data[Y].values
                X = data[X].values
                F = data[F].values
                E = data[E].values
            elif isinstance(data, anndata.AnnData):
                assert Y in data.obs.columns, "Y must be a column in AnnData obs DataFrame."
                assert X in data.obs.columns, "X must be a column in AnnData obs DataFrame."
                assert F in data.obs.columns, "F must be a column in AnnData obs DataFrame."
                assert E in data.obs.columns, "E must be a column in AnnData obs DataFrame."
                Y = data.obs.loc[:, Y].values
                X = data.obs.loc[:, X].values
                F = data.obs.loc[:, F].values
                E = data.obs.loc[:, E].values
            elif isinstance(data, DonorData):
                assert Y in data.donor_data.obs.columns, "Y must be a column in the DonorData object."
                assert X in data.donor_data.obs.columns, "X must be a column in the DonorData object."
                assert F in data.donor_data.obs.columns, "F must be a column in the DonorData object."
                assert E in data.donor_data.obs.columns, "E must be a column in the DonorData object."
                Y = data.donor_data[Y].values
                X = data.donor_data[X].values
                F = data.donor_data[F].values
                E = data.donor_data[E].values

        # type casting
        Y = ensure_float64_array(Y)  # phenotype
        X = ensure_float64_array(X)  # genetics
        E = ensure_float64_array(E)  # environment
        F = ensure_float64_array(F)  # covariates

        self.Y = Y
        self.X = X
        self.E = E
        self.F = F

        self.verbose = verbose

    def interaction_test(
        self,
        exact: bool = False,
    ) -> np.ndarray:
        """Perform the interaction test for association between y and G.

        Parameters
        ----------
        G : np.ndarray
            Genotype data.
        exact : bool, optional
            If True, perform the exact test. If False, perform the approximate test using GPs.
        """
        # type casting
        # G = ensure_float64_array(G)

        if exact or self.X.shape[1] == 1:
            if not exact:
                logger.info("Exact test not requested, but only one variant provided. Using exact test.")

            iterator = range(self.X.shape[1])
            if self.verbose:
                iterator = tqdm(iterator, desc="Exact test")

            self.pvs = np.array([self.single_interaction_test(self.X[:, [i]]) for i in iterator])
            return self.pvs

        # learn a covariance on the null model (no variant effect; this is a hack, should be changed)
        gp = GP2KronSumLR(Y=self.Y, Cn=FreeFormCov(1), G=self.E, F=self.F, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)

        # fit null
        self.lmm = LMM(self.Y, self.F, gp.covar.solve)
        self.lmm.process(self.X)
        pv = self.lmm.getPv()  # noqa: F841
        beta = self.lmm.getBetaSNP()
        beta_ste = self.lmm.getBetaSNPste()  # noqa: F841
        lrt = self.lmm.getLRT()  # noqa: F841

        # make interaction test
        Yhat = self.F.dot(self.lmm.beta_F) - self.X * beta
        Yr = self.Y - Yhat
        PY = gp.covar.solve(Yr) / self.lmm.s2

        # score statistics
        W = np.einsum("ns,nk->nsk", self.X, self.E)
        WPY = np.einsum("nsk,ns->sk", W, PY)
        Q = np.einsum("sk,sk->s", WPY, WPY)

        # eigenvalues
        PW = np.zeros_like(W)
        for i in range(W.shape[2]):
            PW[:, :, i] = (
                gp.covar.solve(W[:, :, i] - Yhat) / self.lmm.s2
            )  # added the denominator here, which was missing (was a bug. Probably not influential)
        Lambda = np.einsum("nsk,nsl->skl", W, PW)
        lambdas = compute_eigenvals(Lambda)

        self.pvs = np.array([davies_pvalue(Q[i], lambdas[i]) for i in range(self.X.shape[1])])
        return self.pvs

    def _P(
        self,
        X: np.ndarray,
        gp: GP2KronSumLR,
    ) -> np.ndarray:
        """
        Compute the projection of X onto the null space of the mean model.

        Parameters
        ----------
        X : np.ndarray
            Input data to project.
        gp : GP2KronSumLR
            Gaussian process model.
        """
        # type casting
        X = ensure_float64_array(X)

        KiX = gp.covar.solve(X)
        FtKiX = gp.mean.W.T.dot(KiX)
        Areml_inv = la.inv(gp.mean.W.T.dot(gp.covar.solve(gp.mean.W)))
        KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(Areml_inv.dot(FtKiX)))
        #    KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(gp.Areml.solve(FtKiX)))
        out = KiX - KiFAiFtKiX
        return out

    def single_interaction_test(
        self,
        g: np.ndarray,
    ) -> np.ndarray:
        """Single interaction test.

        Parameters
        ----------
        g : np.ndarray
            Genotype data for a single variant.
        """
        # type casting
        g = ensure_float64_array(g)  # single column as genotype

        # fit exact null model
        F1 = np.concatenate([self.F, g], 1)
        gp = GP2KronSumLR(Y=self.Y, Cn=FreeFormCov(1), G=self.E, F=F1, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)

        # make interaction test
        PY = self._P(self.Y, gp)

        # score statistics
        W = g * self.E
        WPY = W.T.dot(PY)
        Q = (WPY**2).sum()

        # eigenvalues
        PW = self._P(W, gp)
        Lambda = W.T.dot(PW)
        lambdas = compute_eigenvals(Lambda)
        return davies_pvalue(Q, lambdas)
