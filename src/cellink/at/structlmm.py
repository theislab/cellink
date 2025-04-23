import logging

import numpy as np
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR
from limix_lmm import LMM
from tqdm import tqdm
import scipy.linalg as la

from cellink.at.utils import davies_pvalue, ensure_float64_array


logger = logging.getLogger(__name__)


# util function to compute eigenvalues using numpy
def compute_eigenvals(
        Lambda: np.ndarray,
    ) -> np.ndarray:
    """Compute eigenvalues of a matrix."""
    Lambdat = Lambda.astype(np.float32)
    lambdas = np.linalg.eigvalsh(Lambdat)
    return lambdas.astype(np.float64)


class OurStructLMM:
    """Faster version of StructLMM."""

    def __init__(
        self,
        y: np.ndarray,
        E: np.ndarray,
        F: np.ndarray,
        verbose: bool = False
    ) -> None:
        """Initialize the OurStructLMM class."""

        # type casting
        y = ensure_float64_array(y)
        E = ensure_float64_array(E)
        F = ensure_float64_array(F)

        self.y = y
        self.E = E
        self.F = F

        self.verbose = verbose

    def interaction_test(
        self,
        G: np.ndarray,
        exact: bool = False,
    ) -> np.ndarray:
        """"""
        # type casting
        G = ensure_float64_array(G)

        if exact or G.shape[1] == 1:
            if not exact:
                logger.info("Exact test not requested, but only one variant provided. Using exact test.")

            iterator = range(G.shape[1])
            if self.verbose:
                iterator = tqdm(iterator, desc="Exact test")

            self.pvs = np.array([self.single_interaction_test(G[:, [i]]) for i in iterator])
            return self.pvs

        # learn a covariance on the null model (no variant effect; this is a hack, should be changed)
        gp = GP2KronSumLR(Y=self.y, Cn=FreeFormCov(1), G=self.E, F=self.F, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)  # noqa: F841

        # fit null
        self.lmm = LMM(self.y, self.F, gp.covar.solve)
        self.lmm.process(G)
        pv = self.lmm.getPv()  # noqa: F841
        beta = self.lmm.getBetaSNP()  # noqa: F841
        beta_ste = self.lmm.getBetaSNPste()  # noqa: F841
        lrt = self.lmm.getLRT()  # noqa: F841

        # make interaction test
        Yhat = self.F.dot(self.lmm.beta_F) - G * beta
        Yr = self.y - Yhat
        PY = gp.covar.solve(Yr) / self.lmm.s2

        # score statistics
        W = np.einsum("ns,nk->nsk", G, self.E)
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

        self.pvs = np.array([davies_pvalue(Q[i], lambdas[i]) for i in range(G.shape[1])])
        self.G = G
        return self.pvs

    
    def _P(
        self,
        X: np.ndarray,
        gp: GP2KronSumLR,
    ) -> np.ndarray:
        """
        Compute the projection of X onto the null space of the mean model.
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
        """Single interaction test."""
        # type casting
        g = ensure_float64_array(g)

        # fit exact null model
        F1 = np.concatenate([self.F, g], 1)
        gp = GP2KronSumLR(Y=self.y, Cn=FreeFormCov(1), G=self.E, F=F1, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)

        # make interaction test
        PY = self._P(self.y,gp)

        # score statistics
        W = g * self.E
        WPY = W.T.dot(PY)
        Q = (WPY**2).sum()

        # eigenvalues
        PW = self._P(W,gp)
        Lambda = W.T.dot(PW)
        lambdas = compute_eigenvals(Lambda)
        return davies_pvalue(Q, lambdas)
