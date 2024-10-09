from time import time

import numpy as np
import scipy.linalg as la
import scipy.stats as st


class GWAS:
    r"""
    Linear model for univariate association testing
    between `P` phenotypes and `S` inputs (`P`x`S` tests)

    Parameters
    ----------
    Y : (`N`, `P`) ndarray
        outputs
    F : (`N`, `K`) ndarray
        covariates. If not specified, an intercept is assumed.
    """

    def __init__(self, Y, F=None):
        if F is None:
            F = np.ones((Y.shape[0], 1))
        self.Y = Y
        self.F = F
        self.df = Y.shape[0] - F.shape[1]
        self._fit_null()

    def _fit_null(self):
        """Internal functon. Fits the null model"""
        self.FY = np.dot(self.F.T, self.Y)
        self.FF = np.dot(self.F.T, self.F)
        self.YY = np.einsum("ip,ip->p", self.Y, self.Y)
        # calc beta_F0 and s20
        self.A0i = la.inv(self.FF)
        self.beta_F0 = np.dot(self.A0i, self.FY)
        self.s20 = (self.YY - np.einsum("kp,kp->p", self.FY, self.beta_F0)) / self.df

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
            inputs
        verbose : bool
            verbose flag.
        """
        t0 = time()
        # precompute some stuff
        GY = np.dot(G.T, self.Y)
        GG = np.einsum("ij,ij->j", G, G)
        FG = np.dot(self.F.T, G)

        # Let us denote the inverse of Areml as
        # Ainv = [[A0i + m mt / n, m], [mT, n]]
        A0iFG = np.dot(self.A0i, FG)
        n = 1.0 / (GG - np.einsum("ij,ij->j", FG, A0iFG))
        M = -n * A0iFG
        self.beta_F = (
            self.beta_F0[:, None, :]
            + np.einsum("ks,sp->ksp", M, np.dot(M.T, self.FY)) / n[None, :, None]
        )
        self.beta_F += np.einsum("ks,sp->ksp", M, GY)
        self.beta_g = np.einsum("ks,kp->sp", M, self.FY)
        self.beta_g += n[:, None] * GY

        # sigma
        self.s2 = self.YY - np.einsum("kp,ksp->sp", self.FY, self.beta_F)
        self.s2 -= GY * self.beta_g
        self.s2 /= self.df

        # dlml and pvs
        self.lrt = -self.df * np.log(self.s2 / self.s20)
        self.pv = st.chi2(1).sf(self.lrt)

        t1 = time()
        if verbose:
            print("Tested for %d variants in %.2f s" % (G.shape[1], t1 - t0))

    def getPv(self):
        """
        Get pvalues

        Returns
        -------
        pv : ndarray
        """
        return self.pv

    def getBetaSNP(self):
        """
        get effect size SNPs

        Returns
        -------
        beta : ndarray
        """
        return self.beta_g

    def getLRT(self):
        """
        get lik ratio test statistics

        Returns
        -------
        lrt : ndarray
        """
        return self.lrt

    def getBetaSNPste(self):
        """
        get standard errors on betas

        Returns
        -------
        beta_ste : ndarray
        """
        beta = self.getBetaSNP()
        pv = self.getPv()
        z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
        ste = beta / z
        return ste
