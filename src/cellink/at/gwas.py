import numpy as np
import scipy.linalg as la
import scipy.stats as st
from src.cellink.at.utils import ensure_float64_array


class GWAS:
    """Linear model for univariate association testing between `1` phenotypes and `S` inputs (`1`x`S` tests)"""

    def __init__(
        self,
        Y: np.ndarray,
        F: np.ndarray = None
    ) -> None:
        """
        Initialize the GWAS class.

        Parameters
        ----------
            Y : (`N`, `1`) ndarray
                outputs
            F : (`N`, `K`) ndarray
                covariates. If not specified, an intercept is assumed.

        Notes
        -----
            To check if:
                * Y is a numpy array
                * Y has two dimensions (either a column vector to model intercept or a matrix with covariates)
            To check if:
                * F is a numpy array
                * F has two dimensions (either a column vector to model intercept or a matrix with covariates)
                * F has the same number of rows as Y
        """
        # sanity checks
        assert isinstance(Y, np.ndarray), "Y must be a numpy array"
        assert Y.ndim == 2, "Y must be a 2D numpy array"

        if F is None:
            F = np.ones((Y.shape[0], 1))

        assert isinstance(F, np.ndarray), "F must be a numpy array"
        assert F.ndim == 2, "F must be a 2D numpy array"
        assert Y.shape[0] == F.shape[0], "Y and F must have the same number of rows"

        # type casting
        Y = ensure_float64_array(Y)
        F = ensure_float64_array(F)

        self.Y = Y
        self.F = F

        self.df = Y.shape[0] - F.shape[1]  # degrees of freedom, N-K

        self._fit_null()  # fit the null model

    def _fit_null(
        self,
    ) -> None:
        """Internal functon.

        Fits the null model (i.e. the model without the covariate for which we want to test the association).
        The null model is fitted using the closed form solution of the linear model.
        One of the several possible source of the closed form solution is:
        Hastie, Trevor,
        Tibshirani, Robert and Friedman, Jerome.
        The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc., 2017.
        """
        # Information that we need to compute the null model
        self.FY = np.dot(self.F.T, self.Y)
        self.FF = np.dot(self.F.T, self.F)
        self.YY = np.einsum("ip,ip->p", self.Y, self.Y)

        # calc beta_F0 and s20
        self.A0i = la.inv(self.FF)
        self.beta_F0 = np.dot(self.A0i, self.FY)
        self.s20 = (self.YY - np.einsum("kp,kp->p", self.FY, self.beta_F0)) / self.df

    def test_association(
        self,
        G: np.ndarray
    ) -> None:
        """Test association between phenotype and genotype matrix.

        Each column of G is tested independently from the others.
        The test is performed using the likelihood ratio test (LRT) statistic.
        The LRT statistic is computed as:
        .. math::
            LRT = -df * log( marginal likelihood under H1 / marginal likelihood under H0 )
        where s2 is the variance of the residuals of the model with the covariate and s20 is the variance of the residuals of the null model.
        Uses the Woodbury Matrix Identity to invert the matrix in the LRT statistic.
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
            inputs
        """
        # type casting
        G = ensure_float64_array(G)

        # precompute products
        GY = np.dot(G.T, self.Y)
        GG = np.einsum("ij,ij->j", G, G)
        FG = np.dot(self.F.T, G)

        # Let us denote the inverse of Areml as
        # Ainv = [[A0i + m mt / n, m], [mT, n]]

        # Note that here there is a trick to avoid computing the inverse of the matrix Ainv every time, the Woodbury matrix identity is used.
        A0iFG = np.dot(self.A0i, FG)
        n = 1.0 / (GG - np.einsum("ij,ij->j", FG, A0iFG))
        M = -n * A0iFG
        self.beta_F = self.beta_F0[:, None, :] + np.einsum("ks,sp->ksp", M, np.dot(M.T, self.FY)) / n[None, :, None]
        self.beta_F += np.einsum("ks,sp->ksp", M, GY)
        self.beta_g = np.einsum("ks,kp->sp", M, self.FY)
        self.beta_g += n[:, None] * GY

        # Compute variance of the alternative model based on the residuals
        self.s2 = self.YY - np.einsum("kp,ksp->sp", self.FY, self.beta_F)
        self.s2 -= GY * self.beta_g
        self.s2 /= self.df

        # Perform the likelihood ratio test and compute p-values. The survival function of the chi2 distribution is used to compute the p-values (more numerically stable than the cumulative distribution function).
        self.lrt = -self.df * np.log(self.s2 / self.s20)
        self.pv = st.chi2(1).sf(self.lrt)

    def getPv(
        self,
    ) -> np.ndarray:
        """
        Get pvalues

        Returns
        -------
        pv : ndarray
        """
        return self.pv

    def getBetaSNP(
        self,
    ) -> np.ndarray:
        """
        get effect size SNPs

        Returns
        -------
        beta : ndarray
        """
        return self.beta_g

    def getLRT(
        self,
    ) -> np.ndarray:
        """
        get lik ratio test statistics

        Returns
        -------
        lrt : ndarray
        """
        return self.lrt

    def getBetaSNPste(
        self,
    ) -> np.ndarray:
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
