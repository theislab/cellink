from __future__ import annotations

from typing import Literal

import anndata
import numpy as np
import scipy.linalg as la
import scipy.stats as st

from cellink._core import DonorData
from cellink.at.base_model import align_to_index, fetch_raw_slot, observation_index, to_numpy, variant_matrix
from cellink.at.utils import ensure_float64_array

__all__ = ["GWAS"]


class GWAS:
    """Linear model for univariate association testing between `1` phenotypes and `S` inputs (`1`x`S` tests)"""

    def __init__(
        self,
        Y: str,
        F: str | None = None,
        *,
        data: DonorData | anndata.AnnData,
        target_level: Literal["donor", "cell"] | None = None,
    ) -> None:
        """
        Initialize the GWAS class.

        Parameters
        ----------
            Y : str
                outputs. A formula string (e.g. ``"phenotype"``) resolved
                against `data`.
            F : str, optional
                covariates. If not specified, an intercept-only column is used.
                A formula string (e.g. ``"age + sex"``) resolved against `data`;
                an intercept column is kept for the formula.
            data : DonorData or AnnData
                Data container `Y`/`F` are resolved against.
            target_level : {"donor", "cell"}, optional
                Required when `data` is a `DonorData`, since a `DonorData` has
                values at both levels.

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
        if isinstance(data, DonorData) and target_level == "cell":
            raise ValueError(
                "GWAS reads variants from `dd.G.X`, which is one row per donor, so a cell-level "
                "phenotype would pair each genotype with many cells and treat them as independent "
                'observations. Aggregate the phenotype instead (Y="dmean(<gene>)", '
                'target_level="donor"), or use StructLMM, which models the cell-level structure.'
            )

        Y_df = fetch_raw_slot(data, Y, "Y", target_level=target_level, add_intercept=False)
        # remember which observations the null is fitted on, so `test_association` can
        # verify that whatever it is handed lines up with them
        self._obs_index = None if Y_df.attrs.get("has_dummy_index", False) else Y_df.index
        Y = to_numpy(Y_df)

        if F is None:
            F = np.ones((Y.shape[0], 1))
        else:
            F = to_numpy(fetch_raw_slot(data, F, "F", target_level=target_level, add_intercept=True))

        # sanity checks
        assert isinstance(Y, np.ndarray) and Y.ndim == 2, "Y (resolved from formula) must be a 2D numpy array"
        assert isinstance(F, np.ndarray) and F.ndim == 2, "F (resolved from formula) must be a 2D numpy array"
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

    def test_association(self, data: DonorData | anndata.AnnData) -> None:
        """Test association between the phenotype and every variant in `data`.

        Each variant is a column of `data.G.X` (DonorData) or `data.X` (AnnData), and
        is tested independently of the others, one by one. The test is a likelihood
        ratio test,

        .. math::
            LRT = -df * log( s2 / s20 )

        where s2 is the residual variance of the model including the variant and s20
        that of the null model. Uses the Woodbury matrix identity to avoid re-inverting
        the design matrix for each variant.

        Parameters
        ----------
        data : DonorData | anndata.AnnData
            input data
        """
        G = align_to_index(variant_matrix(data), observation_index(data), self._obs_index)

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
        Get effect size SNPs

        Returns
        -------
        beta : ndarray
        """
        return self.beta_g

    def getLRT(
        self,
    ) -> np.ndarray:
        """
        Get lik ratio test statistics

        Returns
        -------
        lrt : ndarray
        """
        return self.lrt

    def getBetaSNPste(
        self,
    ) -> np.ndarray:
        """
        Get standard errors on betas

        Returns
        -------
        beta_ste : ndarray
        """
        beta = self.getBetaSNP()
        pv = self.getPv()
        z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
        ste = beta / z
        return ste
