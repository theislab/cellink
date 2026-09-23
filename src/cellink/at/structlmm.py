from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import anndata
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse
from tqdm import tqdm

from cellink._core import DonorData
from cellink.at.base_model import fetch_raw_slot, to_numpy
from cellink.at.utils import compute_eigenvals, davies_pvalue, ensure_float64_array

if TYPE_CHECKING:
    from limix_core.gp import GP2KronSumLR

__all__ = ["StructLMM"]

logger = logging.getLogger(__name__)


class StructLMM:
    """Faster version of StructLMM."""

    def __init__(
        self,
        y: str,
        E: str,
        F: str | None = None,
        verbose: bool = False,
        *,
        data: DonorData | anndata.AnnData,
        target_level: Literal["donor", "cell"] | None = None,
    ) -> None:
        """Initialize the StructLMM class.

        Parameters
        ----------
        y : str
            Phenotype: a formula string or bare column name resolved against `data`.
        E : str
            Environment (context) design, resolved against `data` as an
            ``(n_samples, n_environments)`` matrix -- not a covariance. A one-hot of a
            cell state is ``"cell_state - 1"``; per-donor cell-type proportions are
            ``"dmean(celltype) - 1"``. No intercept column is added, since a constant
            environment carries no interaction.
        F : str, optional
            Covariates, resolved against `data` with an intercept column kept. If
            omitted, an intercept-only mean model is used.
        verbose: bool, optional
            Show a progress bar over variants on the exact path.
        data : DonorData or AnnData
            Container `y`/`E`/`F` are resolved against.
        target_level : {"donor", "cell"}, optional
            Level to resolve at. Required for a `DonorData`: unlike GWAS and Skat, a
            GxE model is meaningful at either level (a cell-state `E` or a donor-level
            one), so there is no sensible default to pick for you.
        """
        assert isinstance(data, anndata.AnnData | DonorData), "data must be an anndata.AnnData or a DonorData"
        assert isinstance(y, str), "y must be a string"
        assert isinstance(E, str), "E must be a string"

        y = to_numpy(fetch_raw_slot(data, y, "y", target_level=target_level, add_intercept=False))
        E = to_numpy(fetch_raw_slot(data, E, "E", target_level=target_level, add_intercept=False))
        if F is None:
            F = np.ones((y.shape[0], 1))
        else:
            F = to_numpy(fetch_raw_slot(data, F, "F", target_level=target_level, add_intercept=True))

        # type casting
        y = ensure_float64_array(y)
        E = ensure_float64_array(E)
        F = ensure_float64_array(F)

        assert (
            y.shape[0] == E.shape[0] == F.shape[0]
        ), f"y, E and F must have the same number of rows, got {y.shape[0]}, {E.shape[0]}, {F.shape[0]}"

        self.y = y
        self.E = E
        self.F = F

        self._data = data
        self._target_level = target_level
        self.verbose = verbose

    def _variants(self, data: DonorData | anndata.AnnData) -> np.ndarray:
        """Pull the variant matrix out of `data`, aligned with the rows of `y`."""
        if not isinstance(data, anndata.AnnData | DonorData):
            raise TypeError(
                f"expected a DonorData or AnnData, got {type(data).__name__}. "
                "Wrap a derived matrix in an AnnData before testing it."
            )
        X = data.G.X if isinstance(data, DonorData) else data.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        elif hasattr(X, "compute"):  # dask, e.g. from read_sgkit_zarr
            X = X.compute()
        G = ensure_float64_array(X)

        if G.shape[0] == self.y.shape[0]:
            return G
        # A cell-level phenotype with donor-level variants: broadcast each donor's
        # genotype to its cells, the same expansion `crepeat()` performs in a formula.
        if isinstance(data, DonorData) and G.shape[0] == data.G.n_obs and self.y.shape[0] == data.C.n_obs:
            donor_ids = data.C.obs[data.donor_id]
            if isinstance(donor_ids.dtype, pd.CategoricalDtype):
                donor_ids = donor_ids.astype(donor_ids.cat.categories.dtype)
            return G[data.G.obs_names.get_indexer(donor_ids), :]
        raise ValueError(f"variants have {G.shape[0]} rows but y has {self.y.shape[0]}")

    def interaction_test(
        self,
        data: DonorData | anndata.AnnData,
        exact: bool = False,
    ) -> np.ndarray:
        """Test the genotype x environment interaction for every variant in `data`.

        The variant set is `data.G.X` for a `DonorData` and `data.X` for an `AnnData`;
        subset the object beforehand to test a region. When the phenotype is cell-level
        and the variants are donor-level, each donor's genotype is broadcast to its
        cells.

        Parameters
        ----------
        data : DonorData or AnnData
            Source of the variants to test.
        exact : bool, optional
            If True, perform the exact test. If False, perform the approximate test using GPs.
        """
        try:
            from limix_core.covar import FreeFormCov
            from limix_core.gp import GP2KronSumLR
        except ImportError as e:
            raise ImportError("StructLMM requires `limix-core`. Install it with:\n\n    pip install limix-core") from e

        G = self._variants(data)

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
        self.info_opt = gp.optimize(verbose=False)

        try:
            from limix_lmm import LMM
        except ImportError as e:
            raise ImportError(
                "The approximate (non-exact) interaction test requires `limix-lmm`. "
                "Install it with:\n\n    pip install limix-lmm"
            ) from e

        # fit null
        self.lmm = LMM(self.y, self.F, gp.covar.solve)
        self.lmm.process(G)
        pv = self.lmm.getPv()  # noqa: F841
        beta = self.lmm.getBetaSNP()
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
        try:
            from limix_core.covar import FreeFormCov
            from limix_core.gp import GP2KronSumLR
        except ImportError as e:
            raise ImportError("StructLMM requires `limix-core`. Install it with:\n\n    pip install limix-core") from e

        # type casting
        g = ensure_float64_array(g)

        # fit exact null model
        F1 = np.concatenate([self.F, g], 1)
        gp = GP2KronSumLR(Y=self.y, Cn=FreeFormCov(1), G=self.E, F=F1, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)

        # make interaction test
        PY = self._P(self.y, gp)

        # score statistics
        W = g * self.E
        WPY = W.T.dot(PY)
        Q = (WPY**2).sum()

        # eigenvalues
        PW = self._P(W, gp)
        Lambda = W.T.dot(PW)
        lambdas = compute_eigenvals(Lambda)
        return davies_pvalue(Q, lambdas)
