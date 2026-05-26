from __future__ import annotations

import logging
from time import time
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import issparse
from sklearn.decomposition import NMF
from sklearn.utils.extmath import safe_sparse_dot

logger = logging.getLogger(__name__)

_LARGE = 1e100
_SMALL = 1e-10


class JointNMFWrapper:
    """
    Joint NMF decomposition for paired healthy / disease single-cell data.

    Parameters
    ----------
    Xh
        Dense matrix (cells x genes) for the healthy condition.
    Xd
        Dense matrix (cells x genes) for the disease condition.
        Must share the same gene dimension as ``Xh``.
    n_shared
        Number of programs shared between healthy and disease (KC in paper).
    n_healthy_specific
        Number of programs unique to the healthy condition (KH in paper).
    n_disease_specific
        Number of programs unique to the disease condition (KD in paper).
    gamma
        Coupling strength. Higher values force shared programs to align more.
    mu
        L2 regularisation on loading matrices. If None, estimated from data.
    n_init
        Number of random NMF initialisations; best reconstruction is kept.
    max_iters
        Maximum multiplicative-update iterations.
    tol
        Convergence tolerance on relative change in cost.
    random_state
        Random seed for reproducibility.
    """

    def __init__(
        self,
        Xh: np.ndarray,
        Xd: np.ndarray,
        *,
        n_shared: int = 10,
        n_healthy_specific: int = 5,
        n_disease_specific: int = 5,
        gamma: float = 1.0,
        mu: Optional[float] = None,
        n_init: int = 5,
        max_iters: int = 1000,
        tol: float = _SMALL,
        random_state: int = 0,
    ):
        self.n_shared = n_shared
        self.n_healthy_specific = n_healthy_specific
        self.n_disease_specific = n_disease_specific
        self.gamma = gamma
        self.max_iters = max_iters
        self.tol = tol

        self.Xh = sparse.csr_matrix(Xh, dtype=np.float64)
        self.Xd = sparse.csr_matrix(Xd, dtype=np.float64)
        _max_h = self.Xh.max()
        _max_d = self.Xd.max()
        if _max_h > 0:
            self.Xh = self.Xh / _max_h
        if _max_d > 0:
            self.Xd = self.Xd / _max_d

        nh_total = n_shared + n_healthy_specific
        nd_total = n_shared + n_disease_specific

        self.Wh, self.Hh = self._best_nmf(self.Xh, nh_total, n_init, random_state)
        self.Wd, self.Hd = self._best_nmf(self.Xd, nd_total, n_init, random_state + 1)

        self._align()

        if mu is None:
            diff_h = 0.5 * self._frob(self.Xh - self.Wh.dot(self.Hh)) ** 2
            diff_d = 0.5 * self._frob(self.Xd - self.Wd.dot(self.Hd)) ** 2
            denom = self._frob(self.Wh) ** 2 + self._frob(self.Wd) ** 2
            self.mu = (diff_h + diff_d) / (denom + _SMALL)
        else:
            self.mu = mu

    def fit(self) -> "JointNMFWrapper":
        """Run multiplicative updates until convergence."""
        t0 = time()
        chi2 = self._cost()
        old_chi2 = _LARGE
        niter = 0

        while niter < self.max_iters and abs((old_chi2 - chi2) / old_chi2) > self.tol:
            self._update_Wh()
            self._update_Hh()
            self._update_Wd()
            self._update_Hd()

            old_chi2 = chi2
            chi2 = self._cost()

            if not np.isfinite(chi2):
                raise ValueError("Joint NMF diverged (NaN/Inf cost). Check input data.")

            if niter % 50 == 0:
                logger.debug(
                    f"JointNMF iter {niter}: cost={chi2:.4f}, "
                    f"Δ={100*(old_chi2-chi2)/old_chi2:.3f}%"
                )
            niter += 1

        elapsed = (time() - t0) / 60.0
        logger.info(f"Joint NMF converged after {niter} iterations ({elapsed:.2f} min)")
        return self

    @property
    def Wh(self) -> np.ndarray:
        """Healthy cell x factor loadings (dense)."""
        return self._Wh.toarray() if issparse(self._Wh) else self._Wh

    @Wh.setter
    def Wh(self, value):
        self._Wh = sparse.csr_matrix(value)

    @property
    def Wd(self) -> np.ndarray:
        return self._Wd.toarray() if issparse(self._Wd) else self._Wd

    @Wd.setter
    def Wd(self, value):
        self._Wd = sparse.csr_matrix(value)

    @property
    def Hh(self) -> np.ndarray:
        return self._Hh.toarray() if issparse(self._Hh) else self._Hh

    @Hh.setter
    def Hh(self, value):
        self._Hh = sparse.csr_matrix(value)

    @property
    def Hd(self) -> np.ndarray:
        return self._Hd.toarray() if issparse(self._Hd) else self._Hd

    @Hd.setter
    def Hd(self, value):
        self._Hd = sparse.csr_matrix(value)

    def _cost(self) -> float:
        Wshh = self._Wh[:, : self.n_shared]
        Wshd = self._Wd[:, : self.n_shared]
        d1 = 0.5 * self._frob(self.Xh - safe_sparse_dot(self._Wh, self._Hh)) ** 2
        d2 = 0.5 * self._frob(self.Xd - safe_sparse_dot(self._Wd, self._Hd)) ** 2
        d3 = (self.mu / 2) * (self._frob(self._Wh) ** 2 + self._frob(self._Wd) ** 2)
        d4 = (self.gamma / 2) * self._frob(Wshh - Wshd) ** 2
        return float(d1 + d2 + d3 + d4)

    def _update_Wh(self):
        scale = np.append(
            (self.gamma + self.mu) * np.ones(self.n_shared),
            self.mu * np.ones(self.n_healthy_specific),
        )
        Wshd = self._Wd[:, : self.n_shared]
        num1 = safe_sparse_dot(self.Xh, self._Hh.T)
        zeros = sparse.csr_matrix(
            np.zeros((self.Xh.shape[0], self.n_healthy_specific))
        )
        num2 = sparse.hstack([Wshd.multiply(self.gamma), zeros])
        den = (
            safe_sparse_dot(self._Wh, safe_sparse_dot(self._Hh, self._Hh.T))
            + safe_sparse_dot(self._Wh, np.diag(scale))
        )
        self._Wh = self._Wh.multiply((num1 + num2) / (den + _SMALL)).tocsr()

    def _update_Hh(self):
        num = safe_sparse_dot(self._Wh.T, self.Xh)
        den = safe_sparse_dot(safe_sparse_dot(self._Wh.T, self._Wh), self._Hh)
        self._Hh = self._Hh.multiply(num / (den + _SMALL)).tocsr()

    def _update_Wd(self):
        scale = np.append(
            (self.gamma + self.mu) * np.ones(self.n_shared),
            self.mu * np.ones(self.n_disease_specific),
        )
        Wshh = self._Wh[:, : self.n_shared]
        num1 = safe_sparse_dot(self.Xd, self._Hd.T)
        zeros = np.zeros((self.Xd.shape[0], self.n_disease_specific))
        num2 = sparse.hstack([Wshh.multiply(self.gamma), sparse.csr_matrix(zeros)])
        den = (
            safe_sparse_dot(self._Wd, safe_sparse_dot(self._Hd, self._Hd.T))
            + safe_sparse_dot(self._Wd, np.diag(scale))
        )
        self._Wd = self._Wd.multiply((num1 + num2) / (den + _SMALL)).tocsr()

    def _update_Hd(self):
        num = safe_sparse_dot(self._Wd.T, self.Xd)
        den = safe_sparse_dot(safe_sparse_dot(self._Wd.T, self._Wd), self._Hd)
        self._Hd = self._Hd.multiply(num / (den + _SMALL)).tocsr()

    @staticmethod
    def _best_nmf(
        X: sparse.spmatrix, n_components: int, n_init: int, seed: int
    ) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
        best_err = _LARGE
        best_W = best_H = None
        for i in range(n_init):
            model = NMF(n_components=n_components, random_state=seed + i)
            W = model.fit_transform(X)
            if model.reconstruction_err_ < best_err:
                best_err = model.reconstruction_err_
                best_W = sparse.csr_matrix(W)
                best_H = sparse.csr_matrix(model.components_)
        return best_W, best_H

    def _align(self):
        """Reorder columns of Wh and Wd so shared programs are best-matched."""
        Wh_dense = self._Wh.toarray()
        Wd_dense = self._Wd.toarray()

        corr = np.corrcoef(Wh_dense.T, Wd_dense.T)
        n_h = Wh_dense.shape[1]
        n_d = Wd_dense.shape[1]
        corr = corr[:n_h, n_h:]  # shape (n_h, n_d)

        reorder_h, reorder_d = [], []
        corr_copy = corr.copy()
        for _ in range(min(n_h, n_d)):
            argmax = np.argmax([corr_copy[i, j] for i, j in enumerate(corr_copy.argmax(axis=1))])
            i = argmax
            j = corr_copy.argmax(axis=1)[argmax]
            reorder_h.append(i)
            reorder_d.append(j)
            corr_copy[i, :] = -2
            corr_copy[:, j] = -2

        reorder_h += [k for k in range(n_h) if k not in reorder_h]
        reorder_d += [k for k in range(n_d) if k not in reorder_d]

        self._Wh = sparse.csr_matrix(Wh_dense[:, reorder_h])
        self._Hh = sparse.csr_matrix(self._Hh.toarray()[reorder_h, :])
        self._Wd = sparse.csr_matrix(Wd_dense[:, reorder_d])
        self._Hd = sparse.csr_matrix(self._Hd.toarray()[reorder_d, :])

    @staticmethod
    def _frob(M) -> float:
        return sparse.linalg.norm(M, ord="fro") if issparse(M) else np.linalg.norm(M, "fro")