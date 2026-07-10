"""Sparse torch implementation of Seismic cell-type specificity + GWAS association.

Ported from ratpy (``ratpy.external.seismic``, branch ``seismic_implementation``,
commit ``9cb38af``, original author Antonio Nappi). This is a from-scratch
re-implementation of the seismicGWAS R package's core algorithm
(``calc_specificity`` + ``get_ct_trait_associations``) using sparse torch
operations, so it never materialises a dense genes x cells matrix. This
makes it usable on atlases with millions of cells, where :func:`run_seismic`
(the R/rpy2 subprocess backend, which densifies the full expression matrix
to export a CSV) runs out of memory.

Numerically validated against the R backend on a downsampled atlas subset
(see ``tests/test_seismic_torch.py``): cell-type p-values from both backends
are highly concordant (Spearman rho > 0.9), with small differences expected
from float32 vs. R's float64 arithmetic and minor epsilon-handling
differences in the variance denominator.
"""

import logging
import time
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.distributions as td
import torch.linalg as la
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import issparse
import scanpy as sc

logger = logging.getLogger(__name__)

__all__ = ["SparseScore", "RegressionNLL", "run_seismic_torch"]


class SparseScore(nn.Module):
    """Compute Seismic cell-type specificity scores from a sparse expression matrix.

    Parameters
    ----------
    E : torch.Tensor
        ``[M, G]`` (cells x genes) expression matrix (log-normalised). Dense
        or sparse; converted to a CSR sparse tensor internally so genes x
        cells is never densified.
    """

    def __init__(self, E: torch.Tensor):
        super().__init__()
        E = E.to_sparse_csr()

        device, dtype = E.device, E.dtype
        self.register_buffer("E_csr", E)
        self.register_buffer("E2_csr", self._square_csr(E))

        I = torch.sparse_csr_tensor(
            E.crow_indices(),
            E.col_indices(),
            torch.ones_like(E.values()),
            size=E.shape,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("I_csr", I)

        self.register_buffer("Et", self.E_csr.transpose(0, 1))
        self.register_buffer("Et2", self.E2_csr.transpose(0, 1))
        self.register_buffer("It", self.I_csr.transpose(0, 1))

    @staticmethod
    def _square_csr(A):
        return torch.sparse_csr_tensor(
            A.crow_indices(),
            A.col_indices(),
            A.values().clone().pow_(2),
            size=A.shape,
            device=A.device,
            dtype=A.dtype,
        )

    def forward(self, masks: torch.Tensor, return_all: bool = False, hacked: bool = False):
        """Compute specificity scores.

        Parameters
        ----------
        masks : torch.Tensor
            ``[M, C]`` binary cluster-membership matrix (cells x cell types).
        return_all : bool
            If True, also return the raw one-sided z-test probability and
            the in-group expression fraction.

        Returns
        -------
        torch.Tensor
            ``[G, C]`` specificity scores (genes x cell types).
        """
        eps = 0.0

        n_in = masks.sum(0) + eps
        n_out = (1 - masks).sum(0) + eps

        w_in = masks / n_in.unsqueeze(0)
        w_out = (1 - masks) / n_out.unsqueeze(0)

        mu_in = torch.sparse.mm(self.Et, w_in).T
        mu_out = torch.sparse.mm(self.Et, w_out).T
        ex2_in = torch.sparse.mm(self.Et2, w_in).T
        ex2_out = torch.sparse.mm(self.Et2, w_out).T

        var_in = (ex2_in - mu_in**2) * (n_in[:, None] / (n_in[:, None] - 1.0 + eps))
        var_out = (ex2_out - mu_out**2) * (n_out[:, None] / (n_out[:, None] - 1.0 + eps))

        denom = torch.sqrt(var_in / (n_in[:, None] + eps) + var_out / (n_out[:, None] + eps))
        z = (mu_in - mu_out) / denom
        p_torch = td.Normal(0, 1, validate_args=False).cdf(z)  # [C, G]

        Im = torch.sparse.mm(self.It, masks)  # [G, C]
        r_in = (Im / (n_in.unsqueeze(0) + eps)).T  # [C, G]

        pr = p_torch * r_in  # [C, G]
        pr_sum = pr.sum(0, keepdim=True)  # [1, G]
        valid = pr_sum > 0

        denom_safe = torch.where(valid, pr_sum, torch.ones_like(pr_sum))
        hacked_t = pr.new_tensor(hacked, dtype=torch.bool)
        s = torch.where(hacked_t, pr, pr / denom_safe)
        s = s * valid.to(s.dtype)

        s = s.T  # [G, C]
        if return_all:
            return s, p_torch, r_in
        return s


class RegressionNLL(nn.Module):
    """Closed-form univariate regression of GWAS z-scores against a gene-level covariate.

    Equivalent to fitting, independently for each column ``s`` of ``G``:
    ``z = beta0 + beta1 * G[:, s] + eps``, via a likelihood-ratio test against
    the intercept-only null model. This is the association test seismicGWAS
    uses to link cell-type specificity scores with MAGMA gene-level GWAS
    z-scores.

    Parameters
    ----------
    z : torch.Tensor
        ``[N]`` 1D vector of per-gene GWAS z-scores (e.g. MAGMA ``ZSTAT``).
    """

    def __init__(self, z: torch.Tensor):
        super().__init__()
        assert z.ndim == 1, "z must be 1D [N]"
        self.register_buffer("Y", z[:, None])
        self.F = torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
        self.df = z.shape[0] - self.F.shape[1]
        self._fit_null()

    def _fit_null(self):
        F, Y = self.F, self.Y
        self.FY = F.T @ Y
        self.FF = F.T @ F
        self.YY = (Y * Y).sum(0)
        self.A0i = la.inv(self.FF)
        self.beta_F0 = self.A0i @ self.FY
        self.s20 = (self.YY - (self.FY * self.beta_F0).sum(0)) / self.df

    def forward(self, G: torch.Tensor, verbose: bool = False, return_all: bool = False):
        """Regress ``self.Y`` against each column of ``G``.

        Parameters
        ----------
        G : torch.Tensor
            ``[N, S]`` per-gene covariate matrix (e.g. cell-type specificity
            scores), aligned row-for-row with the ``z`` passed at init.
        return_all : bool
            If True, also return the one-sided p-value, effect size, and
            standard error per column of ``G``.
        """
        F, Y = self.F, self.Y
        df = self.df
        t0 = time.time()

        GY = G.T @ Y
        GG = (G * G).sum(0)
        FG = F.T @ G
        A0iFG = self.A0i @ FG
        n = 1.0 / (GG - (FG * A0iFG).sum(0))
        M = -n * A0iFG

        self.beta_F = (
            self.beta_F0[:, None]
            + torch.einsum("ks,sp->ksp", M, M.T * self.FY) / n[None, :, None]
        )
        self.beta_F += torch.einsum("ks,sp->ksp", M, GY)
        self.beta_g = torch.einsum("ks,kp->sp", M, self.FY)
        self.beta_g += n[:, None] * GY
        self.s2 = self.YY - torch.einsum("kp,ksp->sp", self.FY, self.beta_F)
        self.s2 -= GY * self.beta_g
        self.s2 /= df

        self.lrt = -df * torch.log(self.s2 / self.s20)
        nll = -self.lrt.sum()

        if verbose:
            logger.info(f"RegressionNLL: {G.shape[1]} columns in {time.time() - t0:.2f}s")

        if return_all:
            pval_two_sided = torch.tensor(
                st.chi2(1).sf(self.lrt.cpu().data.numpy()), device=self.F.device
            )
            pval_one_sided = torch.where(
                self.beta_g > 0, pval_two_sided / 2.0, 1.0 - (pval_two_sided / 2.0)
            )
            z = np.sign(self.beta_g.cpu().data.numpy()) * np.sqrt(
                st.chi2.ppf(1.0 - pval_two_sided.cpu().data.numpy(), df=1)
            )
            z = torch.tensor(z, device=self.F.device)
            ste = self.beta_g / z
            return nll, pval_one_sided, self.beta_g, ste

        return nll


def _adata_to_sparse_csr_tensor(adata: AnnData, layer: str | None, dtype=torch.float32) -> torch.Tensor:
    """Build a torch sparse CSR tensor [cells, genes] directly from adata, no densification."""
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if not issparse(X):
        return torch.tensor(np.asarray(X), dtype=dtype)
    X = X.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr.astype(np.int64)),
        torch.from_numpy(X.indices.astype(np.int64)),
        torch.from_numpy(X.data.astype(np.float32)).to(dtype),
        size=X.shape,
    )


def run_seismic_torch(
    adata: AnnData,
    magma_file: Union[str, Path],
    cell_type_col: str,
    species: Literal["human", "mouse"] = "human",
    layer: str | None = None,
    min_genes: int = 250,
    min_cells: int = 50,
    magma_gene_col: str = "GENE",
    magma_z_col: str = "ZSTAT",
    device: str = "cpu",
    prefix: str | None = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """Run the sparse-torch Seismic backend end to end.

    Drop-in alternative to :func:`run_seismic` (the R/seismicGWAS subprocess
    backend) that never densifies the genes x cells matrix, so it scales to
    atlases with millions of cells. Returns a DataFrame with the same
    ``cell_type`` / ``pvalue`` / ``FDR`` columns as :func:`run_seismic`, so
    downstream code (e.g. plotting, Cauchy combination) is backend-agnostic.

    Parameters
    ----------
    adata : AnnData
        Single-cell data. ``species="mouse"`` is not auto-translated (unlike
        the R backend) — pass an adata already indexed by the target species'
        gene symbols if you need cross-species gene mapping.
    magma_file : str or Path
        MAGMA ``.genes.out`` file with columns ``magma_gene_col``/``magma_z_col``.
    cell_type_col : str
        Column in ``adata.obs`` with cell-type labels.
    layer : str, optional
        ``adata.layers`` key to use as expression. Defaults to ``adata.X``.
    device : str, default "cpu"
        torch device. Sparse ops here are CPU-friendly; only use "cuda" for
        very large atlases where a GPU is free.
    prefix : str, optional
        If given and ``save_results``, associations are written to
        ``{prefix}_associations.tsv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_type``, ``pvalue``, ``beta``, ``se``, ``FDR``.
    """
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs")
    if species == "mouse":
        logger.warning(
            "run_seismic_torch does not auto-translate mouse gene IDs "
            "(unlike the R backend's translate_gene_ids). Pre-map adata.var_names "
            "to human orthologs before calling this function."
        )

    adata = adata[~adata.obs[cell_type_col].isna()].copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if "log1p" not in adata.uns_keys():
        logger.info("run_seismic_torch: log-normalising data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    logger.info(f"run_seismic_torch: building sparse tensor ({adata.n_obs:,} cells x {adata.n_vars:,} genes)")
    Et = _adata_to_sparse_csr_tensor(adata, layer).to(device)

    cell_types = sorted(adata.obs[cell_type_col].astype(str).unique())
    ct_codes = adata.obs[cell_type_col].astype(str).map({ct: i for i, ct in enumerate(cell_types)}).values
    mask = torch.zeros((adata.n_obs, len(cell_types)), dtype=torch.float32, device=device)
    mask[torch.arange(adata.n_obs), torch.tensor(ct_codes, dtype=torch.long)] = 1.0

    logger.info(f"run_seismic_torch: computing specificity scores for {len(cell_types)} cell types")
    score_module = SparseScore(Et)
    with torch.no_grad():
        scores = score_module(mask)  # [G, C]

    scores_df = pd.DataFrame(
        scores.cpu().numpy(), index=adata.var_names, columns=cell_types
    )

    magma_df = pd.read_csv(magma_file, sep=r"\s+")
    if magma_gene_col not in magma_df.columns or magma_z_col not in magma_df.columns:
        raise ValueError(
            f"MAGMA file must have columns '{magma_gene_col}' and '{magma_z_col}'; "
            f"found {list(magma_df.columns)}"
        )
    magma_df = magma_df.set_index(magma_gene_col)

    shared_genes = scores_df.index.intersection(magma_df.index)
    if len(shared_genes) < 200:
        raise ValueError(
            f"Only {len(shared_genes)} genes shared between expression data and MAGMA output "
            "— check that gene identifiers match (gene symbols vs Ensembl IDs)."
        )
    logger.info(f"run_seismic_torch: {len(shared_genes)} genes shared with MAGMA output")

    scores_aligned = scores_df.loc[shared_genes]
    magma_aligned = magma_df.loc[shared_genes]

    Zt = torch.tensor(magma_aligned[magma_z_col].values, dtype=torch.float32, device=device)
    G = torch.tensor(scores_aligned.values.astype(np.float32), dtype=torch.float32, device=device)

    logger.info("run_seismic_torch: running per-cell-type association test")
    reg = RegressionNLL(Zt)
    with torch.no_grad():
        _, pval_one_sided, beta_g, ste = reg(G, return_all=True)

    pvals = pval_one_sided.cpu().numpy().ravel()
    betas = beta_g.cpu().numpy().ravel()
    stes = ste.cpu().numpy().ravel()

    from statsmodels.stats.multitest import multipletests

    _, fdr, _, _ = multipletests(pvals, method="fdr_bh")

    associations_df = pd.DataFrame({
        "cell_type": cell_types,
        "pvalue": pvals,
        "beta": betas,
        "se": stes,
        "FDR": fdr,
    }).sort_values("pvalue").reset_index(drop=True)

    if save_results and prefix:
        out_file = f"{prefix}_associations.tsv"
        associations_df.to_csv(out_file, sep="\t", index=False)
        logger.info(f"run_seismic_torch: saved {out_file}")

    return associations_df
