import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
import torch
import torch.linalg as la
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

__all__ = ["SparseScore", "RegressionNLL", "run_seismic_torch"]


class SparseScore(nn.Module):
    def __init__(self, E):
        super().__init__()
        import scipy.sparse as sp_scipy

        E = E.to_sparse_csr()
        device = E.device
        M, G = E.shape

        # Build transposed sparse matrices on CPU (avoids CUDA format-conversion
        # buffers during every forward pass, which can be nnz * 12 bytes each).
        # scipy CSR.T.tocsr() is efficient: it never materialises a dense matrix.
        crow_np = E.crow_indices().cpu().numpy()
        col_np = E.col_indices().cpu().numpy()
        vals_f32 = E.values().cpu().float().numpy()  # float32 to halve GPU memory
        vals_sq = vals_f32**2

        E_sp = sp_scipy.csr_matrix((vals_f32, col_np, crow_np), shape=(M, G))
        E2_sp = sp_scipy.csr_matrix((vals_sq, col_np, crow_np), shape=(M, G))
        I_sp = sp_scipy.csr_matrix((np.ones(len(col_np), dtype=np.float32), col_np, crow_np), shape=(M, G))

        def _scipy_to_torch_csr_t(mat, dev):
            """Transpose [M,G] scipy CSR → [G,M] torch CSR on *dev*."""
            mat_t = mat.T.tocsr()
            return torch.sparse_csr_tensor(
                torch.from_numpy(mat_t.indptr.astype("int64")),
                torch.from_numpy(mat_t.indices.astype("int64")),
                torch.from_numpy(mat_t.data.astype(np.float32)),
                size=(G, M),
                dtype=torch.float32,
            ).to(dev)

        self.register_buffer("Et_csr", _scipy_to_torch_csr_t(E_sp, device))
        self.register_buffer("Et2_csr", _scipy_to_torch_csr_t(E2_sp, device))
        self.register_buffer("It_csr", _scipy_to_torch_csr_t(I_sp, device))

        with torch.no_grad():
            ones = torch.ones(M, 1, dtype=torch.float32, device=device)
            E_sum = torch.sparse.mm(self.Et_csr, ones)  # [G, 1]
            E2_sum = torch.sparse.mm(self.Et2_csr, ones)  # [G, 1]
        self.register_buffer("E_sum", E_sum)
        self.register_buffer("E2_sum", E2_sum)
        self._M = M

    def forward_sparse(self, masks: torch.Tensor, return_all: bool = False):
        """
        2-D forward that accepts a sparse mask [M, C] in any sparse layout.

        Non-COO layouts (CSR/CSC/BSR/BSC) are converted to COO first, since the
        index-scatter below needs explicit row/column indices.

        Memory savings vs forward():
          - w_in is built by index-scatter (no .to_dense() on the input mask)
          - mu_out / ex2_out are computed analytically from E_sum / E2_sum,
            eliminating two sparse.mm calls and their backward Et_csr.T
            format-conversion allocations (~6.8 GiB each on large datasets).
          - It_csr @ w_in reuses the already-built w_in dense matrix.
        """
        eps = 1e-8
        if masks.layout is not torch.sparse_coo:
            masks = masks.to_sparse_coo()
        masks = masks.coalesce()
        if masks.dim() != 2:
            raise ValueError(f"sparse masks must be 2D, got {tuple(masks.shape)}")
        M, C = masks.shape
        device = masks.device

        idx = masks.indices()  # [2, nnz]
        row_idx, col_idx = idx[0], idx[1]
        vals = masks.values().to(torch.float32)

        # n_in [C], n_out [C]
        n_in = torch.zeros(C, device=device, dtype=torch.float32)
        n_in.scatter_add_(0, col_idx, vals)
        n_in = n_in + eps
        n_out = M - n_in + eps  # (1-mask).sum(0) + eps, exact for any mask

        # w_in [M, C] dense — built from sparse indices, never calls .to_dense()
        w_in = torch.zeros(M, C, device=device, dtype=torch.float32)
        w_in[row_idx, col_idx] = vals / n_in[col_idx]

        # mu_in, ex2_in via sparse.mm — these carry gradients
        mu_in = torch.sparse.mm(self.Et_csr, w_in).T  # [C, G]
        ex2_in = torch.sparse.mm(self.Et2_csr, w_in).T  # [C, G]

        # mu_out, ex2_out analytically from cached buffers — avoids 2 sparse.mm
        mu_out = (self.E_sum.T - n_in[:, None] * mu_in) / n_out[:, None]  # [C, G]
        ex2_out = (self.E2_sum.T - n_in[:, None] * ex2_in) / n_out[:, None]  # [C, G]

        var_in = (ex2_in - mu_in**2).clamp_min(0.0) * (n_in[:, None] / (n_in[:, None] - 1.0 + eps))
        var_out = (ex2_out - mu_out**2).clamp_min(0.0) * (n_out[:, None] / (n_out[:, None] - 1.0 + eps))

        denom = torch.sqrt(var_in / (n_in[:, None] + eps) + var_out / (n_out[:, None] + eps))
        z = (mu_in - mu_out) / (denom + eps)
        p_torch = torch.special.ndtr(z)  # [C, G]

        # r_in = (It_csr @ mask / n_in).T = (It_csr @ w_in).T  (w_in = mask/n_in)
        Im = torch.sparse.mm(self.It_csr, w_in)  # [G, C]
        r_in = Im.T  # [C, G]

        pr = p_torch * r_in  # [C, G]
        pr_sum = pr.sum(0, keepdim=True)  # [1, G]
        valid = pr_sum > 0

        denom_safe = torch.where(valid, pr_sum, torch.ones_like(pr_sum))
        s = torch.where(valid, pr / denom_safe, torch.zeros_like(pr))

        s = s.T  # [G, C]
        if return_all:
            return s, p_torch, r_in
        return s

    def forward(self, masks: torch.Tensor, return_all: bool = False):
        if masks.layout is not torch.strided:
            return self.forward_sparse(masks, return_all=return_all)

        eps = 1e-8

        if masks.dim() == 2:
            masks = masks.to(torch.float32)
            M_cells = masks.shape[0]
            n_in_raw = masks.sum(0)
            n_in = n_in_raw + eps
            n_out = M_cells - n_in_raw + eps

            w_in = masks / n_in.unsqueeze(0)

            mu_in = torch.sparse.mm(self.Et_csr, w_in).T  # [C, G]
            ex2_in = torch.sparse.mm(self.Et2_csr, w_in).T  # [C, G]

            mu_out = (self.E_sum.T - n_in[:, None] * mu_in) / n_out[:, None]
            ex2_out = (self.E2_sum.T - n_in[:, None] * ex2_in) / n_out[:, None]

            var_in = (ex2_in - mu_in**2).clamp_min(0.0) * (n_in[:, None] / (n_in[:, None] - 1.0 + eps))
            var_out = (ex2_out - mu_out**2).clamp_min(0.0) * (n_out[:, None] / (n_out[:, None] - 1.0 + eps))

            denom = torch.sqrt(var_in / (n_in[:, None] + eps) + var_out / (n_out[:, None] + eps))
            z = (mu_in - mu_out) / (denom + eps)
            p_torch = torch.special.ndtr(z)  # [C, G]

            Im = torch.sparse.mm(self.It_csr, masks)  # [G, C]
            r_in = (Im / (n_in.unsqueeze(0) + eps)).T  # [C, G]

            pr = p_torch * r_in  # [C, G]
            pr_sum = pr.sum(0, keepdim=True)  # [1, G]
            valid = pr_sum > 0  # [1, G] bool

            denom_safe = torch.where(valid, pr_sum, torch.ones_like(pr_sum))  # [1, G]
            s = torch.where(valid, pr / denom_safe, torch.zeros_like(pr))  # [C, G]

            s = s.T  # [G, C]
            if return_all:
                return s, p_torch, r_in
            return s

        if masks.dim() != 3:
            raise ValueError(f"masks must be 2D or 3D, got {tuple(masks.shape)}")

        masks = masks.to(torch.float32)
        M_cells, C, S = masks.shape

        n_in_raw = masks.sum(0)  # [C, S]
        n_in = n_in_raw + eps
        n_out = M_cells - n_in_raw + eps

        w_in = masks / n_in.unsqueeze(0)  # [M, C, S]
        w_in_flat = w_in.reshape(M_cells, C * S)

        mu_in = torch.sparse.mm(self.Et_csr, w_in_flat).T.reshape(C, S, -1)  # [C,S,G]
        ex2_in = torch.sparse.mm(self.Et2_csr, w_in_flat).T.reshape(C, S, -1)

        n_in_e = n_in[:, :, None]
        n_out_e = n_out[:, :, None]
        E_s = self.E_sum.squeeze(1)[None, None, :]  # [1,1,G]
        E2_s = self.E2_sum.squeeze(1)[None, None, :]

        mu_out = (E_s - n_in_e * mu_in) / n_out_e
        ex2_out = (E2_s - n_in_e * ex2_in) / n_out_e

        var_in = (ex2_in - mu_in**2).clamp_min(0.0) * (n_in_e / (n_in_e - 1.0 + eps))
        var_out = (ex2_out - mu_out**2).clamp_min(0.0) * (n_out_e / (n_out_e - 1.0 + eps))

        denom = torch.sqrt(var_in / (n_in_e + eps) + var_out / (n_out_e + eps))
        z = (mu_in - mu_out) / (denom + eps)
        p_torch = torch.special.ndtr(z)  # [C,S,G]

        Im = torch.sparse.mm(self.It_csr, masks.reshape(M_cells, C * S)).reshape(-1, C, S)  # [G,C,S]
        r_in = Im.permute(1, 2, 0) / (n_in_e + eps)  # [C,S,G]

        pr = p_torch * r_in  # [C,S,G]
        pr_sum = pr.sum(0, keepdim=True)  # [1,S,G]
        valid = pr_sum > 0

        denom_safe = torch.where(valid, pr_sum, torch.ones_like(pr_sum))
        s = torch.where(valid, pr / denom_safe, torch.zeros_like(pr))

        s = s.permute(2, 0, 1).contiguous()  # [G,C,S]
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

        self.beta_F = self.beta_F0[:, None] + torch.einsum("ks,sp->ksp", M, M.T * self.FY) / n[None, :, None]
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
            pval_two_sided = torch.tensor(st.chi2(1).sf(self.lrt.cpu().data.numpy()), device=self.F.device)
            pval_one_sided = torch.where(self.beta_g > 0, pval_two_sided / 2.0, 1.0 - (pval_two_sided / 2.0))
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
    magma_file: str | Path,
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

    scores_df = pd.DataFrame(scores.cpu().numpy(), index=adata.var_names, columns=cell_types)

    magma_df = pd.read_csv(magma_file, sep=r"\s+")
    if magma_gene_col not in magma_df.columns or magma_z_col not in magma_df.columns:
        raise ValueError(
            f"MAGMA file must have columns '{magma_gene_col}' and '{magma_z_col}'; " f"found {list(magma_df.columns)}"
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

    associations_df = (
        pd.DataFrame(
            {
                "cell_type": cell_types,
                "pvalue": pvals,
                "beta": betas,
                "se": stes,
                "FDR": fdr,
            }
        )
        .sort_values("pvalue")
        .reset_index(drop=True)
    )

    if save_results and prefix:
        out_file = f"{prefix}_associations.tsv"
        associations_df.to_csv(out_file, sep="\t", index=False)
        logger.info(f"run_seismic_torch: saved {out_file}")

    return associations_df
