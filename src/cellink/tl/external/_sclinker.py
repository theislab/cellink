from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from collections import Counter
from scipy import sparse

logger = logging.getLogger(__name__)


ENHANCER_TISSUES = Literal[
    "BLD", "BRN", "GI", "LNG", "LIV", "KID", "SKIN", "FAT", "HRT", "ALL"
]

ENHANCER_TISSUE_MAP = {
    "BLD": "Blood",
    "BRN": "Brain",
    "GI": "Colon/Intestine",
    "LNG": "Lung",
    "LIV": "Liver",
    "KID": "Kidney",
    "SKIN": "Skin",
    "FAT": "Adipose",
    "HRT": "Heart",
    "ALL": "All tissues (non-specific)",
}



def compute_celltype_programs(
    adata: AnnData,
    celltype_col: str,
    *,
    min_cells_per_type: int = 10,
    use_raw: bool = False,
    method: str = "wilcoxon",
    prefix: str = "celltype",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute cell-type gene programs by differential expression (one vs rest).

    Each gene receives a probabilistic score in [0, 1] derived from the
    Wilcoxon rank-sum test p-value, following the sc-linker paper:

        X = -2 * log(p_adj)   [chi2_2 distributed]
        score = (X - min(X)) / (max(X) - min(X))

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in ``adata.X``.
    celltype_col
        Column in ``adata.obs`` with cell-type labels.
    min_cells_per_type
        Cell types with fewer cells are skipped.
    use_raw
        Passed to ``sc.tl.rank_genes_groups``.
    method
        DE method; ``"wilcoxon"`` is recommended.
    prefix
        Prefix for the DE key stored in ``adata.uns``.
    out_dir
        If provided and ``save=True``, saves CSV files here.
    save
        Whether to write output CSV files.

    Returns
    -------
    dict
        ``{"pval": DataFrame, "logfold": DataFrame, "score": DataFrame,
           "genescores": DataFrame}``
        All DataFrames are (genes x cell_types).
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"'{celltype_col}' not found in adata.obs")

    de_key = f"{celltype_col}_DE"

    counts = Counter(adata.obs[celltype_col])
    adata.obs[f"{celltype_col}_counts"] = [counts[ct] for ct in adata.obs[celltype_col]]
    adata_filtered = adata[adata.obs[f"{celltype_col}_counts"] > min_cells_per_type].copy()

    logger.info(
        f"Running rank_genes_groups for {celltype_col} "
        f"({adata_filtered.n_obs} cells after filtering)"
    )
    sc.tl.rank_genes_groups(
        adata_filtered,
        celltype_col,
        key_added=de_key,
        use_raw=use_raw,
        method=method,
        n_genes=adata_filtered.n_vars,
    )
    adata.uns[de_key] = adata_filtered.uns[de_key]

    results = _extract_de_matrices(adata, de_key, label_col=celltype_col)
    genescores = _compute_genescores(results["score"])

    if (genescores.index.str.startswith("ENSG").mean() > 0.5
            and "gene_name" in adata.var.columns):
        gene_name_map = adata.var["gene_name"].dropna().to_dict()
        for key in list(results.keys()):
            results[key].index = results[key].index.map(
                lambda g: gene_name_map.get(g, g)
            )
        genescores = results["score"].copy()
        genescores = _compute_genescores(genescores)
        logger.info(
            "Mapped var_names from ENSG IDs to HGNC gene names using adata.var['gene_name']. "
            "This is required for matching against Roadmap/ABC TargetGene columns."
        )
    elif genescores.index.str.startswith("ENSG").mean() > 0.5:
        logger.warning(
            "var_names appear to be ENSG IDs but no 'gene_name' column found in adata.var. "
            "The Roadmap/ABC TargetGene column uses HGNC gene names, so annotation will "
            "produce empty results unless var_names are HGNC symbols. "
            "Add HGNC names to adata.var['gene_name'] before calling this function, "
            "or ensure adata.var_names are already HGNC symbols."
        )

    results["genescores"] = genescores

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            df.to_csv(out_dir / f"{prefix}_{key}.csv")
        logger.info(f"Saved cell-type program matrices to {out_dir}")

    return results


def compute_diseaseprogression_programs(
    adata: AnnData,
    celltype_col: str,
    diagnosis_col: str,
    healthy_label: str,
    disease_label: str,
    *,
    min_cells_per_group: int = 5,
    use_raw: bool = False,
    method: str = "wilcoxon",
    prefix: str = "disease",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute disease-progression gene programs.

    For each cell type present in both healthy and disease tissue, performs
    DE (disease cells of that type vs healthy cells of that type).
    Contamination genes (outlier low-score genes from global disease DE) are
    zeroed out before normalisation, following the sc-linker approach.

    Parameters
    ----------
    adata
        AnnData with both healthy and disease cells.
    celltype_col
        Column in ``adata.obs`` with cell-type labels.
    diagnosis_col
        Column in ``adata.obs`` with diagnosis / disease status.
    healthy_label
        Value in ``diagnosis_col`` that denotes healthy samples.
    disease_label
        Value in ``diagnosis_col`` that denotes disease samples.
    min_cells_per_group
        Minimum cells in each (healthy/disease x cell type) group.
    use_raw
        Passed to ``sc.tl.rank_genes_groups``.
    method
        DE method.
    prefix
        Prefix for output file names.
    out_dir
        Directory to write CSV output files.
    save
        Whether to write output CSV files.

    Returns
    -------
    dict
        Same structure as :func:`compute_celltype_programs`.
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"'{celltype_col}' not found in adata.obs")
    if diagnosis_col not in adata.obs.columns:
        raise ValueError(f"'{diagnosis_col}' not found in adata.obs")

    disease_label_mapping = {healthy_label: "Healthy", disease_label: "Disease"}

    disease_subset = adata[adata.obs[diagnosis_col] == disease_label].copy()
    sc.tl.rank_genes_groups(
        disease_subset,
        groupby=celltype_col,
        reference="rest",
        n_genes=disease_subset.n_vars,
        method=method,
        use_raw=use_raw,
    )
    contamination = _compute_contamination(disease_subset, celltype_col)
    adata.uns[f"contamination_{celltype_col}"] = contamination

    # Build DEstatus column: "Healthy_<celltype>" or "Disease_<celltype>"
    adata.obs["_DEstatus"] = [
        disease_label_mapping.get(diag, "Unknown") + "_" + ct
        for diag, ct in zip(adata.obs[diagnosis_col], adata.obs[celltype_col])
    ]
    destatus_counts = Counter(adata.obs["_DEstatus"])

    cell_types = sorted(set(adata.obs[celltype_col]))
    processed_cell_types = []

    for ct in cell_types:
        healthy_key = f"Healthy_{ct}"
        disease_key = f"Disease_{ct}"
        n_h = destatus_counts.get(healthy_key, 0)
        n_d = destatus_counts.get(disease_key, 0)
        if n_h < min_cells_per_group or n_d < min_cells_per_group:
            logger.debug(f"Skipping {ct}: healthy={n_h}, disease={n_d}")
            continue

        de_key = f"{ct}_DE"
        logger.info(f"Computing disease-progression DE for {ct} (H={n_h}, D={n_d})")
        sc.tl.rank_genes_groups(
            adata,
            groupby="_DEstatus",
            reference=healthy_key,
            groups=[disease_key],
            key_added=de_key,
            n_genes=adata.n_vars,
            method=method,
            use_raw=use_raw,
        )
        processed_cell_types.append(ct)

    all_de_keys = [f"{ct}_DE" for ct in processed_cell_types]
    results = _extract_de_matrices_disease(adata, all_de_keys, contamination, celltype_col)
    genescores = _compute_genescores(results["score"])
    results["genescores"] = genescores

    adata.obs.drop(columns=["_DEstatus"], inplace=True, errors="ignore")

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            df.to_csv(out_dir / f"{prefix}_{key}.csv")
        logger.info(f"Saved disease-progression program matrices to {out_dir}")

    return results


def compute_nmf_programs(
    adata: AnnData,
    *,
    n_components: Optional[int] = None,
    n_extra: int = 10,
    celltype_col: str = "cell_type",
    layer: Optional[str] = "counts",
    normalize: bool = True,
    random_state: int = 0,
    device: str = "cuda",
    prefix: str = "nmf",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute NMF cellular process programs (single healthy tissue).

    Parameters
    ----------
    adata
        AnnData. Raw counts preferred (from ``layer`` or ``adata.X``).
    n_components
        Number of NMF factors. Defaults to ``n_cell_types + n_extra``.
    n_extra
        Added to the number of annotated cell types when auto-setting components.
    celltype_col
        Column in ``adata.obs`` for auto-setting ``n_components``.
    layer
        Layer to use. If None, uses ``adata.X``.
    normalize
        Divide each matrix by its global maximum (as in original sc-linker).
    random_state
        NMF random seed.
    device : str, default ``"cuda"``
        Device for torchnmf backend: ``"cuda"`` or ``"cpu"``.

        - ``"cuda"`` — uses GPU if available, raises a clear warning if CUDA is
          not found and falls back to CPU.
        - ``"cpu"``  — forces CPU even if a GPU is present.

        If ``torchnmf`` is not installed at all, cellink logs an install hint
        and falls back to sklearn NMF (which is slower but always available).
    prefix
        File name prefix for output CSVs.
    out_dir
        Directory for output CSVs.
    save
        Whether to save CSVs.

    Returns
    -------
    W : DataFrame
        Cell x factor (cell programs), index = obs_names.
    H : DataFrame
        Gene x factor (gene programs), index = var_names.
    corr : DataFrame
        Gene x factor (Pearson correlation between gene expression and W scores).

    Notes
    -----
    Backend priority:

    1. **torchnmf** (GPU or CPU), install with ``pip install torchnmf``.
       On large matrices (>50k cells) this is 5-20x faster than sklearn.
    2. **sklearn NMF** with ``init='nndsvda'`` + ``solver='cd'``. Significantly faster than ``init='random'`` but still slow
       on very large matrices.
    """
    from sklearn.decomposition import NMF

    if n_components is None:
        n_ct = len(set(adata.obs.get(celltype_col, []))) if celltype_col in adata.obs else 10
        n_components = n_ct + n_extra
        logger.info(f"Setting n_components = {n_ct} cell types + {n_extra} = {n_components}")

    X = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)
    if normalize:
        X = X / (np.max(X) + 1e-12)

    logger.info(f"Fitting NMF with {n_components} components on {X.shape} matrix")

    W_arr = None
    H_arr = None
    try:
        import torch
        from torchnmf.nmf import NMF as TorchNMF
    except ImportError:
        logger.warning(
            "torchnmf is not installed — falling back to sklearn NMF, which is "
            "significantly slower on large matrices (>50k cells).\n"
            "Install the faster backend with:\n"
            "  pip install torchnmf"
        )
        torch = None  

    if torch is not None:
        if device == "cuda":
            if torch.cuda.is_available():
                _device = "cuda"
                logger.info(
                    f"torchnmf: using GPU ({torch.cuda.get_device_name(0)})"
                )
            else:
                logger.warning(
                    "device='cuda' requested but no CUDA GPU found. "
                    "Falling back to CPU for torchnmf. "
                    "Pass device='cpu' explicitly to suppress this warning."
                )
                _device = "cpu"
        else:
            _device = "cpu"
            logger.info("torchnmf: using CPU (device='cpu' requested)")

        X_t = torch.tensor(X, dtype=torch.float32, device=_device)
        model_t = TorchNMF(X_t.shape, rank=n_components).to(_device)
        model_t.fit(X_t, beta=2, max_iter=200, tol=1e-4)
        W_arr = model_t.H.T.detach().cpu().numpy()   # (n_cells,    n_components)
        H_arr = model_t.W.detach().cpu().numpy()     # (n_features, n_components)
        del X_t, model_t

    if W_arr is None:
        logger.info(
            "Using sklearn NMF with init='nndsvda' + solver='cd'. "
            "This is slower than torchnmf on large matrices. "
            "Install torchnmf for GPU-accelerated NMF: pip install torchnmf"
        )
        model = NMF(
            n_components=n_components,
            init="nndsvda",   # truncated-SVD warm start — ~5-10x faster than "random"
            solver="cd",      # coordinate descent — faster than multiplicative update
            max_iter=500,
            tol=1e-4,
            random_state=random_state,
        )
        W_arr = model.fit_transform(X)
        H_arr = model.components_.T  # (n_features, n_components)

    W = pd.DataFrame(W_arr, index=adata.obs_names, columns=[f"NMF_{i}" for i in range(n_components)])
    H = pd.DataFrame(H_arr, index=adata.var_names, columns=[f"NMF_{i}" for i in range(n_components)])
    corr = _compute_nmf_gene_correlations(X, W_arr, adata.var_names, W.columns)

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        W.to_csv(out_dir / f"{prefix}_cellprograms.csv")
        H.to_csv(out_dir / f"{prefix}_geneprograms.csv")
        corr.to_csv(out_dir / f"{prefix}_correlation_cellprograms.csv")

    return W, H, corr


def compute_joint_nmf_programs(
    adata_healthy: AnnData,
    adata_disease: AnnData,
    *,
    n_shared: int = 10,
    n_healthy_specific: int = 5,
    n_disease_specific: int = 5,
    gamma: float = 1.0,
    layer: Optional[str] = None,
    random_state: int = 0,
    prefix: str = "joint_nmf",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute joint NMF programs across healthy and disease tissue.

    Decomposes healthy (H) and disease (D) matrices jointly:

        H ≈ [L_shared_H | L_unique_H] x F_H
        D ≈ [L_shared_D | L_unique_D] x F_D

    with a coupling term y/2 ||L_shared_H - L_shared_D||² that forces
    the shared programs to be similar.

    Parameters
    ----------
    adata_healthy
        AnnData for healthy tissue.
    adata_disease
        AnnData for disease tissue. Must share var_names with ``adata_healthy``.
    n_shared
        Number of shared programs between healthy and disease.
    n_healthy_specific
        Number of healthy-specific programs.
    n_disease_specific
        Number of disease-specific programs.
    gamma
        Coupling strength (higher → shared programs more similar).
    layer
        Layer to use as expression matrix. If None, uses ``adata.X``.
    random_state
        Random seed for NMF initialisation.
    prefix
        File name prefix for output CSVs.
    out_dir
        Directory for output CSVs.
    save
        Whether to save CSVs.

    Returns
    -------
    dict with keys:
        ``"Wh"``  : healthy cell x factor loadings (shared + healthy-specific)
        ``"Wd"``  : disease cell x factor loadings (shared + disease-specific)
        ``"Hh"``  : gene x factor weights (healthy)
        ``"Hd"``  : gene x factor weights (disease)
        ``"shared_Wh"``  : healthy cell x shared-factor loadings
        ``"shared_Wd"``  : disease cell x shared-factor loadings
        ``"unique_Hh"`` : gene x healthy-specific-factor weights
        ``"unique_Hd"`` : gene x disease-specific-factor weights
    """
    from ._joint_nmf import JointNMFWrapper

    common_genes = adata_healthy.var_names.intersection(adata_disease.var_names)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between healthy and disease AnnData objects.")

    Xh = _get_dense(adata_healthy[:, common_genes], layer)
    Xd = _get_dense(adata_disease[:, common_genes], layer)

    logger.info(
        f"Joint NMF: healthy={Xh.shape}, disease={Xd.shape}, "
        f"shared={n_shared}, H-specific={n_healthy_specific}, D-specific={n_disease_specific}"
    )

    jnmf = JointNMFWrapper(
        Xh=Xh,
        Xd=Xd,
        n_shared=n_shared,
        n_healthy_specific=n_healthy_specific,
        n_disease_specific=n_disease_specific,
        gamma=gamma,
        random_state=random_state,
    )
    jnmf.fit()

    n_total_h = n_shared + n_healthy_specific
    n_total_d = n_shared + n_disease_specific

    h_cols = [f"Shared_{i}" for i in range(n_shared)] + [f"Healthy_{i}" for i in range(n_healthy_specific)]
    d_cols = [f"Shared_{i}" for i in range(n_shared)] + [f"Disease_{i}" for i in range(n_disease_specific)]

    Wh_df = pd.DataFrame(jnmf.Wh, index=adata_healthy.obs_names, columns=h_cols)
    Wd_df = pd.DataFrame(jnmf.Wd, index=adata_disease.obs_names, columns=d_cols)
    Hh_df = pd.DataFrame(jnmf.Hh.T, index=common_genes, columns=h_cols)
    Hd_df = pd.DataFrame(jnmf.Hd.T, index=common_genes, columns=d_cols)

    results = {
        "Wh": Wh_df,
        "Wd": Wd_df,
        "Hh": Hh_df,
        "Hd": Hd_df,
        "shared_Wh": Wh_df.iloc[:, :n_shared],
        "shared_Wd": Wd_df.iloc[:, :n_shared],
        "unique_Hh": Hh_df.iloc[:, n_shared:],
        "unique_Hd": Hd_df.iloc[:, n_shared:],
    }

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            df.to_csv(out_dir / f"{prefix}_{key}.csv")

    return results


def _get_dense(adata: AnnData, layer: Optional[str]) -> np.ndarray:
    """Return a dense float32 expression matrix from an AnnData layer or X."""
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if sparse.issparse(X):
        return X.toarray().astype(np.float64)
    return np.array(X, dtype=np.float64)


def _compute_contamination(adata: AnnData, celltype_col: str) -> Dict[str, list]:
    """Identify outlier genes from global disease DE (sc-linker paper method)."""
    contamination: Dict[str, list] = {}
    for ct in set(adata.obs[celltype_col]):
        try:
            scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])[ct]
            names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])[ct]
            threshold = np.mean(scores) - 6 * np.std(scores)
            contamination[ct] = names[scores < threshold].tolist()
        except (KeyError, TypeError):
            contamination[ct] = []
    return contamination


def _extract_de_matrices(
    adata: AnnData, de_key: str, label_col: str
) -> Dict[str, pd.DataFrame]:
    """Extract pval / logfold / score matrices from adata.uns DE results."""
    genes = list(set(adata.var_names))
    gene2idx = {g: i for i, g in enumerate(genes)}
    cellsubsets = list(adata.uns[de_key]["names"].dtype.fields.keys())

    pval_mtx = np.zeros((len(genes), len(cellsubsets)))
    logfold_mtx = np.zeros_like(pval_mtx)
    score_mtx = np.zeros_like(pval_mtx)

    for gene_row, pval_row, lf_row, sc_row in zip(
        adata.uns[de_key]["names"],
        adata.uns[de_key]["pvals_adj"],
        adata.uns[de_key]["logfoldchanges"],
        adata.uns[de_key]["scores"],
    ):
        for j, cs in enumerate(cellsubsets):
            g = gene_row[cs]
            if g in gene2idx:
                idx = gene2idx[g]
                pval_mtx[idx, j] = pval_row[cs]
                logfold_mtx[idx, j] = lf_row[cs]
                score_mtx[idx, j] = sc_row[cs]

    level = label_col.split("_")[-1] if "_" in label_col else "2"
    col_names = [f"{cs}_L{level}" for cs in cellsubsets]

    return {
        "pval": pd.DataFrame(pval_mtx, index=genes, columns=col_names),
        "logfold": pd.DataFrame(logfold_mtx, index=genes, columns=col_names),
        "score": pd.DataFrame(score_mtx, index=genes, columns=col_names),
    }


def _extract_de_matrices_disease(
    adata: AnnData,
    de_keys: List[str],
    contamination: Dict[str, list],
    celltype_col: str,
) -> Dict[str, pd.DataFrame]:
    """Extract DE matrices from disease-progression analysis."""
    genes = list(set(adata.var_names))
    gene2idx = {g: i for i, g in enumerate(genes)}

    all_pval, all_lf, all_sc = [], [], []
    col_names = []

    for de_key in de_keys:
        ct = de_key.replace("_DE", "")
        ct_contamination = set(contamination.get(ct, []))
        cellsubsets = list(adata.uns[de_key]["names"].dtype.fields.keys())

        pval_mtx = np.zeros((len(genes), len(cellsubsets)))
        logfold_mtx = np.zeros_like(pval_mtx)
        score_mtx = np.zeros_like(pval_mtx)

        for gene_row, pval_row, lf_row, sc_row in zip(
            adata.uns[de_key]["names"],
            adata.uns[de_key]["pvals_adj"],
            adata.uns[de_key]["logfoldchanges"],
            adata.uns[de_key]["scores"],
        ):
            for j, cs in enumerate(cellsubsets):
                g = gene_row[cs]
                if g not in gene2idx:
                    continue
                idx = gene2idx[g]
                if g in ct_contamination:
                    pval_mtx[idx, j] = 1.0
                else:
                    pval_mtx[idx, j] = pval_row[cs]
                    logfold_mtx[idx, j] = lf_row[cs]
                    score_mtx[idx, j] = sc_row[cs]

        all_pval.append(pd.DataFrame(pval_mtx, index=genes, columns=cellsubsets))
        all_lf.append(pd.DataFrame(logfold_mtx, index=genes, columns=cellsubsets))
        all_sc.append(pd.DataFrame(score_mtx, index=genes, columns=cellsubsets))

    if not all_pval:
        return {"pval": pd.DataFrame(), "logfold": pd.DataFrame(), "score": pd.DataFrame()}

    return {
        "pval": pd.concat(all_pval, axis=1),
        "logfold": pd.concat(all_lf, axis=1),
        "score": pd.concat(all_sc, axis=1),
    }


def _compute_genescores(score_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    Transform score matrix to probabilistic gene scores in [0, 1].

    Follows sc-linker: clip negative scores to 0, apply survival function of
    normal distribution (converts z-score to one-sided p-value), add epsilon,
    compute X = -2*log(p), then min-max normalize to [0, 1].
    """
    if score_mtx.empty:
        return score_mtx

    clipped = score_mtx.clip(lower=0)
    pvals = pd.DataFrame(
        scipy.stats.norm.sf(clipped.values),
        index=clipped.index,
        columns=clipped.columns,
    )
    pvals = pvals + 1e-8
    X = -2 * np.log(pvals)

    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom = col_max - col_min
    denom = denom.replace(0, 1.0)

    genescores = (X - col_min) / denom
    return genescores


def _compute_nmf_gene_correlations(
    X: np.ndarray, W: np.ndarray, var_names, nmf_cols
) -> pd.DataFrame:
    """Pearson correlation between gene expression and NMF cell program weights."""
    nrow, ncol = X.shape
    correlations = []
    Xsp = sparse.csc_matrix(X)

    for i in range(W.shape[1]):
        y = W[:, i]
        yy = y - y.mean()
        ys = yy / (np.sqrt(np.dot(yy, yy)) + 1e-12)

        xm = np.asarray(Xsp.mean(axis=0)).ravel()
        xs_sq = np.add.reduceat(Xsp.data ** 2, Xsp.indptr[:-1]) - nrow * xm * xm
        xs_sq = np.maximum(xs_sq, 1e-12)
        xs = np.sqrt(xs_sq)

        correl = np.add.reduceat(Xsp.data * ys[Xsp.indices], Xsp.indptr[:-1]) / xs
        correlations.append(correl)

    return pd.DataFrame(np.vstack(correlations).T, columns=nmf_cols, index=var_names)