from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from scipy import sparse

logger = logging.getLogger(__name__)


ENHANCER_TISSUES = Literal["BLD", "BRN", "GI", "LNG", "LIV", "KID", "SKIN", "FAT", "HRT", "ALL"]

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
) -> dict[str, pd.DataFrame]:
    """
    Compute cell-type gene programs by differential expression (one vs rest).

    Each gene receives a score in [0, 1] from the Wilcoxon rank-sum test, as in
    sc-linker: the z-score of the test, clipped at 0, gives a one-sided p-value p, and

        X = -2 * log(p + 1e-8)
        score = (X - min(X)) / (max(X) - min(X))

    within each cell type.

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

    logger.info(f"Running rank_genes_groups for {celltype_col} " f"({adata_filtered.n_obs} cells after filtering)")
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
    results, _ = _maybe_remap_ensg_to_hgnc(results, adata, context="compute_celltype_programs")
    genescores = _compute_genescores(results["score"])
    results["genescores"] = genescores

    if save and out_dir is not None:
        out_dir = Path(out_dir).expanduser()
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
) -> dict[str, pd.DataFrame]:
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

    adata.obs["_DEstatus"] = [
        disease_label_mapping.get(diag, "Unknown") + "_" + ct
        for diag, ct in zip(adata.obs[diagnosis_col], adata.obs[celltype_col], strict=False)
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

    if "_DEstatus" in adata.obs.columns:
        del adata.obs["_DEstatus"]

    if save and out_dir is not None:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            df.to_csv(out_dir / f"{prefix}_{key}.csv")
        logger.info(f"Saved disease-progression program matrices to {out_dir}")

    return results


def compute_nmf_programs(
    adata: AnnData,
    *,
    n_components: int | None = None,
    n_extra: int = 10,
    celltype_col: str = "cell_type",
    layer: str | None = None,
    normalize: bool = True,
    use_highly_variable: bool | None = None,
    min_cells: int = 0,
    collapse_threshold: float | None = 0.8,
    random_state: int = 0,
    device: str | None = None,
    prefix: str = "nmf",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute NMF cellular process programs as in sc-linker (Jagadeesh et al. 2022).

    NMF factorizes the cells x genes matrix of log-normalized expression into latent
    factors, each with a contribution to every cell (``W``) and a weight for every gene
    (``H``). The cellular process program of a factor is the Pearson correlation, across
    cells, between the expression of each gene and the contribution of the factor. Factors
    whose contributions correlate above ``collapse_threshold`` are collapsed into one.

    Parameters
    ----------
    adata
        AnnData with log-normalized expression in ``adata.X`` or ``layer``.
    n_components
        Number of NMF factors. Defaults to ``n_cell_types + n_extra``.
    n_extra
        Added to the number of annotated cell types when auto-setting components.
    celltype_col
        Column in ``adata.obs`` for auto-setting ``n_components``.
    layer
        Layer with the log-normalized expression. If None, uses ``adata.X``.
    normalize
        Divide the matrix the factors are fit on by its global maximum.
    use_highly_variable
        Fit the factors on the genes marked in ``adata.var["highly_variable"]``
        (``sc.pp.highly_variable_genes``). None uses them when the column exists. The
        correlations cover every expressed gene either way.
    min_cells
        Fit the factors only on genes detected in at least ``min_cells`` cells. The
        default 0 keeps every gene.
    collapse_threshold
        A factor whose contributions to the cells correlate above this value with those of
        an earlier factor is dropped, so that each group of correlated factors is kept as a
        single factor. None keeps every factor.
    random_state
        NMF random seed.
    device : str or None, default None
        None fits with scikit-learn's NMF (``init="random"``, coordinate descent), as
        sc-linker does. ``"cuda"`` or ``"cpu"`` fits with torchnmf on that device (on the
        CPU when no GPU is found); without torchnmf, scikit-learn is used.
    prefix
        File name prefix for output CSVs.
    out_dir
        Directory for output CSVs.
    save
        Whether to save CSVs: ``{prefix}_cellprograms.csv`` (``W``),
        ``{prefix}_geneprograms.csv`` (``H``), ``{prefix}_correlation_cellprograms.csv``
        (``corr``) and ``{prefix}_genescores.csv``, the gene scores in [0, 1]: ``corr``
        with negative correlations set to 0, divided by the maximum of each program.

    Returns
    -------
    W : DataFrame
        Cell x factor contributions, index = obs_names.
    H : DataFrame
        Gene x factor weights for the genes the factors were fit on.
    corr : DataFrame
        Gene x factor Pearson correlation between gene expression and ``W``, for every gene
        expressed in at least one cell.
    """
    from sklearn.decomposition import NMF

    if n_components is None:
        n_ct = len(set(adata.obs.get(celltype_col, []))) if celltype_col in adata.obs else 10
        n_components = n_ct + n_extra
        logger.info(f"Setting n_components = {n_ct} cell types + {n_extra} = {n_components}")

    X_all = adata.layers[layer] if layer and layer in adata.layers else adata.X
    fit_genes = np.ones(adata.n_vars, dtype=bool)
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var.columns
    if use_highly_variable:
        if "highly_variable" not in adata.var.columns:
            raise ValueError(
                "use_highly_variable=True needs adata.var['highly_variable'] (sc.pp.highly_variable_genes)."
            )
        fit_genes &= adata.var["highly_variable"].to_numpy(dtype=bool)
    if min_cells:
        fit_genes &= np.asarray((X_all > 0).sum(axis=0)).ravel() >= min_cells
    if not fit_genes.any():
        raise ValueError("No gene is left to fit the NMF factors on.")
    X_fit = X_all if fit_genes.all() else X_all[:, fit_genes]
    logger.info(f"Fitting {n_components} NMF factors on {X_fit.shape[0]} cells x {X_fit.shape[1]} genes")

    torch = None
    if device is not None:
        try:
            import torch
            from torchnmf.nmf import NMF as TorchNMF
        except ImportError:
            logger.warning("torchnmf is not installed (pip install torchnmf); fitting with scikit-learn's NMF")
            torch = None

    if torch is not None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("device='cuda' requested but no CUDA GPU found; fitting with torchnmf on the CPU")
            device = "cpu"
        X = X_fit.astype(np.float32).toarray() if sparse.issparse(X_fit) else np.array(X_fit, dtype=np.float32)
        if normalize:
            X /= np.max(X) + 1e-12
        X_t = torch.from_numpy(X).to(device)
        model_t = TorchNMF(X_t.shape, rank=n_components).to(device)
        model_t.fit(X_t, beta=2, max_iter=200, tol=1e-4)
        W_arr = model_t.H.detach().cpu().numpy()
        H_arr = model_t.W.detach().cpu().numpy()
        del X, X_t, model_t
    else:
        X = X_fit.astype(np.float64) if sparse.issparse(X_fit) else np.asarray(X_fit, dtype=np.float64)
        if normalize:
            X = X / (X.max() + 1e-12)
        model = NMF(n_components=n_components, init="random", random_state=random_state)
        W_arr = model.fit_transform(X)
        H_arr = model.components_.T
        del X

    kept = _collapse_correlated_factors(W_arr, collapse_threshold)
    if len(kept) < n_components:
        logger.info(
            f"Collapsed {n_components - len(kept)} of {n_components} factors correlated above {collapse_threshold}"
        )
    columns = [f"NMF_{i}" for i in kept]
    W_arr, H_arr = W_arr[:, kept], H_arr[:, kept]
    W = pd.DataFrame(W_arr, index=adata.obs_names, columns=columns)
    H = pd.DataFrame(H_arr, index=adata.var_names[fit_genes], columns=columns)

    X_csc = sparse.csc_matrix(X_all, copy=True)
    X_csc.eliminate_zeros()
    expressed = np.diff(X_csc.indptr) > 0
    corr = _compute_nmf_gene_correlations(X_csc[:, expressed], W_arr, adata.var_names[expressed], columns)
    del X_csc
    genescores = corr.clip(lower=0)
    genescores = genescores / genescores.max().replace(0, 1.0)

    gene_frames, _ = _maybe_remap_ensg_to_hgnc(
        {"H": H, "corr": corr, "genescores": genescores}, adata, context="compute_nmf_programs"
    )
    H, corr, genescores = gene_frames["H"], gene_frames["corr"], gene_frames["genescores"]

    if save and out_dir is not None:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        W.to_csv(out_dir / f"{prefix}_cellprograms.csv")
        H.to_csv(out_dir / f"{prefix}_geneprograms.csv")
        corr.to_csv(out_dir / f"{prefix}_correlation_cellprograms.csv")
        genescores.to_csv(out_dir / f"{prefix}_genescores.csv")

    return W, H, corr


def compute_joint_nmf_programs(
    adata_healthy: AnnData,
    adata_disease: AnnData,
    *,
    n_shared: int = 10,
    n_healthy_specific: int = 5,
    n_disease_specific: int = 5,
    gamma: float = 1.0,
    layer: str | None = None,
    random_state: int = 0,
    prefix: str = "joint_nmf",
    out_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compute joint NMF programs across healthy and disease tissue.

    Decomposes the healthy (H) and disease (D) genes x cells matrices jointly:

        H ≈ [L_shared_H | L_unique_H] x F_H
        D ≈ [L_shared_D | L_unique_D] x F_D

    where L are gene loadings (genes x programs) and F are cell weights (programs x
    cells), with a coupling term y/2 ||L_shared_H - L_shared_D||² that forces the
    gene loadings of the shared programs to be similar. Only genes are paired, so
    healthy and disease may have different numbers of cells.

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
        Xh=Xh.T,
        Xd=Xd.T,
        n_shared=n_shared,
        n_healthy_specific=n_healthy_specific,
        n_disease_specific=n_disease_specific,
        gamma=gamma,
        random_state=random_state,
    )
    jnmf.fit()

    h_cols = [f"Shared_{i}" for i in range(n_shared)] + [f"Healthy_{i}" for i in range(n_healthy_specific)]
    d_cols = [f"Shared_{i}" for i in range(n_shared)] + [f"Disease_{i}" for i in range(n_disease_specific)]

    Wh_df = pd.DataFrame(jnmf.Hh.T, index=adata_healthy.obs_names, columns=h_cols)
    Wd_df = pd.DataFrame(jnmf.Hd.T, index=adata_disease.obs_names, columns=d_cols)
    Hh_df = pd.DataFrame(jnmf.Wh, index=common_genes, columns=h_cols)
    Hd_df = pd.DataFrame(jnmf.Wd, index=common_genes, columns=d_cols)

    gene_frames, _ = _maybe_remap_ensg_to_hgnc(
        {"Hh": Hh_df, "Hd": Hd_df}, adata_healthy, context="compute_joint_nmf_programs"
    )
    Hh_df, Hd_df = gene_frames["Hh"], gene_frames["Hd"]

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
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            df.to_csv(out_dir / f"{prefix}_{key}.csv")

    return results


def _maybe_remap_ensg_to_hgnc(
    frames: dict[str, pd.DataFrame], adata: AnnData, *, context: str = ""
) -> tuple[dict[str, pd.DataFrame], bool]:
    """
    Remap a set of gene-indexed DataFrames from ENSG IDs to HGNC symbols.

    All ``frames`` must share the same (gene) index. If that index is
    majority ENSG IDs, remaps every frame's index via ``adata.var['gene_name']``
    and returns the remapped frames. This is required for matching gene
    programs against the Roadmap/ABC ``TargetGene`` columns, which use HGNC
    symbols, not Ensembl IDs. Without it, ``genescores_to_abc_road_bedgraph``
    and ``genescores_to_annotations`` produce all-zero annotations.

    If the index is majority ENSG but no ``gene_name`` column exists, warns
    and returns ``frames`` unchanged (nothing to remap with).

    Returns
    -------
    (frames, remapped) : the (possibly remapped) frames, and whether a remap
        was actually performed.
    """
    any_index = next(iter(frames.values())).index
    if any_index.str.startswith("ENSG").mean() <= 0.5:
        return frames, False

    where = f" ({context})" if context else ""
    if "gene_name" not in adata.var.columns:
        logger.warning(
            f"var_names appear to be ENSG IDs but no 'gene_name' column found in adata.var{where}. "
            "The Roadmap/ABC TargetGene column uses HGNC gene names, so annotation will "
            "produce empty results unless var_names are HGNC symbols. "
            "Add HGNC names to adata.var['gene_name'] before calling this function, "
            "or ensure adata.var_names are already HGNC symbols."
        )
        return frames, False

    gene_name_map = adata.var["gene_name"].dropna().to_dict()
    remapped = {key: df.set_axis(df.index.map(lambda g: gene_name_map.get(g, g)), axis=0) for key, df in frames.items()}
    logger.info(
        f"Mapped var_names from ENSG IDs to HGNC gene names using adata.var['gene_name']{where}. "
        "This is required for matching against Roadmap/ABC TargetGene columns."
    )
    return remapped, True


def _get_dense(adata: AnnData, layer: str | None) -> np.ndarray:
    """Return a dense float32 expression matrix from an AnnData layer or X."""
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if sparse.issparse(X):
        return X.toarray().astype(np.float64)
    return np.array(X, dtype=np.float64)


def _compute_contamination(adata: AnnData, celltype_col: str) -> dict[str, list]:
    """Identify outlier genes from global disease DE (sc-linker paper method)."""
    contamination: dict[str, list] = {}
    for ct in set(adata.obs[celltype_col]):
        try:
            scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])[ct]
            names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])[ct]
            threshold = np.mean(scores) - 6 * np.std(scores)
            contamination[ct] = names[scores < threshold].tolist()
        except (KeyError, TypeError):
            contamination[ct] = []
    return contamination


def _extract_de_matrices(adata: AnnData, de_key: str, label_col: str) -> dict[str, pd.DataFrame]:
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
        strict=False,
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
    de_keys: list[str],
    contamination: dict[str, list],
    celltype_col: str,
) -> dict[str, pd.DataFrame]:
    """Extract DE matrices from disease-progression analysis."""
    genes = list(set(adata.var_names))
    gene2idx = {g: i for i, g in enumerate(genes)}

    all_pval, all_lf, all_sc = [], [], []

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
            strict=False,
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


def _collapse_correlated_factors(W: np.ndarray, threshold: float | None) -> list[int]:
    """Indices of the factors kept when factors correlated above ``threshold`` are collapsed.

    A factor is dropped when its column of ``W`` correlates above ``threshold`` with the
    column of a factor kept before it.
    """
    if threshold is None or W.shape[1] < 2:
        return list(range(W.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(W, rowvar=False)
    kept: list[int] = []
    for j in range(W.shape[1]):
        if not any(r[j, k] > threshold for k in kept):
            kept.append(j)
    return kept


def _compute_nmf_gene_correlations(X: np.ndarray, W: np.ndarray, var_names, nmf_cols) -> pd.DataFrame:
    """Pearson correlation between gene expression and NMF cell program weights.

    Genes that are zero in every cell have no variance and get correlation 0.
    """
    nrow = X.shape[0]
    Xsp = sparse.csc_matrix(X)
    xm = np.asarray(Xsp.sum(axis=0, dtype=np.float64)).ravel() / nrow
    xs_sq = np.asarray(Xsp.power(2).sum(axis=0, dtype=np.float64)).ravel() - nrow * xm * xm
    xs = np.sqrt(np.maximum(xs_sq, 1e-12))

    yy = np.asarray(W, dtype=np.float64)
    yy = yy - yy.mean(axis=0)
    ys = yy / (np.sqrt((yy * yy).sum(axis=0)) + 1e-12)

    correl = np.asarray(Xsp.T @ ys) / xs[:, None]
    return pd.DataFrame(correl, columns=nmf_cols, index=var_names)
