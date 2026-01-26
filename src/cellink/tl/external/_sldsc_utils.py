import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
import os
import h5py
import numexpr as ne
from tqdm import tqdm


logger = logging.getLogger(__name__)


def preprocess_for_sldsc(
    adata: AnnData,
    *,
    celltype_col: str,
    log_transform: bool = True,
    filter_protein_coding: bool = True,
    filter_expressed: bool = True,
    filter_mhc: bool = True,
    mhc_chr: str = None,
    mhc_start: int = None,
    mhc_end: int = None,
    fetch_annotation: bool = True,
    genome_build: Literal["GRCh37", "GRCh38"] = "GRCh37",
    gene_identifier_mode: str = "name",
    remove_version_suffix: bool = True,
    gene_col: str | None = "gene",
    biotype_col: str | None = None,
    chr_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    inplace: bool = True,
) -> tuple[AnnData, pd.DataFrame, pd.DataFrame] | None:
    """
    Preprocess single-cell data for S-LDSC cell-type-specific analysis.

    This function performs comprehensive preprocessing including:
    - Optional log1p transformation
    - Gene annotation fetching from Ensembl BioMart (GRCh37 or GRCh38)
    - Gene filtering (protein-coding, expressed, unique names, MHC exclusion)
    - Computation of mean expression per cell type
    - Computation of specificity scores (Duncan et al. 2025; doi:10.1038/s41593-024-01834-w)

    Parameters
    ----------
    adata
        Annotated data matrix of shape `n_obs` x `n_vars` (cells x genes).
    celltype_col
        Column name in `adata.obs` containing cell type labels.
    log_transform
        Whether to apply log1p transformation. Set to False if already log-transformed.
    filter_protein_coding
        Whether to filter for protein-coding genes only.
    filter_expressed
        Whether to filter out genes with zero expression across all cells.
    filter_mhc
        Whether to exclude genes in the MHC region (chr6:25-34Mb by default).
    mhc_chr
        Chromosome containing MHC region (default: "6").
    mhc_start
        Start position of MHC region in base pairs.
    mhc_end
        End position of MHC region in base pairs.
    fetch_annotation
        Whether to fetch gene annotations from Ensembl BioMart.
        If False, expects existing annotation columns in adata.var.
    genome_build
        Genome build version: "GRCh37" or "GRCh38". Only used if fetch_annotation=True.
    gene_identifier_mode
        Gene identifier: "name" or "ensembl". Only used if fetch_annotation=True.
       remove_version_suffix
        Whether to remove version suffixes from gene names or gene IDs (e.g., ENSG00000123456.7 → ENSG00000123456).
    gene_col
        Column name for gene symbols or IDs. If None, uses var_names if fetch_annotation is True is auto-detected.
    biotype_col
        Column name for gene biotype. Auto-detected if None.
    chr_col
        Column name for chromosome. Auto-detected if None.
    start_col
        Column name for gene start position. Auto-detected if None.
    end_col
        Column name for gene end position. Auto-detected if None.
    inplace
        Whether to update `adata` in place or return a copy.

    Returns
    -------
    AnnData, pd.DataFrame, pd.DataFrame
        - Filtered AnnData object
        - Cluster-normalized-to-1000 matrix and specificity derived from that (genes x cell types)
        - Specificity scores per cell type (genes x cell types)
        Returns None if inplace=True.

    Raises
    ------
    AssertionError
        If celltype_col is not present in adata.obs.
    ValueError
        If required annotation columns are missing and fetch_annotation=False.
    ImportError
        If pybiomart is not installed and fetch_annotation=True.

    Examples
    --------
    >>> # Using GRCh37 (default)
    >>> adata_filtered, mean_expr, specificity = preprocess_for_sldsc(adata, celltype_col="cell_type", inplace=False)
    >>> # Using GRCh38
    >>> adata_filtered, mean_expr, specificity = preprocess_for_sldsc(
    ...     adata, celltype_col="cell_type", genome_build="GRCh38", inplace=False
    ... )
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs")



    if fetch_annotation:
        anno_df = _fetch_ensembl_annotation(genome_build=genome_build, gene_identifier_mode=gene_identifier_mode)
        if gene_col is None:
            adata.var["gene"] = adata.var_names
        adata.var["gene_upper"] = adata.var[gene_col].str.upper()
        if remove_version_suffix:
            logger.info("Removing version suffixes from Gene IDs")
            adata.var["gene_upper"] = adata.var["gene_upper"].str.replace(r"\..*$", "", regex=True)

        adata = _map_gene_annotation(adata, anno_df, gene_col)

        biotype_col = "gene_biotype"
        chr_col = "chrom"
        start_col = "start"
        end_col = "end"
    else:
        gene_col = _pick_var_col(adata, ["gene_symbol", "gene_name", "symbol", "hgnc_symbol", "gene"], gene_col)
        biotype_col = _pick_var_col(adata, ["gene_biotype", "biotype", "feature_biotype", "gene_type"], biotype_col)
        chr_col = _pick_var_col(adata, ["chrom", "chr", "chromosome", "seqname"], chr_col)
        start_col = _pick_var_col(adata, ["start", "start_position", "gene_start"], start_col)
        end_col = _pick_var_col(adata, ["end", "end_position", "gene_end"], end_col)
    adata.var_names = adata.var[gene_col].astype(str)
    adata.var_names_make_unique()

    logger.info(
        f"Using annotation columns: gene={gene_col}, biotype={biotype_col}, chr={chr_col}, start={start_col}, end={end_col}"
    )

    logger.info("Applying gene filters")
    masks = {}

    if filter_protein_coding and biotype_col:
        biotype = adata.var[biotype_col].astype(str).str.lower()
        masks["protein_coding"] = biotype.isin(["protein_coding", "protein-coding", "protein coding"])
        logger.info(f"Protein-coding genes: {masks['protein_coding'].sum()}")
    else:
        masks["protein_coding"] = pd.Series(True, index=adata.var_names)
        if filter_protein_coding:
            logger.warning("No biotype column found; skipping protein-coding filter")

    if filter_expressed:
        X = adata.X
        if sparse.issparse(X):
            gene_sum = np.asarray(X.sum(axis=0)).ravel()
        else:
            gene_sum = X.sum(axis=0).ravel()
        masks["expressed"] = pd.Series(gene_sum > 0, index=adata.var_names)
        logger.info(f"Expressed genes: {masks['expressed'].sum()}")
    else:
        masks["expressed"] = pd.Series(True, index=adata.var_names)

    masks["unique"] = pd.Series(True, index=adata.var_names)

    if filter_mhc and all(c for c in [chr_col, start_col, end_col]):
        in_mhc_chr = adata.var[chr_col] == str(mhc_chr)
        overlaps_mhc = in_mhc_chr & (adata.var[end_col] >= mhc_start) & (adata.var[start_col] <= mhc_end)
        masks["not_mhc"] = ~overlaps_mhc.fillna(False)
        logger.info(f"Non-MHC genes: {masks['not_mhc'].sum()}")
    else:
        masks["not_mhc"] = pd.Series(True, index=adata.var_names)
        if filter_mhc:
            logger.warning("Missing chr/start/end columns; skipping MHC filter")

    mask_keep = pd.Series(True, index=adata.var_names)
    for mask_name, mask in masks.items():
        mask_keep &= mask

    n_before = adata.n_vars
    n_after = mask_keep.sum()
    logger.info(f"Keeping {n_after} / {n_before} genes after filtering")

    adata = adata[:, mask_keep.values].copy()


    if log_transform:
        # Work with categorical clusters
        clusters_cat = adata.obs[celltype_col].astype("category")
        cluster_names = clusters_cat.cat.categories.to_list()
        n_clusters = len(cluster_names)
        n_cells, n_genes = adata.shape
        logger.info(f"n_cells = {n_cells}, n_genes = {n_genes}, n_clusters = {n_clusters}")

        # matrix: genes × clusters
        avg_matrix = np.zeros((n_genes, n_clusters), dtype=np.float64)
        X = adata.X # could be csr_matrix or dense

        
        # Compute per-cluster log1p mean
        logger.info("Applying log1p transformation")
        for j, cl in enumerate(tqdm(cluster_names, desc="Aggregating clusters")):
            # indices of cells in this cluster
            idx = np.where(clusters_cat.values == cl)[0]
            if idx.size == 0:
            # no cells in this cluster (shouldn't usually happen, but just in case)
                avg_matrix[:, j] = 0.0
                continue

            # subset expression for these cells: shape (n_cells_in_cluster, n_genes)
            X_sub = X[idx, :]

            # Convert to dense if sparse
            if hasattr(X_sub, "toarray"):
                X_sub = X_sub.toarray()

            # log1p transform and average over cells (axis 0, since rows=cells, cols=genes)
            # Using numexpr to speed up log1p and sum
            # log1p(X_sub) is applied per element; sum over cells => axis=0 => length n_genes
            # careful: numexpr works on 1D or 2D arrays; we keep it 2D here
            log1p_X_sub = ne.evaluate("log1p(X_sub)")
            avg_expr = log1p_X_sub.mean(axis=0)  # 1D, length n_genes

            # Store as genes × clusters → [gene, cluster_index]
            avg_matrix[:, j] = avg_expr
        
        df = pd.DataFrame(
            avg_matrix,
            index=adata.var_names,      # genes as rows
            columns=cluster_names       # clusters as columns
        )

        logger.info("Log1p applied.")


    if not log_transform:
        raise ValueError("This preprocessing path expects log_transform=True (needs cluster-level log1p matrix).")

    # Wide table from the matrix you computed
    exp_wide = df.copy().reset_index()
    exp_wide = exp_wide.rename(columns={"index": "gene"})

    # If reset_index() produced column named "index" instead:
    if "gene" not in exp_wide.columns and "index" in exp_wide.columns:
        exp_wide = exp_wide.rename(columns={"index": "gene"})

    clusters = [c for c in exp_wide.columns if c != "gene"]

    # copy wide table
    exp = exp_wide.copy()

    # add_count(gene)
    exp["n"] = exp.groupby("gene")["gene"].transform("count")

    # keep only genes with n == 1  (THIS is stricter than your old "unique" mask)
    exp = exp.loc[exp["n"] == 1].drop(columns=["n"])

    # gather/melt to long
    exp = exp.melt(
        id_vars="gene",
        var_name="ClusterID",   # cluster name
        value_name="Expr_sum_mean"  # your log1p mean expression
    )

    logger.info(f"Computing mean expression for {celltype_col}")
    # normalize within each cluster to sum to 1000
    exp["Expr_sum_mean"] = (
        exp["Expr_sum_mean"] * 1000.0 /
        exp.groupby("ClusterID")["Expr_sum_mean"].transform("sum")
    )
    mean_expr_df = exp.pivot(index="gene", columns="ClusterID", values="Expr_sum_mean")


    logger.info("Computing specificity scores")

    # specificity: fraction of gene's total that comes from this cluster
    exp["specificity"] = (
        exp["Expr_sum_mean"] /
        exp.groupby("gene")["Expr_sum_mean"].transform("sum")
    )
    specificity_df = exp.pivot(index="gene", columns="ClusterID", values="specificity")


    if not ((specificity_df.values >= 0) & (specificity_df.values <= 1)).all():
        logger.warning("Some specificity values outside [0, 1] range")

    logger.info(f"Final data shape: {adata.shape}")
    logger.info(f"Mean expression shape: {mean_expr_df.shape}")
    logger.info(f"Specificity shape: {specificity_df.shape}")

    if inplace:
        return None
    return adata, mean_expr_df, specificity_df


def generate_sldsc_genesets(
    specificity_df: pd.DataFrame,
    adata: AnnData,
    *,
    out_dir: str | Path,
    top_frac: float = 0.10,
    gene_col: str | None = "gene",          # e.g. "gene" (symbols) OR "ensembl_gene_id"
    accession_col: str | None = None,       # if you have an explicit Ensembl ID column, pass it (recommended)
    remove_version_suffix: bool = True,
    include_control: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate cell-type-specific gene sets for S-LDSC analysis.

    Expects specificity_df to be genes × cell types, indexed by gene identifiers
    (symbols or Ensembl IDs). Writes one .GeneSet per cell type containing top N%
    genes by specificity, using accession (typically Ensembl gene IDs).
    """
    out_dir = Path(out_dir)

    # ---- Safety checks ----
    if specificity_df.index.name != "gene":
        # not required, but helps debugging
        logger.info(f"specificity_df index name is '{specificity_df.index.name}', expected 'gene' (ok).")

    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists. Set overwrite=True to proceed.")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing gene sets to {out_dir}")

    # ---- Build mapping from specificity_df genes -> accessions to write ----
    spec_genes = pd.Index(specificity_df.index.astype(str))

    # Normalize the specificity_df index
    spec_upper = spec_genes.str.upper()
    if remove_version_suffix:
        spec_upper = spec_upper.str.replace(r"\..*$", "", regex=True)

    # Case 1: specificity_df already contains Ensembl IDs and you want to write them as-is.
    # We detect this if most genes look like ENSG...
    ensembl_like = spec_upper.str.match(r"^ENSG\d+$", na=False).mean() > 0.5

    # If user provided accession_col, use it as the authoritative output IDs
    if accession_col is not None:
        if accession_col not in adata.var.columns:
            raise ValueError(f"Column '{accession_col}' not found in adata.var")

        acc = adata.var[accession_col].astype(str).str.upper()
        if remove_version_suffix:
            acc = acc.str.replace(r"\..*$", "", regex=True)

        # Decide what to match on (gene_col or var_names)
        if gene_col is not None and gene_col in adata.var.columns:
            key = adata.var[gene_col].astype(str).str.upper()
            if remove_version_suffix:
                key = key.str.replace(r"\..*$", "", regex=True)
        else:
            key = pd.Index(adata.var_names.astype(str)).str.upper()
            if remove_version_suffix:
                key = key.str.replace(r"\..*$", "", regex=True)

        map_df = pd.DataFrame({"key": key.values, "accession": acc.values}).dropna()
        map_df = map_df.drop_duplicates(subset=["key"], keep="first")

        # Map specificity genes -> accession
        gene_to_acc = pd.Series(map_df["accession"].values, index=map_df["key"].values)
        accessions = gene_to_acc.reindex(spec_upper)

        overlap_mask = accessions.notna()
        if overlap_mask.sum() == 0:
            raise ValueError("No overlapping genes between specificity_df and adata.var mapping (accession_col).")

        logger.info(f"Overlapping genes after mapping: {overlap_mask.sum()}/{len(specificity_df)}")
        specificity_df = specificity_df.loc[overlap_mask.values]
        accessions = accessions.loc[overlap_mask.values]

        # Replace index with accessions (what LDSC wants)
        specificity_df = specificity_df.copy()
        specificity_df.index = accessions.values

    else:
        # No accession_col supplied.
        # If specificity_df already looks like Ensembl IDs, just use it.
        if ensembl_like:
            logger.info("specificity_df index looks like Ensembl IDs; using them directly.")
            specificity_df = specificity_df.copy()
            specificity_df.index = spec_upper.values
        else:
            # Fall back to matching against adata.var[gene_col] and writing those IDs.
            if gene_col is None or gene_col not in adata.var.columns:
                raise ValueError(
                    "specificity_df index does not look like Ensembl IDs, and no valid gene_col/accession_col provided."
                )

            adata_key = adata.var[gene_col].astype(str).str.upper()
            if remove_version_suffix:
                adata_key = adata_key.str.replace(r"\..*$", "", regex=True)

            overlap = spec_upper.intersection(pd.Index(adata_key))
            if overlap.empty:
                raise ValueError("No overlapping genes found between specificity_df index and adata.var[gene_col].")

            logger.info(f"Overlapping genes: {len(overlap)}/{specificity_df.shape[0]}")
            specificity_df = specificity_df.loc[spec_upper.isin(overlap).values].copy()
            specificity_df.index = spec_upper[spec_upper.isin(overlap)].values  # normalized

    # ---- Select top genes per cell type ----
    n_genes = specificity_df.shape[0]
    k = max(1, int(np.ceil(top_frac * n_genes)))
    logger.info(f"Selecting top {k} genes ({top_frac*100:.1f}%) per cell type")

    summary = []
    for celltype in specificity_df.columns:
        top_genes = specificity_df[celltype].nlargest(k).index.astype(str).unique()

        safe_celltype = _safe_filename(celltype)
        out_path = out_dir / f"{safe_celltype}.GeneSet"

        with open(out_path, "w") as f:
            for gene_id in top_genes:
                f.write(f"{gene_id}\n")

        summary.append({"cell_type": celltype, "n_genes": len(top_genes), "output_path": str(out_path)})
        logger.debug(f"Wrote {len(top_genes)} genes for {celltype}")

    # ---- Control geneset ----
    if include_control:
        control_path = out_dir / "Control.GeneSet"
        with open(control_path, "w") as f:
            for gene_id in specificity_df.index.astype(str):
                f.write(f"{gene_id}\n")
        logger.info(f"Wrote control gene set with {len(specificity_df)} genes")

    summary_df = pd.DataFrame(summary)
    logger.info(f"Generated {len(summary)} cell-type-specific gene sets")
    return summary_df



def _fetch_ensembl_annotation(
    genome_build: Literal["GRCh37", "GRCh38"] = "GRCh37", gene_identifier_mode: str = "ensembl"
) -> pd.DataFrame:
    """
    Fetch gene annotations from Ensembl using pybiomart.

    Parameters
    ----------
    genome_build
        Genome build version: "GRCh37" or "GRCh38".
    gene_identifier_mode
        Gene identifier: "name" (gene symbols) or "ensembl" (Ensembl IDs).

    Returns
    -------
    pd.DataFrame
        Gene annotations with columns: gene, chrom, start, end, gene_biotype.
    """
    try:
        from pybiomart import Server
    except ImportError as e:
        raise ImportError(
            "pybiomart is required for fetching gene annotations. " "Install it with: pip install pybiomart"
        ) from e

    if genome_build == "GRCh37":
        logger.info("Querying Ensembl BioMart (GRCh37)...")
        server = Server(host="http://grch37.ensembl.org")
    elif genome_build == "GRCh38":
        logger.info("Querying Ensembl BioMart (GRCh38)...")
        server = Server(host="http://www.ensembl.org")
    else:
        raise ValueError(f"Invalid genome_build: {genome_build}. Must be 'GRCh37' or 'GRCh38'")

    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]

    attributes = [
        "hgnc_symbol",
        "external_gene_name",
        "ensembl_gene_id",
        "chromosome_name",
        "start_position",
        "end_position",
        "gene_biotype",
    ]

    logger.info(f"Fetching gene annotations from {genome_build}...")
    anno = dataset.query(attributes=attributes, use_attr_names=True)

    anno.columns = [c.strip() for c in anno.columns]

    anno = anno.rename(columns={
        # Pretty BioMart labels
        "HGNC symbol": "hgnc_symbol",
        "Gene name": "external_gene_name",
        "Gene stable ID": "ensembl_gene_id",
        "Chromosome/scaffold name": "chrom",
        "Gene start (bp)": "start",
        "Gene end (bp)": "end",
        "Gene type": "gene_biotype",

        # Attribute-name style (very common with GRCh38)
        "hgnc_symbol": "hgnc_symbol",
        "external_gene_name": "external_gene_name",
        "ensembl_gene_id": "ensembl_gene_id",
        "chromosome_name": "chrom",
        "start_position": "start",
        "end_position": "end",
        "gene_biotype": "gene_biotype",
    })

    if gene_identifier_mode == "name":
        anno["gene"] = anno["hgnc_symbol"].replace("", pd.NA)
        anno["gene"] = anno["gene"].fillna(anno["external_gene_name"])
    elif gene_identifier_mode == "ensembl":
        anno["gene"] = anno["ensembl_gene_id"].replace("", pd.NA)
    else:
        raise ValueError(f"Invalid mode: {gene_identifier_mode}. Must be 'name' or 'ensembl'.")

    anno = anno[["gene", "chrom", "start", "end", "gene_biotype"]].dropna(subset=["gene"])

    logger.info(f"Fetched annotations for {len(anno)} genes from {genome_build}")
    return anno


def _map_gene_annotation(adata: AnnData, anno_df: pd.DataFrame, gene_col: str | None = "gene") -> AnnData:
    """Map gene annotations to adata.var."""
    anno_cols = ["chrom", "start", "end", "gene_biotype"]

    conflicts = [c for c in anno_cols if c in adata.var.columns]
    if conflicts:
        logger.info(f"Dropping conflicting columns from adata.var before merge: {conflicts}")
        adata.var = adata.var.drop(columns=conflicts)

    anno_df["gene_upper"] = anno_df["gene"].astype(str).str.upper()

    anno_df = anno_df.drop_duplicates(subset=["gene_upper"])

    merged = adata.var.merge(anno_df[["gene_upper"] + anno_cols], on="gene_upper", how="left")

    adata.var = merged.set_index(adata.var.index)

    logger.info(f"Annotated {(~merged['chrom'].isna()).sum()} / {adata.n_vars} genes.")

    return adata


def _pick_var_col(adata: AnnData, candidates: list[str], default: str | None) -> str | None:
    """Select first existing column from candidates."""
    if default and default in adata.var.columns:
        return default
    for col in candidates:
        if col in adata.var.columns:
            return col
    return default


def _normalize_chromosome(chr_series: pd.Series) -> pd.Series:
    """Normalize chromosome labels to standard format."""
    normalized = chr_series.astype(str).str.replace("^chr", "", regex=True).str.upper()
    return normalized.str.extract(r"^([0-9XYM]+)", expand=False)


def _compute_celltype_means(adata: AnnData, celltype_col: str) -> pd.DataFrame:
    """Compute mean expression per cell type."""
    celltypes = pd.Index(adata.obs[celltype_col].astype("category")).categories

    means_list = []
    col_names = []

    for celltype in celltypes:
        mask = (adata.obs[celltype_col] == celltype).values
        n_cells = mask.sum()
        col_names.append(str(celltype))

        if n_cells == 0:
            means_list.append(np.full(adata.n_vars, np.nan, dtype=float))
            logger.warning(f"Cell type '{celltype}' has 0 cells")
            continue

        X_sub = adata.X[mask, :]
        if sparse.issparse(X_sub):
            means = np.asarray(X_sub.mean(axis=0)).ravel()
        else:
            means = X_sub.mean(axis=0).ravel()

        means_list.append(means)

    mean_expr_df = pd.DataFrame(
        np.column_stack(means_list) if means_list else np.empty((adata.n_vars, 0)),
        index=adata.var_names,
        columns=col_names,
    )

    return mean_expr_df


def _compute_specificity(mean_expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute specificity scores using Duncan et al. 2019 method.

    Specificity(gene, celltype) = mean(gene, celltype) / sum(mean(gene, all_celltypes))
    """
    gene_sums = mean_expr_df.sum(axis=1)
    denom = gene_sums.replace(0, np.nan)
    specificity = mean_expr_df.div(denom, axis=0).fillna(0.0)

    return specificity


def _safe_filename(s: str) -> str:
    """Convert string to safe filename."""
    s = str(s).strip().replace(" ", "_")
    s = s.replace("(", "_").replace(")", "_")
    return re.sub(r"[^\w\.\+\-]+", "_", s)


def generate_gene_coord_file(
    out_path: str | Path,
    *,
    genome_build: Literal["GRCh37", "GRCh38"] = "GRCh37",
    gene_identifier_mode: str = "ensembl",
    remove_version_suffix: bool = True,
    add_chr_prefix: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate a gene coordinate file for S-LDSC analysis from Ensembl BioMart.

    Fetches all genes from Ensembl and creates a tab-delimited file with columns:
    GENE, CHR, START, END

    Parameters
    ----------
    out_path
        Output file path (e.g., "gene_coords.txt").
    genome_build
        Genome build version: "GRCh37" or "GRCh38".
    gene_identifier_mode
        Gene identifier: "name" (gene symbols) or "ensembl" (Ensembl IDs).
    remove_version_suffix
        Whether to remove version suffixes from gene IDs (e.g., ENSG00000123456.7 → ENSG00000123456).
    add_chr_prefix
        Whether to add "chr" prefix to chromosome names (e.g., "1" → "chr1").
    overwrite
        Whether to overwrite existing output file.

    Raises
    ------
    FileExistsError
        If out_path exists and overwrite=False.
    ImportError
        If pybiomart is not installed.

    Examples
    --------
    >>> # Fetch all genes with Ensembl IDs from GRCh37
    >>> coord_df = generate_gene_coord_file("gene_coords.txt", gene_identifier_mode="ensembl", genome_build="GRCh37")
    >>> # Fetch with gene symbols from GRCh38
    >>> coord_df = generate_gene_coord_file(
    ...     "gene_coords_grch38.txt", gene_identifier_mode="name", genome_build="GRCh38"
    ... )
    """
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file {out_path} already exists. Set overwrite=True to proceed.")

    logger.info(f"Fetching gene annotations from Ensembl {genome_build}...")
    anno_df = _fetch_ensembl_annotation(genome_build=genome_build, gene_identifier_mode=gene_identifier_mode)

    coord_df = anno_df[["gene", "chrom", "start", "end"]].copy()
    coord_df.columns = ["GENE", "CHR", "START", "END"]

    if remove_version_suffix:
        logger.info("Removing version suffixes from gene identifiers")
        coord_df["GENE"] = coord_df["GENE"].astype(str).str.replace(r"\..*$", "", regex=True)

    coord_df["CHR"] = coord_df["CHR"].astype(str)
    if add_chr_prefix:
        coord_df["CHR"] = coord_df["CHR"].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
    else:
        coord_df["CHR"] = coord_df["CHR"].str.replace("^chr", "", regex=True)

    coord_df["START"] = coord_df["START"].astype(int)
    coord_df["END"] = coord_df["END"].astype(int)

    n_before = len(coord_df)
    coord_df = coord_df.drop_duplicates(subset=["GENE"], keep="first")
    n_after = len(coord_df)

    if n_before != n_after:
        logger.warning(f"Removed {n_before - n_after} duplicate gene entries")

    coord_df = coord_df.sort_values(["CHR", "START"])

    logger.info(f"Writing {len(coord_df)} gene coordinates to {out_path}")
    coord_df.to_csv(out_path, sep="\t", index=False)

    logger.info(f"Successfully created gene coordinate file: {out_path}")
