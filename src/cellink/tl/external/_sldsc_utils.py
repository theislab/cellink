import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

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
        - Mean log expression per cell type (genes x cell types)
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

    if log_transform:
        logger.info("Applying log1p transformation")
        sc.pp.log1p(adata)
        logger.info("Log1p applied.")

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

    masks["unique"] = ~adata.var[gene_col].duplicated(
        keep="first"
    )  # TODO: Does this make sense and not rather the sum?
    logger.info(f"Unique gene names: {masks['unique'].sum()}")

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

    logger.info(f"Computing mean expression for {celltype_col}")
    mean_expr_df = _compute_celltype_means(adata, celltype_col)

    logger.info("Computing specificity scores")
    specificity_df = _compute_specificity(mean_expr_df)

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
    gene_col: str | None = "gene",
    remove_version_suffix: bool = True,
    include_control: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate cell-type-specific gene sets for S-LDSC analysis.

    Creates .GeneSet files for each cell type containing top N% genes by specificity,
    using Ensembl gene IDs (accessions) as required by LDSC.

    Parameters
    ----------
    specificity_df
        DataFrame of specificity scores (genes × cell types).
        Index should be numeric positions matching adata.var indices.
    adata
        AnnData object containing gene annotations.
    out_dir
        Output directory for .GeneSet files.
    top_frac
        Fraction of genes to select per cell type (e.g., 0.10 for top 10%).
    gene_col
        Column name for gene symbols or IDs.
    remove_version_suffix
        Whether to remove version suffixes from gene names or gene IDs (e.g., ENSG00000123456.7 → ENSG00000123456).
    include_control
        Whether to create a Control.GeneSet file containing all genes.
    overwrite
        Whether to overwrite existing output directory.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: cell_type, n_genes, output_path.

    Raises
    ------
    ValueError
        If accession_col not found in adata.var or if specificity_df index is invalid.
    FileExistsError
        If out_dir exists and overwrite=False.

    Examples
    --------
    >>> summary = generate_sldsc_genesets(specificity_df, adata, out_dir="ldsc_genesets", top_frac=0.10)
    >>> print(summary)
    """
    out_dir = Path(out_dir)

    if gene_col not in adata.var.columns:
        raise ValueError(f"Column '{gene_col}' not found in adata.var")
    adata.var["gene_upper"] = adata.var[gene_col].str.upper()
    if remove_version_suffix:
        logger.info("Removing version suffixes from Gene IDs")
        adata.var["gene_upper"] = adata.var["gene_upper"].str.replace(r"\..*$", "", regex=True)
    specificity_df.index = specificity_df.index.str.upper()

    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists. Set overwrite=True to proceed.")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing gene sets to {out_dir}")

    overlap = specificity_df.index.intersection(adata.var["gene_upper"])
    if overlap.empty:
        raise ValueError("No overlapping genes found between specificity_df and adata.var")
    logger.info(f"Overlapping genes: {len(overlap)}/{specificity_df.shape[0]}")

    specificity_df = specificity_df.loc[overlap]

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
    anno = dataset.query(attributes=attributes)

    anno = anno.rename(
        columns={
            "HGNC symbol": "hgnc_symbol",
            "Gene name": "external_gene_name",
            "Gene stable ID": "ensembl_gene_id",
            "Chromosome/scaffold name": "chrom",
            "Gene start (bp)": "start",
            "Gene end (bp)": "end",
            "Gene type": "gene_biotype",
        }
    )

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
    return re.sub(r"[^\w\.-]+", "_", s)


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
