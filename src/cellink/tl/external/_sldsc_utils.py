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
    mhc_chr: str = "6",  ######################################################
    mhc_start: int = 25_000_000,  ######################################################
    mhc_end: int = 34_000_000,  ######################################################
    fetch_annotation: bool = True,
    genome_build: Literal["GRCh37", "GRCh38"] = "GRCh37",
    gene_col: str | None = None,
    biotype_col: str | None = None,
    chr_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> tuple[AnnData, pd.DataFrame, pd.DataFrame] | None:
    """
    Preprocess single-cell data for S-LDSC cell-type-specific analysis.

    This function performs comprehensive preprocessing including:
    - Optional log1p transformation
    - Gene annotation fetching from Ensembl BioMart (GRCh37 or GRCh38)
    - Gene filtering (protein-coding, expressed, unique names, MHC exclusion)
    - Computation of mean expression per cell type
    - Computation of specificity scores (Duncan et al. 2019 method)

    Parameters
    ----------
    adata
        Annotated data matrix of shape `n_obs` × `n_vars` (cells × genes).
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
    gene_col
        Column name for gene symbols. If None, uses var_names or auto-detects.
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
    copy
        Whether to modify a copy of the input object. Not compatible with inplace=False.

    Returns
    -------
    AnnData, pd.DataFrame, pd.DataFrame
        - Filtered AnnData object
        - Mean log expression per cell type (genes × cell types)
        - Specificity scores per cell type (genes × cell types)
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
        logger.info(f"Log1p applied. Sparse matrix: {sparse.issparse(adata.X)}")

    # Step 2: Fetch or validate gene annotations
    if fetch_annotation:
        anno_df = _fetch_ensembl_annotation(genome_build=genome_build)
        adata = _map_gene_annotation(adata, anno_df, gene_col)

        # Update column references after annotation
        gene_col = "gene" if gene_col is None else gene_col
        biotype_col = "gene_biotype"
        chr_col = "chrom"
        start_col = "start"
        end_col = "end"
    else:
        # Auto-detect columns if not provided
        gene_col = _pick_var_col(adata, ["gene_symbol", "gene_name", "symbol", "hgnc_symbol", "gene"], gene_col)
        biotype_col = _pick_var_col(adata, ["gene_biotype", "biotype", "feature_biotype", "gene_type"], biotype_col)
        chr_col = _pick_var_col(adata, ["chrom", "chr", "chromosome", "seqname"], chr_col)
        start_col = _pick_var_col(adata, ["start", "start_position", "gene_start"], start_col)
        end_col = _pick_var_col(adata, ["end", "end_position", "gene_end"], end_col)

    logger.info(
        f"Using annotation columns: gene={gene_col}, biotype={biotype_col}, "
        f"chr={chr_col}, start={start_col}, end={end_col}"
    )

    # Step 3: Gene filtering
    logger.info("Applying gene filters")
    masks = {}

    # 3a: Protein-coding filter
    if filter_protein_coding and biotype_col:
        biotype = adata.var[biotype_col].astype(str).str.lower()
        masks["protein_coding"] = biotype.isin(["protein_coding", "protein-coding", "protein coding"])
        logger.info(f"Protein-coding genes: {masks['protein_coding'].sum()}")
    else:
        masks["protein_coding"] = pd.Series(True, index=adata.var_names)
        if filter_protein_coding:
            logger.warning("No biotype column found; skipping protein-coding filter")

    # 3b: Expressed genes filter
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

    # 3c: Unique gene names
    gene_name = (adata.var[gene_col] if gene_col else adata.var_names).astype(str)
    masks["unique"] = ~gene_name.duplicated(keep="first")
    logger.info(f"Unique gene names: {masks['unique'].sum()}")

    # 3d: MHC exclusion
    if filter_mhc and all(c for c in [chr_col, start_col, end_col]):
        chr_norm = _normalize_chromosome(adata.var[chr_col])
        pos_start = pd.to_numeric(adata.var[start_col], errors="coerce")
        pos_end = pd.to_numeric(adata.var[end_col], errors="coerce")

        in_mhc_chr = chr_norm == str(mhc_chr)
        overlaps_mhc = in_mhc_chr & (pos_end >= mhc_start) & (pos_start <= mhc_end)
        masks["not_mhc"] = ~overlaps_mhc.fillna(False)
        logger.info(f"Non-MHC genes: {masks['not_mhc'].sum()}")
    else:
        masks["not_mhc"] = pd.Series(True, index=adata.var_names)
        if filter_mhc:
            logger.warning("Missing chr/start/end columns; skipping MHC filter")

    # Combine all masks
    mask_keep = pd.Series(True, index=adata.var_names)
    for mask_name, mask in masks.items():
        mask_keep &= mask

    n_before = adata.n_vars
    n_after = mask_keep.sum()
    logger.info(f"Keeping {n_after} / {n_before} genes after filtering")

    # Apply filter (create copy to free memory)
    adata = adata[:, mask_keep.values].copy()

    # Clean var_names
    if gene_col:
        adata.var_names = adata.var[gene_col].astype(str)
    adata.var_names_make_unique()

    # Step 4: Compute mean expression per cell type
    logger.info(f"Computing mean expression for {celltype_col}")
    mean_expr_df = _compute_celltype_means(adata, celltype_col)

    # Step 5: Compute specificity scores (Duncan et al. 2019)
    logger.info("Computing specificity scores")
    specificity_df = _compute_specificity(mean_expr_df)

    # Validate specificity
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
    accession_col: str = "accession",
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
    accession_col
        Column name in adata.var containing Ensembl gene IDs.
    remove_version_suffix
        Whether to remove version suffixes from Ensembl IDs (e.g., ENSG00000123456.7 → ENSG00000123456).
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

    # Validate inputs
    if accession_col not in adata.var.columns:
        raise ValueError(f"Column '{accession_col}' not found in adata.var")

    # Create output directory
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists. Set overwrite=True to proceed.")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing gene sets to {out_dir}")

    # Convert specificity row index to integer positions
    try:
        idx = pd.to_numeric(specificity_df.index, errors="coerce").astype(int).to_numpy()
        if np.isnan(idx).any():
            raise ValueError("Specificity DataFrame index contains non-numeric values")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert specificity_df index to integer positions: {e}") from e

    # Validate indices are within bounds
    if idx.max() >= adata.n_vars or idx.min() < 0:
        raise ValueError(
            f"Specificity index out of bounds: [{idx.min()}, {idx.max()}] " f"for adata with {adata.n_vars} genes"
        )

    # Extract Ensembl accessions at these positions
    accessions = adata.var[accession_col].astype(str).iloc[idx].reset_index(drop=True)

    # Remove version suffixes if requested
    if remove_version_suffix:
        logger.info("Removing version suffixes from Ensembl IDs")
        accessions = accessions.str.replace(r"\..*$", "", regex=True)

    # Replace numeric indices with accessions
    specificity_df = specificity_df.copy()
    specificity_df.index = accessions.values

    # Compute number of top genes to select
    n_genes = specificity_df.shape[0]
    k = max(1, int(np.ceil(top_frac * n_genes)))
    logger.info(f"Selecting top {k} genes ({top_frac*100:.1f}%) per cell type")

    # Generate gene set for each cell type
    summary = []
    for celltype in specificity_df.columns:
        # Select top-k genes by specificity
        top_genes = specificity_df[celltype].nlargest(k).index.astype(str).unique()

        # Write gene set file
        safe_celltype = _safe_filename(celltype)
        out_path = out_dir / f"{safe_celltype}.GeneSet"

        with open(out_path, "w") as f:
            for gene_id in top_genes:
                f.write(f"{gene_id}\n")

        summary.append({"cell_type": celltype, "n_genes": len(top_genes), "output_path": str(out_path)})
        logger.debug(f"Wrote {len(top_genes)} genes for {celltype}")

    # Create control gene set if requested
    if include_control:
        control_path = out_dir / "Control.GeneSet"
        with open(control_path, "w") as f:
            for gene_id in specificity_df.index.astype(str):
                f.write(f"{gene_id}\n")
        logger.info(f"Wrote control gene set with {len(specificity_df)} genes")

    summary_df = pd.DataFrame(summary)
    logger.info(f"Generated {len(summary)} cell-type-specific gene sets")

    return summary_df


# ==================== Helper Functions ====================


def _fetch_ensembl_annotation(genome_build: Literal["GRCh37", "GRCh38"] = "GRCh37") -> pd.DataFrame:
    """
    Fetch gene annotations from Ensembl using pybiomart.

    Parameters
    ----------
    genome_build
        Genome build version: "GRCh37" or "GRCh38".

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

    # Select appropriate server based on genome build
    if genome_build == "GRCh37":
        logger.info("Querying Ensembl BioMart (GRCh37)...")
        server = Server(host="http://grch37.ensembl.org")
    elif genome_build == "GRCh38":
        logger.info("Querying Ensembl BioMart (GRCh38)...")
        server = Server(host="http://www.ensembl.org")
    else:
        raise ValueError(f"Invalid genome_build: {genome_build}. Must be 'GRCh37' or 'GRCh38'")

    # Get human dataset
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]

    # Query for gene annotations
    attributes = [
        "hgnc_symbol",
        "external_gene_name",
        "chromosome_name",
        "start_position",
        "end_position",
        "gene_biotype",
    ]

    logger.info(f"Fetching gene annotations from {genome_build}...")
    anno = dataset.query(attributes=attributes)

    # Standardize column names
    anno = anno.rename(
        columns={
            "HGNC symbol": "hgnc_symbol",
            "Gene name": "external_gene_name",
            "Chromosome/scaffold name": "chrom",
            "Gene start (bp)": "start",
            "Gene end (bp)": "end",
            "Gene type": "gene_biotype",
        }
    )

    # Create unified gene column (prefer HGNC, fallback to external name)
    anno["gene"] = anno["hgnc_symbol"].replace("", pd.NA)
    anno["gene"] = anno["gene"].fillna(anno["external_gene_name"])

    anno = anno[["gene", "chrom", "start", "end", "gene_biotype"]].dropna(subset=["gene"])

    logger.info(f"Fetched annotations for {len(anno)} genes from {genome_build}")
    return anno


def _map_gene_annotation(adata: AnnData, anno_df: pd.DataFrame, gene_col: str | None = None) -> AnnData:
    """Map gene annotations to adata.var."""
    # Deduplicate annotation on uppercase gene symbols
    anno_clean = anno_df.dropna(subset=["gene"]).copy()
    anno_clean["gene_upper"] = anno_clean["gene"].astype(str).str.upper()
    anno_clean = anno_clean.drop_duplicates(subset=["gene_upper"], keep="first")

    # Normalize chromosome labels
    anno_clean["chrom"] = anno_clean["chrom"].astype(str).str.replace("^chr", "", regex=True).str.strip()
    anno_clean.loc[anno_clean["chrom"].isin(["M", "MT", "Mt", "MtDNA"]), "chrom"] = "MT"

    # Ensure numeric coordinates
    anno_clean["start"] = pd.to_numeric(anno_clean["start"], errors="coerce")
    anno_clean["end"] = pd.to_numeric(anno_clean["end"], errors="coerce")

    # Build lookup dictionaries
    map_chrom = pd.Series(anno_clean["chrom"].values, index=anno_clean["gene_upper"]).to_dict()
    map_start = pd.Series(anno_clean["start"].values, index=anno_clean["gene_upper"]).to_dict()
    map_end = pd.Series(anno_clean["end"].values, index=anno_clean["gene_upper"]).to_dict()
    map_biotype = pd.Series(anno_clean["gene_biotype"].values, index=anno_clean["gene_upper"]).to_dict()

    # Determine gene source
    if gene_col and gene_col in adata.var.columns:
        genes = adata.var[gene_col]
    else:
        genes = pd.Series(adata.var_names, index=adata.var_names)

    # Add uppercase key and map annotations
    adata.var["gene_upper"] = genes.astype(str).str.upper()
    adata.var["chrom"] = adata.var["gene_upper"].map(map_chrom)
    adata.var["start"] = adata.var["gene_upper"].map(map_start)
    adata.var["end"] = adata.var["gene_upper"].map(map_end)
    adata.var["gene_biotype"] = adata.var["gene_upper"].map(map_biotype)

    n_annotated = adata.var["chrom"].notna().sum()
    logger.info(f"Annotated {n_annotated} / {adata.n_vars} genes with GRCh37 coordinates")

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


if __name__ == "__main__":
    adata = sc.read_h5ad(
        "/Users/larnoldt/sc-genetics/Sc-annotation_generation/ayshan_dataset_subsampled_joined_chunks.h5ad"
    )

    # Using GRCh37 (default)
    adata_filtered, mean_expr, specificity = preprocess_for_sldsc(
        adata,
        celltype_col="roi_group_coarse",  # cell_type
        genome_build="GRCh37",
        inplace=False,
    )

    """
    # Using GRCh38
    adata_filtered, mean_expr, specificity = preprocess_for_sldsc(
        adata,
        celltype_col="roi_group_coarse",
        genome_build="GRCh38",
        inplace=False
    )
    """

    # Skip annotation fetching if already present
    adata_filtered, mean_expr, specificity = preprocess_for_sldsc(
        adata, celltype_col="roi_group_coarse", fetch_annotation=False, inplace=False
    )
