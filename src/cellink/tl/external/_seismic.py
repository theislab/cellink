import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Union, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
import scanpy as sc

logger = logging.getLogger(__name__)


def run_seismic(
    adata: AnnData,
    magma_file: str | Path,
    cell_type_col: str,
    n_pcs: int = 50,
    species: Literal["human", "mouse"] = "human",
    min_genes: int = 250,
    min_cells: int = 50,
    influential_genes: bool = False,
    influential_cell_types: list[str] | None = None,
    top_n_associations: int = 20,
    prefix: str = None,
    save_results: bool = True,
    plot_associations: bool = True,
    plot_influential: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Run seismic analysis to link cell types with GWAS traits using single-cell data.
    
    seismic (Single-cEll dIsease-relevance statIstical testing via Multi-resolution
    Cell-type specificity) identifies cell type-trait associations and influential
    genes driving these associations.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell expression data.
    magma_file : str or Path
        Path to MAGMA gene-level summary statistics file.
    cell_type_col : str
        Column name in adata.obs containing cell type annotations.
    n_pcs : int, default=50
        Number of principal components to compute if not already present.
    species : {'human', 'mouse'}, default='human'
        Species of the single-cell data.
    min_genes : int, default=250
        Minimum number of genes for cell filtering.
    min_cells : int, default=50
        Minimum number of cells for gene filtering.
    influential_genes : bool, default=False
        Whether to compute influential gene analysis for significant cell type-trait pairs.
    influential_cell_types : list of str, optional
        Specific cell types to analyze for influential genes. If None and influential_genes=True,
        analyzes all significant cell types.
    top_n_associations : int, default=20
        Number of top associations to plot.
    prefix : str, optional
        Prefix for output files. Default is "seismic".
    save_results : bool, default=True
        Whether to save results to files.
    plot_associations : bool, default=True
        Whether to plot top associations.
    plot_influential : bool, default=True
        Whether to plot influential genes (if computed).
    
    Returns
    -------
    pd.DataFrame or tuple
        If influential_genes=False: DataFrame with cell type-trait associations
        If influential_genes=True: tuple of (associations_df, dict of influential_genes_dfs)
    
    Raises
    ------
    RuntimeError
        If R or required R packages are not available.
    ValueError
        If cell_type_col is not in adata.obs.
    
    Examples
    --------
    >>> # Basic seismic analysis
    >>> associations = run_seismic(
    ...     dd,
    ...     magma_file="trait.genes.out",
    ...     cell_type_col="cell_type",
    ... )
    
    >>> # With influential gene analysis
    >>> associations, influential = run_seismic(
    ...     dd,
    ...     magma_file="trait.genes.out",
    ...     cell_type_col="cell_type",
    ...     influential_genes=True,
    ... )
    """
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs")
    
    if prefix is None:
        prefix = "seismic"
    
    logger.info("Preparing data for seismic analysis")
      
    logger.info("Filtering cells and genes")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    adata = adata[~adata.obs[cell_type_col].isna()].copy()
    
    if "log1p" not in adata.uns_keys():
        logger.info("Log-normalizing data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    logger.info("Exporting data for R")
    
    if issparse(adata.X):
        expr_matrix = adata.X.T.toarray()
    else:
        expr_matrix = adata.X.T
    
    expr_df = pd.DataFrame(
        expr_matrix,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    expr_file = f"{prefix}_expression.csv.gz"
    expr_df.to_csv(expr_file, compression='gzip')
    logger.info(f"Saved expression matrix: {expr_file}")
    
    metadata_file = f"{prefix}_metadata.csv"
    adata.obs[[cell_type_col]].to_csv(metadata_file)
    logger.info(f"Saved cell metadata: {metadata_file}")
    
    r_script = f"{prefix}_seismic.R"
    
    if influential_genes:
        if influential_cell_types is None:
            inf_celltypes_str = "NULL"
        else:
            inf_celltypes_vec = "c(" + ", ".join([f'"{ct}"' for ct in influential_cell_types]) + ")"
            inf_celltypes_str = inf_celltypes_vec
    
    r_code = f'''
library(seismicGWAS)
library(SingleCellExperiment)

# Load data
cat("Loading expression data...\\n")
expr_df <- read.csv("{expr_file}", row.names=1, check.names=FALSE)
metadata <- read.csv("{metadata_file}", row.names=1)

cat("Creating SingleCellExperiment...\\n")
# Create SCE object
sce <- SingleCellExperiment(
    assays = list(logcounts = as.matrix(expr_df))
)
colData(sce) <- DataFrame(metadata)

cat("Calculating cell type specificity...\\n")
# Calculate specificity scores
sscore <- calc_specificity(sce, ct_label_col="{cell_type_col}")

cat("Translating gene IDs...\\n")
# Translate gene IDs if needed
{"sscore_hsa <- translate_gene_ids(sscore, from='mmu_symbol')" if species == "mouse" else "sscore_hsa <- sscore"}

cat("Loading MAGMA results...\\n")
# Load MAGMA results
magma_df <- read.table("{magma_file}", header=TRUE)

cat("Computing cell type-trait associations...\\n")
# Get cell type-trait associations
ct_associations <- get_ct_trait_associations(sscore_hsa, magma_df)

# Save associations
write.table(ct_associations, "{prefix}_associations.tsv", 
            sep="\\t", row.names=FALSE, quote=FALSE)

cat("Saved associations to {prefix}_associations.tsv\\n")

# Plot top associations
{"png('" + prefix + "_top_associations.png', width=800, height=600)" if plot_associations else "# Plotting disabled"}
{"plot_top_associations(ct_associations, limit=" + str(top_n_associations) + ")" if plot_associations else ""}
{"dev.off()" if plot_associations else ""}

'''
    
    if influential_genes:
        r_code += f'''
cat("Computing influential genes...\\n")

# Determine cell types to analyze
if (is.null({inf_celltypes_str})) {{
    sig_celltypes <- ct_associations$cell_type[ct_associations$FDR < 0.05]
    if (length(sig_celltypes) == 0) {{
        cat("Warning: No significant cell types found (FDR < 0.05)\\n")
    }}
}} else {{
    sig_celltypes <- {inf_celltypes_str}
}}

# Analyze each cell type
for (ct in sig_celltypes) {{
    cat(paste0("  Processing ", ct, "...\\n"))
    tryCatch({{
        inf_genes <- find_inf_genes(ct, sscore_hsa, magma_df)
        
        # Save results
        safe_ct <- gsub(" ", "_", ct)
        write.table(inf_genes, 
                    paste0("{prefix}_influential_", safe_ct, ".tsv"),
                    sep="\\t", row.names=FALSE, quote=FALSE)
        
        # Plot if requested
        {"png(paste0('" + prefix + "_influential_', safe_ct, '.png'), width=800, height=600)" if plot_influential else ""}
        {"plot_inf_genes(inf_genes, num_labels=10)" if plot_influential else ""}
        {"dev.off()" if plot_influential else ""}
    }}, error = function(e) {{
        cat(paste0("  Warning: Could not compute influential genes for ", ct, ": ", e$message, "\\n"))
    }})
}}
'''
    
    r_code += '''
cat("Analysis complete!\\n")
'''
    
    with open(r_script, 'w') as f:
        f.write(r_code)
    
    logger.info(f"Created R script: {r_script}")
    
    logger.info("Running seismic analysis in R...")
    try:
        result = subprocess.run(
            ['Rscript', r_script],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"R script failed: {e.stderr}")
        raise RuntimeError(f"Seismic analysis failed: {e.stderr}")
    
    logger.info("Reading results")
    associations_df = pd.read_csv(f"{prefix}_associations.tsv", sep="\t")
    
    if not save_results:
        import os
        for f in [expr_file, metadata_file, r_script]:
            try:
                os.remove(f)
            except:
                pass
    
    influential_results = {}
    if influential_genes:
        if influential_cell_types is None:
            sig_celltypes = associations_df[associations_df['FDR'] < 0.05]['cell_type'].tolist()
        else:
            sig_celltypes = influential_cell_types
        
        for ct in sig_celltypes:
            safe_ct = ct.replace(' ', '_')
            inf_file = f"{prefix}_influential_{safe_ct}.tsv"
            if Path(inf_file).exists():
                inf_df = pd.read_csv(inf_file, sep="\t")
                influential_results[ct] = inf_df
    
    if influential_genes:
        return associations_df, influential_results
    else:
        return associations_df
