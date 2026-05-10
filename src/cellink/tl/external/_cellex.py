import logging
import subprocess
from pathlib import Path
from typing import Literal, Union, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import statsmodels.api as sm
import statsmodels.tools.tools as sm_tools
import os

logger = logging.getLogger(__name__)

try:
    import cellex
    CELLEX_AVAILABLE = True
except ImportError:
    CELLEX_AVAILABLE = False


def run_cellex(
    adata: AnnData,
    cell_type_col: str,
    species: Literal["human", "mouse"] = "human",
    map_to_human: bool = False,
    min_genes: int = 250,
    min_cells: int = 50,
    prefix: str = "cellex_results",
) -> str:
    """
    Run CELLEX to compute cell-type Expression Specificity (ES) profiles.

    CELLEX integrates multiple ES metrics using a "wisdom of the crowd" approach
    to obtain robust cell-type expression specificity scores (ESmu). These scores
    are used downstream by CELLECT to link cell types with GWAS traits.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell expression data (raw counts preferred).
    cell_type_col : str
        Column name in adata.obs containing cell type annotations.
    species : {'human', 'mouse'}, default='human'
        Species of the input data. If 'mouse' and map_to_human=True, gene IDs
        will be converted to human Ensembl IDs for compatibility with CELLECT.
    map_to_human : bool, default=False
        If True and species='mouse', map mouse Ensembl IDs to human Ensembl IDs.
        Requires that adata.var_names are mouse Ensembl IDs (ENSMUSGXXX).
    min_genes : int, default=250
        Minimum number of genes for cell filtering.
    min_cells : int, default=50
        Minimum number of cells for gene filtering.
    prefix : str, optional
        Prefix for output files. Default is "cellex".

    Returns
    -------
    pd.DataFrame
        ESmu DataFrame with genes as rows and cell types as columns.
        Values in [0, 1] representing expression specificity.

    Raises
    ------
    ImportError
        If cellex package is not installed.
    ValueError
        If cell_type_col is not in adata.obs.

    Examples
    --------
    >>> esmu = run_cellex(
    ...     dd.C,
    ...     cell_type_col="predicted.celltype.l2",
    ...     prefix="onek1k_cellex",
    ... )
    >>> print(esmu.head())

    >>> # Mouse data mapped to human
    >>> esmu = run_cellex(
    ...     adata_mouse,
    ...     cell_type_col="cell_type",
    ...     species="mouse",
    ...     map_to_human=True,
    ... )

    Notes
    -----
    - CELLEX expects raw (unnormalized) UMI counts; normalization is handled internally.
    - Gene IDs should be Ensembl IDs for compatibility with CELLECT.
    - For CELLECT, save_results=True will produce the .esmu.csv.gz file that CELLECT
      reads as its specificity input.
    """
    if not CELLEX_AVAILABLE:
        raise ImportError(
            "cellex is required for run_cellex. Install it with: pip install cellex"
        )

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs")

    logger.info("Filtering cells and genes")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata = adata[~adata.obs[cell_type_col].isna()].copy()

    logger.info("Preparing data for CELLEX")
    from scipy.sparse import issparse
    if issparse(adata.X):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X

    data = pd.DataFrame(
        expr_matrix.T,
        index=adata.var_names,
        columns=adata.obs_names,
    )

    metadata = pd.DataFrame(
        {"cell_type": adata.obs[cell_type_col].values},
        index=adata.obs_names,
    )

    logger.info("Creating CELLEX ESObject and computing ES scores")
    eso = cellex.ESObject(data=data, annotation=metadata, verbose=True)
    eso.compute(verbose=True)

    esmu = eso.results["esmu"].copy()

    if species == "mouse" and map_to_human:
        logger.info("Mapping mouse Ensembl IDs to human Ensembl IDs")
        cellex.utils.mapping.mouse_ens_to_human_ens(esmu, drop_unmapped=True, verbose=True)

    out_file = f"{prefix}.esmu.csv"
    esmu.to_csv(out_file, sep='\t')
    logger.info(f"Saved ESmu scores to {out_file}")

    return out_file

	
def fit_LM(specificity_id: str, es_mu: pd.DataFrame, df_magma: pd.DataFrame) -> pd.DataFrame:
    annotations = es_mu.columns[1:] 
    
    df_regression = pd.merge(es_mu, df_magma, left_on='gene', right_on='GENE', how='inner')
    
    results = []
    
    for annotation in annotations:
        y = df_regression['ZSTAT'] 
        X = sm_tools.add_constant(df_regression[annotation]) 
        
        ols = sm.OLS(y, X)
        ols_result = ols.fit()
        
        pval = ols_result.pvalues[1] / 2                           
        pval = 1 - pval if ols_result.params[1] < 0 else pval    
        
        results.append({
            "Name": f"{specificity_id}__{annotation}",
            "Coefficient": ols_result.params[1],
            "Coefficient_std_error": ols_result.bse[1],
            "Coefficient_P_value": pval
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=['Coefficient_P_value'])
    return df_res


def run_cellect_prioritization(
    gene_results_file: Union[str, Path],
    esmu_file: Union[str, Path],
    output_prefix: str = "cellect_results",
    specificity_id: str = "cellex",
    exclude_mhc: bool = True,
) -> Path:
    """
    Run CELLECT linear model prioritization using MAGMA gene-level results and CELLEX ES scores.
    Parameters
    ----------
    gene_results_file : str or Path
        Path to MAGMA gene-level results file (trait.genes.out).
    esmu_file : str or Path
        Path to CELLEX ESmu file (e.g., trait.cellex_results.esmu.csv).
    output_prefix : str
        Prefix for output files. The final results will be saved as {output_prefix}_cellect_results.txt.
    specificity_id : str, default="cellex"
        Identifier for the specificity matrix, used in the output "Name" column.
    exclude_mhc : bool, default=True
        Whether to exclude genes in the MHC region (chr6: 28.4-33.4 Mb) from the analysis, as they can
        confound results due to high gene density and complex LD structure.
    Returns
    -------
    Path
        Path to the output file containing CELLECT prioritization results.
    Raises
    ------
    FileNotFoundError
        If the gene results file or ESmu file does not exist.
    Examples
    --------
    >>> final_results_file = run_cellect_prioritization(
    ...     gene_results_file="results/trait.genes.out",
    ...     esmu_file="results/trait.cellex_results.esmu.csv",
    ...     output_prefix="results/trait",
    ...     exclude_mhc=True
    ... )
    """
    
    logger.info(f"Starting CELLECT Linear Model Prioritization for '{output_prefix}'...")

    gene_results_path = Path(gene_results_file)
    esmu_path = Path(esmu_file)

    if not gene_results_path.exists():
        raise FileNotFoundError(f"MAGMA results file not found: {gene_results_path}")
    if not esmu_path.exists():
        raise FileNotFoundError(f"ESMU file not found: {esmu_path}")

    logger.info(f"Loading MAGMA results: {gene_results_path}")
    df_magma = pd.read_csv(gene_results_path, sep=r'\s+', comment='#')

    if exclude_mhc:
        old_len = len(df_magma)
        df_magma = df_magma[
            (df_magma['START'] < 28477797) | 
            (df_magma['STOP'] > 33448354) | 
            (df_magma['CHR'] != 6)
        ]
        logger.info(f"{old_len - len(df_magma)} MHC region genes excluded.")

    logger.info(f"Loading Expression Specificity Matrix: {esmu_path}")
    es_mu = pd.read_csv(esmu_path, sep='\t')
    
    if 'gene' not in es_mu.columns:
        es_mu = es_mu.rename(columns={es_mu.columns[0]: 'gene'})

    logger.info("Running OLS regressions for each cell type...")
    df_res = fit_LM(specificity_id, es_mu, df_magma)

    outdir = Path(output_prefix).parent
    outdir.mkdir(exist_ok=True, parents=True)
    
    fullname = Path(f"{output_prefix}_prioritization.csv")
    df_res.to_csv(fullname, index=False)
    
    logger.info(f"CELLECT Prioritization successful. Results saved to: {fullname}")
    
    return fullname
