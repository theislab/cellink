import logging
from pathlib import Path
from typing import Literal, Union, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

try:
    import scdrs
    SCDRS_AVAILABLE = True
except ImportError:
    SCDRS_AVAILABLE = False

def run_scdrs(
    adata: AnnData,
    gs_file: str | Path = None,
    gene_sets: dict = None,
    src_species: str = "human",
    trait_name: str = None,
    n_pcs: int = 50,
    n_ctrl: int = 1000,
    weight_opt: Literal["vs", "uniform"] = "vs",
    ctrl_match_key: str = "mean_var",
    n_mean_bin: int = 20,
    n_var_bin: int = 20,
    flag_return_ctrl_raw_score: bool = False,
    flag_return_ctrl_norm_score: bool = True,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: list[str] | None = None,
    group_analysis: list[str] | None = None,
    corr_analysis: list[str] | None = None,
    gene_analysis: bool = False,
    knn_n_neighbors: int = 15,
    knn_n_pcs: int = 20,
    min_genes: int = 250,
    min_cells: int = 50,
    prefix: str = None,
    save_results: bool = True,
    return_adata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...], AnnData]:
    """
    Run scDRS (single-cell disease-relevance score) analysis on AnnData.
    
    scDRS associates individual cells in single-cell RNA-seq data with disease GWAS,
    computing cell-level disease scores and performing downstream analyses.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell expression data.
    gs_file : str or Path, optional
        Path to scDRS gene set file (.gs format).
    gene_sets : dict, optional
        Dictionary with trait names as keys and (gene_list, gene_weights) tuples as values.
        Either gs_file or gene_sets must be provided.
    src_species: str, optional, default='human'
        Species of the input gene sets. 
    trait_name : str, optional
        Name of specific trait to analyze from gs_file. If None, analyzes all traits.
    n_pcs : int, default=50
        Number of principal components to compute if not already present.
    n_ctrl : int, default=1000
        Number of control gene sets for null distribution.
    weight_opt : {'vs', 'uniform'}, default='vs'
        Weighting option: 'vs' for variance-stabilization, 'uniform' for equal weights.
    ctrl_match_key : str, default='mean_var'
        Key for matching control genes (stored in adata.var after preprocessing).
    n_mean_bin : int, default=20
        Number of bins for gene expression mean when matching control genes.
    n_var_bin : int, default=20
        Number of bins for gene expression variance when matching control genes.
    flag_return_ctrl_raw_score : bool, default=False
        Whether to return raw control scores.
    flag_return_ctrl_norm_score : bool, default=True
        Whether to return normalized control scores.
    encode_sex : bool, default=True
        Whether to include sex as a covariate.
    encode_age : bool, default=True
        Whether to include age as a covariate.
    additional_covariates : list of str, optional
        Additional covariates from dd.C.obs to include.
    group_analysis : list of str, optional
        List of cell group annotations in dd.C.obs for group-level analysis.
    corr_analysis : list of str, optional
        List of cell-level continuous variables in dd.C.obs for correlation analysis.
    gene_analysis : bool, default=False
        Whether to perform gene-level correlation analysis.
    knn_n_neighbors : int, default=15
        Number of neighbors for KNN graph (used in heterogeneity analysis).
    knn_n_pcs : int, default=20
        Number of PCs for computing KNN graph.
    min_genes : int, default=250
        Minimum number of genes for cell filtering.
    min_cells : int, default=50
        Minimum number of cells for gene filtering.
    prefix : str, optional
        Prefix for output files. Default is "scdrs".
    save_results : bool, default=True
        Whether to save results to files.
    return_adata : bool, default=False
        Whether to return the AnnData object with scDRS scores added.
    
    Returns
    -------
    pd.DataFrame or tuple or AnnData
        Depending on the analysis performed:
        - If only score computation: DataFrame with scDRS scores
        - If downstream analyses: tuple of DataFrames (scores, group_stats, cell_corr, gene_corr)
        - If return_adata=True: AnnData object with scores in .obs
    
    Raises
    ------
    ImportError
        If scdrs package is not installed.
    ValueError
        If neither gs_file nor gene_sets is provided.
    
    Examples
    --------
    >>> # Basic scDRS analysis
    >>> results = run_scdrs(
    ...     dd,
    ...     gs_file="traits.gs",
    ...     group_analysis=["cell_type"],
    ... )
    
    >>> # With custom gene sets
    >>> gene_sets = {
    ...     "MyDisease": (["GENE1", "GENE2", "GENE3"], [1.5, 2.0, 1.8])
    ... }
    >>> results = run_scdrs(dd, gene_sets=gene_sets)
    """
    if not SCDRS_AVAILABLE:
        raise ImportError(
            "scdrs is required for run_scdrs. Install it with: pip install scdrs"
        )
    
    if gs_file is None and gene_sets is None:
        raise ValueError("Either gs_file or gene_sets must be provided")
    
    if prefix is None:
        prefix = "scdrs"
        
    logger.info("Filtering cells and genes")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if "log1p" not in adata.uns_keys():
        logger.info("Log-normalizing data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    if "X_pca" not in adata.obsm_keys():
        logger.info(f"Computing PCA with {n_pcs} components")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=n_pcs)
    
    covariate_list = []
    
    if encode_sex and "sex" in adata.obs.columns:
        sex_codes = adata.obs.loc[adata.obs_names, "sex"].astype("category").cat.codes
        covariate_list.append(pd.DataFrame(sex_codes, columns=["sex"], index=adata.obs_names))
    
    if encode_age and "age" in adata.obs.columns:
        age_values = adata.obs.loc[adata.obs_names, "age"].values.astype(float)
        age_values = (age_values - age_values.mean()) / age_values.std()
        covariate_list.append(pd.DataFrame(age_values, columns=["age"], index=adata.obs_names))
    
    if additional_covariates:
        for cov in additional_covariates:
            if cov in adata.obs.columns:
                cov_data = adata.obs.loc[adata.obs_names, cov]
                covariate_list.append(pd.DataFrame(cov_data, columns=[cov], index=adata.obs_names))
    
    df_cov = pd.concat(covariate_list, axis=1) if covariate_list else None
    
    logger.info("Preprocessing data for scDRS")
    scdrs.preprocess(adata, cov=df_cov, n_mean_bin=n_mean_bin, n_var_bin=n_var_bin)
    
    if gs_file is not None:
        logger.info(f"Loading gene sets from {gs_file}")
        dict_gs = scdrs.util.load_gs(
            gs_file,
            src_species=src_species, 
            dst_species="human",
            to_intersect=adata.var_names,
        )
        if trait_name is not None:
            dict_gs = {trait_name: dict_gs[trait_name]}
    else:
        dict_gs = gene_sets
    
    logger.info(f"Computing scDRS scores for {len(dict_gs)} trait(s)")
    dict_df_score = {}
    
    for trait in dict_gs:
        logger.info(f"Processing {trait}")
        gene_list, gene_weights = dict_gs[trait]
        
        df_score = scdrs.score_cell(
            data=adata,
            gene_list=gene_list,
            gene_weight=gene_weights,
            ctrl_match_key=ctrl_match_key,
            n_ctrl=n_ctrl,
            weight_opt=weight_opt,
            return_ctrl_raw_score=flag_return_ctrl_raw_score,
            return_ctrl_norm_score=flag_return_ctrl_norm_score,
        )
        
        dict_df_score[trait] = df_score
        
        adata.obs[f"{trait}_norm_score"] = df_score["norm_score"]
        adata.obs[f"{trait}_pval"] = df_score["pval"]
        
        if save_results:
            df_score.to_csv(f"{prefix}.{trait}.full_score.gz", sep="\t", compression="gzip")
            df_score[["raw_score", "norm_score", "pval", "nlog10_pval", "zscore"]].to_csv(
                f"{prefix}.{trait}.score.gz", sep="\t", compression="gzip"
            )
    
    results_downstream = {}
    
    if group_analysis or corr_analysis or gene_analysis:
        if group_analysis:
            logger.info("Computing KNN graph for heterogeneity analysis")
            sc.pp.neighbors(adata, n_neighbors=knn_n_neighbors, n_pcs=knn_n_pcs)
        
        for trait in dict_df_score:
            trait_results = {}
            
            if group_analysis:
                logger.info(f"Performing group analysis for {trait}")
                group_results = scdrs.method.downstream_group_analysis(
                    adata=adata,
                    df_full_score=dict_df_score[trait],
                    group_cols=group_analysis,
                )
                
                for group_col in group_analysis:
                    df_group = group_results[group_col]
                    trait_results[f"group_{group_col}"] = df_group
                    
                    if save_results:
                        df_group.to_csv(
                            f"{prefix}.{trait}.scdrs_group.{group_col}",
                            sep="\t"
                        )
            
            if corr_analysis:
                logger.info(f"Performing correlation analysis for {trait}")
                df_corr = scdrs.method.downstream_corr_analysis(
                    adata=adata,
                    df_full_score=dict_df_score[trait],
                    var_cols=corr_analysis,
                )
                
                trait_results["cell_corr"] = df_corr
                
                if save_results:
                    df_corr.to_csv(
                        f"{prefix}.{trait}.scdrs_cell_corr",
                        sep="\t"
                    )

            if gene_analysis:
                logger.info(f"Performing gene analysis for {trait}")
                df_gene = scdrs.method.downstream_gene_analysis(
                    adata=adata,
                    df_full_score=dict_df_score[trait],
                )
                
                trait_results["gene"] = df_gene
                
                if save_results:
                    df_gene.to_csv(
                        f"{prefix}.{trait}.scdrs_gene",
                        sep="\t"
                    )
            
            results_downstream[trait] = trait_results
    
    if return_adata:
        return adata
    
    if results_downstream:
        return dict_df_score, results_downstream
    else:
        if len(dict_df_score) == 1:
            return list(dict_df_score.values())[0]
        return dict_df_score