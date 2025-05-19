import torch
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Literal, List, Optional, Tuple
from anndata.utils import asarray
import subprocess
from cellink._core.data_fields import DAnn
from cellink._core import DonorData

import logging
logger = logging.getLogger(__name__)

def run_tensorqtl(
    dd: DonorData,
    n_pcs: int = 50,
    mode: Literal["cis_qtl_nominal", "cis_qtl_empirical", "trans_qtl"] = None,
    cis_seed: int = 123456,
    cis_fdr: float = 0.05,
    cis_qvalue_lambda: float = 0.85,
    trans_batch_size: int = 10000,
    trans_return_sparse: bool = True,
    trans_pval_threshold: float = 1e-5,
    trans_maf_threshold: float = 0.05,
    trans_window: int = 5000000,
    prefix: str = None,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: Optional[List[str]] = None,
    dtype: str = "float32"
):
    """
    Run cis- or trans-eQTL mapping using TensorQTL on aggregated donor-level expression data.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing single-cell gene expression data (`dd.C`) and donor-level genotype data (`dd.G`).
    n_pcs : int, default=50
        Number of PCA components to compute for cell-level data if not already present.
    mode : {'cis_qtl_nominal', 'cis_qtl_empirical', 'trans_qtl'}
        Type of QTL analysis to run: nominal cis-QTL mapping, empirical cis-QTL mapping with FDR correction, or trans-QTL mapping.
    cis_seed : int, default=123456
        Random seed for permutation-based empirical cis-QTL mapping.
    cis_fdr : float, default=0.05
        FDR threshold for significant hits in empirical cis-QTL mapping.
    cis_qvalue_lambda : float, default=0.85
        Lambda parameter for the q-value estimation in empirical cis-QTL mapping.
    trans_batch_size : int, default=10000
        Batch size used in trans-QTL mapping.
    trans_return_sparse : bool, default=True
        If True, return sparse results for trans-QTLs to reduce memory usage.
    trans_pval_threshold : float, default=1e-5
        p-value threshold for reporting trans-QTL associations.
    trans_maf_threshold : float, default=0.05
        Minimum allele frequency threshold for trans-QTL variants.
    trans_window : int, default=5000000
        Genomic window (in base pairs) around phenotype for filtering cis effects in trans-QTL mapping.
    prefix : str, optional
        File prefix for saving nominal cis-QTL results (required if `mode='cis_qtl_nominal'`).
    encode_sex : bool, default=True
        If True, include sex as a covariate in the model.
    encode_age : bool, default=True
        If True, include age (z-normalized if necessary) as a covariate in the model.
    additional_covariates : list of str, optional
        List of additional covariate names to include from `dd.G.obs` or `dd.G.obsm`.
    dtype : str, default='float32'
        Data type used when casting covariate matrices.

    Returns
    -------
    results : pd.DataFrame
        DataFrame of QTL mapping results, format depends on the selected mode.
    """

    try:
        from tensorqtl import cis, trans, post
    except ImportError:
        raise ImportError("tensorqtl is required for `run_tensorqtl`. Install with `pip install cellink[tensorqtl]`. Please also isntall rpy2 and R package qvalue.")
    
    if mode == "cis_qtl_nominal" and prefix is None:
        raise ValueError("If mode cis_qtl_nominal, then a prefix must be given.")

    if "X_pca" not in dd.C.obsm:
        logger.info("Calculating PCA.")
        sc.pp.pca(dd.C, n_comps=n_pcs)

    dd.aggregate(key_added="PB", sync_var=True, verbose=True)
    phenotype_df = dd.G.obsm["PB"].T
    phenotype_pos_df = dd.C.var[["start", "end", "chrom"]].rename(columns={"chrom": "chr"})
    
    covariate_list = []
    covariate_list.append(pd.DataFrame(np.ones((dd.shape[0], 1)), columns=["intercept"], index=phenotype_df.columns))

    if encode_sex:
        sex_codes = dd.G.obs["sex"].astype("category").cat.codes
        covariate_list.append(pd.DataFrame(sex_codes.values, columns=["sex"], index=phenotype_df.columns))

    if encode_age:
        age_values = dd.G.obs[["age"]].values.astype(dtype)
        mean = age_values.mean()
        std = age_values.std()
        tolerance = 1e-2
        already_z_normalized = np.isclose(mean, 0.0, atol=tolerance) and np.isclose(std, 1.0, atol=tolerance)
        if not already_z_normalized and std > 0:
            logger.info("Performing z-normalization of age.")
            age_values = (age_values - mean) / std
        covariate_list.append(pd.DataFrame(age_values, columns=["age"], index=phenotype_df.columns))

    if additional_covariates:
        for cov in additional_covariates:
            if cov in dd.G.obs.columns:
                covariate_df = pd.DataFrame(dd.G.obs[[cov]].values.astype(dtype), columns=[cov], index=phenotype_df.columns)
                covariate_list.append(covariate_df)
            elif cov in dd.G.obsm:
                cov_matrix = asarray(dd.G.obsm[cov]).astype(dtype)
                if cov_matrix.ndim == 1:
                    covariate_list.append(pd.DataFrame(cov_matrix, columns=[cov], index=phenotype_df.columns))
                else:
                    covariate_list.append(pd.DataFrame(cov_matrix, columns=[f"{cov}_{i}" for i in range(cov_matrix.shape[1])], index=phenotype_df.columns))
            else:
                raise ValueError(f"Covariate '{cov}' not found in dd.G.obs or dd.G.obsm.")
            
    covariates_df = pd.concat(covariate_list, axis=1)
    
    genotype_df = pd.DataFrame(dd.G.X.T, index=dd.G.var.index, columns=dd.G.obs.index)
    variant_df = dd.G.var[["chrom", "pos"]]#.rename(columns={"chrom": "chr"})
    variant_df["index"] = range(len(variant_df))

    if mode == "cis_qtl_nominal":
        cis.map_nominal(genotype_df, variant_df,
                    phenotype_df,
                    phenotype_pos_df, prefix=prefix, covariates_df=covariates_df)
        results_chr = []
        for chr in np.unique(variant_df["chrom"]):
            results_chr.append(pd.read_parquet(f'{prefix}.cis_qtl_pairs.{chr}.parquet')) 
        results = pd.concat(results_chr, axis=0)
    elif mode == "cis_qtl_empirical":
        results = cis.map_cis(genotype_df, variant_df, 
                            phenotype_df, phenotype_pos_df,
                            covariates_df=covariates_df, seed=cis_seed)
        post.calculate_qvalues(results, fdr=cis_fdr, qvalue_lambda=cis_qvalue_lambda)
    elif mode == "trans_qtl":
        results = trans.map_trans(genotype_df, phenotype_df, covariates_df, batch_size=trans_batch_size,
                        return_sparse=trans_return_sparse, pval_threshold=trans_pval_threshold, maf_threshold=trans_maf_threshold)
        results = trans.filter_cis(results, phenotype_pos_df, variant_df, window=trans_window)

    return results