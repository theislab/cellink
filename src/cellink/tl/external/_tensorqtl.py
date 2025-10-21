import glob
import gzip
import logging
import os
import pickle
import shutil
import importlib.util
import subprocess
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata.utils import asarray

from cellink._core import DonorData
from cellink.io import to_plink

logger = logging.getLogger(__name__)

def read_tensorqtl_results(
    prefix: str = None, 
    mode: str = None,
    cis_output: bool | str = None,
    interaction_df: bool | str = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[dict, pd.DataFrame]]:
    """
    Read TensorQTL result files.

    Parameters
    ----------
    prefix : str, optional
        File prefix used for generating intermediate input/output files. Required for most modes.
    mode : str
        Mode of TensorQTL run (e.g., "cis_nominal", "cis", "trans", "cis_susie").
    cis_output : str or bool, default=True
        - If a string, specifies the path to the cis_output file for modes like `cis_independent` or `cis_susie`.
        - If True, the function will automatically attempt to find the cis_output file in `file_paths`.
        - If False, the cis_output will not be read.
    interaction_df : str or bool, default=True
        - If a string, specifies the path to the interaction terms file for `cis_nominal` mode.
        - If True, the function will attempt to automatically find the interaction file in `file_paths`.
        - If False, the interaction file will not be read.

    Returns
    -------
    pd.DataFrame or tuple
        Parsed results depending on the mode.
    """

    if mode == "cis_nominal":
        cis_qtl_pairs = pd.concat(
            [pd.read_parquet(path) for path in glob.glob(f"{prefix}.cis_qtl_pairs.*.parquet")], axis=0
        )
        if cis_output is not None:
            cis_qtl_signif_pairs = pd.read_parquet(f"{prefix}.cis_qtl.signif_pairs.parquet")
        else:
            cis_qtl_signif_pairs = None
        if interaction_df is not None:
            cis_qtl_top_assoc = pd.read_csv(f"{prefix}.cis_qtl_top_assoc.txt.gz", sep="\t")
        else:
            cis_qtl_top_assoc = None
        results = (cis_qtl_pairs, cis_qtl_signif_pairs, cis_qtl_top_assoc)
    elif mode == "cis":
        results = pd.read_csv(f"{prefix}.cis_qtl.txt.gz", sep="\t")
    elif mode == "cis_independent":
        results = pd.read_csv(f"{prefix}.cis_independent_qtl.txt.gz", sep="\t")
    elif mode == "trans":
        results = pd.read_parquet(f"{prefix}.trans_qtl_pairs.parquet")
    elif mode == "cis_susie" or mode == "trans_susie":
        with open(f"{prefix}.SuSiE.pickle", "rb") as f:
            susie = pickle.load(f)
        susie_summary = pd.read_parquet(f"{prefix}.SuSiE_summary.parquet")
        results = (susie, susie_summary)
    return results

def run_tensorqtl(
    dd: DonorData,
    n_pcs: int = 50,
    mode: Literal["cis_nominal", "cis_independent", "cis", "trans", "cis_susie", "trans_susie"] = None,
    permutations: int = 10000,
    cis_output: str = None,
    interaction_df: str = None,
    susie_loci: str = None,
    window: int = 1000000,
    pval_threshold: float = 1e-5,
    logp: bool = False,
    maf_threshold: float = 0,
    maf_threshold_interaction: float = 0.05,
    dosages: bool = False,
    return_dense: bool = False,
    return_r2: bool = False,
    best_only: bool = False,
    output_text: bool = False,
    batch_size: int = 20000,
    chunk_size: int | str = None,
    disable_beta_approx: bool = False,
    warn_monomorphic: bool = True,
    max_effects: int = 10,
    fdr: float = 0.05,
    qvalue_lambda: float = None,
    seed: int = None,
    prefix: str = None,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: list[str] | None = None,
    dtype: str = "float32",
    run: bool = True,
    read_results: bool = True,
    save_cmd_file: bool = False,
    plink_export_kwargs: dict | None = {},
    remove_intermediate_files: bool = True,
    overwrite_covariates_export: bool = True,
    overwrite_phenotype_export: bool = True,
    overwrite_plink_export: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[dict, pd.DataFrame], str]:
    """
    Run cis- or trans-QTL mapping using TensorQTL on donor-level aggregated expression and genotype data.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing single-cell gene expression (`dd.C`) and donor-level genotype data (`dd.G`).
    mode : {'cis_nominal', 'cis_independent',
            'cis', 'trans', 'cis_susie', 'trans_susie'}, optional
        Type of QTL analysis to perform.
    prefix : str, optional
        File prefix used for generating intermediate input/output files. Required for most modes.
    cis_output : str, optional
        Path to output file for `cis_independent` and `cis_susie` modes.
    interaction_df : str, optional
        Path to interaction terms file required for `cis_nominal` mode.
    susie_loci : str, optional
        Path to SuSiE loci file required for `trans_susie` mode.
    permutations : int, default=10000
        Number of permutations used for empirical cis-QTL analysis.
    fdr : float, default=0.05
        False Discovery Rate threshold for significant hits in empirical cis-QTL mode.
    qvalue_lambda : float, optional
        Lambda parameter for q-value estimation in empirical mode.
    window : int, default=1000000
        Genomic window (in base pairs) around phenotype for filtering cis effects.
    pval_threshold : float, default=1e-5
        P-value threshold for reporting significant QTL associations.
    maf_threshold : float, default=0
        Minimum allele frequency threshold for variants in QTL analysis.
    maf_threshold_interaction : float, default=0.05
        MAF threshold for interaction terms in `cis_nominal` mode.
    best_only : bool, default=False
        If True, only report the best association per phenotype (only applies to some modes).
    batch_size : int, default=20000
        Number of phenotype-variant pairs processed per batch (important for trans modes).
    chunk_size : int or str, optional
        Size of variant chunks processed in cis modes. Can be string like "1M" or integer base pairs.
    max_effects : int, default=10
        Maximum number of independent signals to detect in SuSiE-based modes.
    seed : int, optional
        Random seed for reproducibility, especially for permutation testing.
    logp : bool, default=False
        If True, output -log10(p-values) instead of raw p-values.
    dosages : bool, default=False
        If True, use dosage data for association testing (if available).
    return_dense : bool, default=False
        If True, return dense matrix results (applies to trans-QTL mode).
    return_r2 : bool, default=False
        If True, include r² statistics in results.
    output_text : bool, default=False
        If True, also output results as text files.
    disable_beta_approx : bool, default=False
        If True, disables approximation of beta coefficients.
    warn_monomorphic : bool, default=True
        If True, warnings are issued for monomorphic variants.
    n_pcs : int, default=50
        Number of principal components to compute from single-cell expression data if PCA not already present.
    encode_sex : bool, default=True
        If True, includes donor sex as a covariate.
    encode_age : bool, default=True
        If True, includes donor age (z-normalized if needed) as a covariate.
    additional_covariates : list of str, optional
        Additional covariates from `dd.G.obs` or `dd.G.obsm` to include in the model.
    dtype : str, default="float32"
        Data type to cast covariates and matrices for QTL model input.
    run : bool, default=True
        If True, executes the TensorQTL command. If False, returns the constructed command as a string.
    read_results : bool, default=True
        If True, reads and returns the result files. If False, returns the paths to the output files.
    save_cmd_file : bool, default=False
        If True, saves the constructed TensorQTL command to a file instead of printing.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for `to_plink` function.
    remove_intermediate_files : bool, default=True
        If True, removes the intermediate files.
    overwrite_covariates_export : bool, default=True
        If True, overwrites the covariates export.
    overwrite_phenotype_export : bool, default=True
        If True, overwrites the phenotype export.
    overwrite_plink_export : bool, default=True
        If True, overwrites the plink export.

    Returns
    -------
    pd.DataFrame, tuple, str, or list[str]
        Depending on mode and read_results:
        - If run=True and read_results=True: returns pandas DataFrame(s) or tuple of results.
        - If run=True and read_results=False: returns list of output file paths.
        - If run=False: returns the constructed TensorQTL command as a string.

    Raises
    ------
    ImportError
        If required dependencies (`plink2`, `tensorqtl`) are not found in system path.

    ValueError
        If required parameters (`prefix`, `cis_output`, `susie_loci`) are not provided for the selected mode.
    """
    if run:
        if shutil.which("plink2") is None:
            raise ImportError("plink2 is required for `run_tensorqtl`. Please install it.")
        if importlib.util.find_spec("tensorqtl") is None:
            raise ImportError("tensorqtl is required for `run_tensorqtl`. Please install it.")

    args = {
        "permutations": permutations,
        "window": window,
        "pval_threshold": pval_threshold,
        "logp": logp,
        "maf_threshold": maf_threshold,
        "maf_threshold_interaction": maf_threshold_interaction,
        "dosages": dosages,
        "return_dense": return_dense,
        "return_r2": return_r2,
        "best_only": best_only,
        "output_text": output_text,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "susie_loci": susie_loci,
        "disable_beta_approx": disable_beta_approx,
        "warn_monomorphic": warn_monomorphic,
        "max_effects": max_effects,
        "fdr": fdr,
        "qvalue_lambda": qvalue_lambda,
        "seed": seed,
    }

    if mode == "cis_nominal" and prefix is None:
        raise ValueError("If mode cis_nominal, then a prefix must be given.")

    if "X_pca" not in dd.C.obsm:
        logger.info("Calculating PCA.")
        sc.pp.pca(dd.C, n_comps=n_pcs)

    dd.aggregate(key_added="PB", sync_var=True, verbose=True)

    if not os.path.isfile(f"{prefix}_phenotype.bed.gz") or overwrite_phenotype_export:
        phenotype_df = dd.G.obsm["PB"].T
        phenotype_df.index.name = "Geneid"
        phenotype_pos_df = dd.C.var[["chrom", "start", "end"]].rename(columns={"chrom": "chr"})
        phenotype_pos_df["Geneid"] = phenotype_pos_df.index

        phenotype_write_df = pd.concat([phenotype_pos_df, phenotype_df], axis=1)
        phenotype_write_df = phenotype_write_df.rename(columns={"chr": "#chr"})
        phenotype_write_df = phenotype_write_df.groupby("#chr", sort=False, group_keys=False).apply(
            lambda x: x.sort_values(["start", "end"])
        )
        with gzip.open(f"{prefix}_phenotype.bed.gz", "wt") as f:
            f.write("\t".join(phenotype_write_df.columns.tolist()) + "\n")
            phenotype_write_df.to_csv(f, sep="\t", header=False, index=False)

    if not os.path.isfile(f"{prefix}_donor_features.tsv") or overwrite_covariates_export:
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
                    covariate_df = pd.DataFrame(
                        dd.G.obs[[cov]].values.astype(dtype), columns=[cov], index=phenotype_df.columns
                    )
                    covariate_list.append(covariate_df)
                elif cov in dd.G.obsm:
                    cov_matrix = asarray(dd.G.obsm[cov]).astype(dtype)
                    if cov_matrix.ndim == 1:
                        covariate_list.append(pd.DataFrame(cov_matrix, columns=[cov], index=phenotype_df.columns))
                    else:
                        covariate_list.append(
                            pd.DataFrame(
                                cov_matrix,
                                columns=[f"{cov}_{i}" for i in range(cov_matrix.shape[1])],
                                index=phenotype_df.columns,
                            )
                        )
                else:
                    raise ValueError(f"Covariate '{cov}' not found in dd.G.obs or dd.G.obsm.")

        covariates_df = pd.concat(covariate_list, axis=1)

        covariates_df.index.name = "iid"
        covariates_df = covariates_df.T
        covariates_df.to_csv(f"{prefix}_donor_features.tsv", sep="\t")

    #variant_df = dd.G.var[["chrom", "pos"]]
    #variant_df["index"] = range(len(variant_df))

    ###

    geno = prefix
    covar = f"{prefix}_donor_features.tsv"
    pheno = f"{prefix}_phenotype.bed.gz"

    if not os.path.isfile(f"{prefix}.pgen") or overwrite_plink_export:
        to_plink(dd.G, prefix, **plink_export_kwargs)
        cmd_plink_conversion = f"plink2 --bfile {geno} --make-pgen --out {geno}"
        subprocess.run(cmd_plink_conversion, check=True, shell=True)

    ###

    cmd = f"python -m tensorqtl {geno} {pheno} {prefix} --covariates {covar} --mode {mode}"

    for key, value in args.items():
        if isinstance(value, bool) and value:
            cmd += f" --{key}"
        elif value is not None and not isinstance(value, bool):
            cmd += f" --{key} {value}"

    if mode == "cis_nominal":
        if interaction_df is not None:
            cmd += f" --interaction {interaction_df}"
        if cis_output is not None:
            cmd += f" --cis_output {cis_output}"
    elif mode == "cis_independent":
        if cis_output is None:
            raise ValueError("cis_output can't be None in mode 'cis_independent'. Please provide a valid path.")
        cmd += f" --cis_output {cis_output}"
    elif mode == "cis_susie":
        if cis_output is None:
            raise ValueError("cis_output can't be None in mode 'cis_susie'. Please provide a valid path.")
        cmd += f" --cis_output {cis_output}"
    elif mode == "trans_susie":
        if susie_loci is None:
            raise ValueError("susie_loci can't be None in mode 'trans_susie'. Please provide a valid path.")
        cmd += f" --susie_loci {susie_loci}"

    if run:
        subprocess.run(cmd, check=True, shell=True)
        
        if remove_intermediate_files:
            extensions = [".bim", ".fam", ".bed", ".pgen", ".psam", ".pvar", "_donor_features.tsv", "_phenotype.bed.gz"]
            for ext in extensions:
                filename = prefix + ext
                if os.path.isfile(filename):
                    os.remove(filename)
            
        if read_results:
            results = read_tensorqtl_results(prefix, mode, cis_output=cis_output, interaction_df=interaction_df)
        
        return results

    else:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(cmd + "\n")
        else:
            return cmd
