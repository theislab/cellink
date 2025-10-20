import glob
import gzip
import logging
import os
import shutil
import subprocess
from typing import Literal, Union

import pandas as pd
import scanpy as sc
from anndata.utils import asarray

from cellink._core import DonorData
from cellink.io import to_plink

logger = logging.getLogger(__name__)

def read_jaxqtl_results(
    prefix: str
) -> pd.DataFrame:
    """
    Read jaxQTL output TSV file.

    Parameters
    ----------
    prefix : str
        Prefix of the jaxQTL result file (.tsv.gz).

    Returns
    -------
    pd.DataFrame
        The parsed jaxQTL results.
    """
    results_path = glob.glob(f"{prefix}.*.tsv.gz")[0]
    results = pd.read_csv(results_path, delimiter="\t")
    
    return results

def run_jaxqtl(
    dd: DonorData,
    prefix: str = None,
    out: str = None,
    n_pcs: int = 50,
    add_covar: str | None = None,
    covar_test: str | None = None,
    rm_covar: str | None = None,
    model: Literal["gaussian", "poisson", "NB"] | None = "NB",
    mode: Literal["nominal", "cis", "cis_acat", "fitnull", "covar", "trans", "estimate_ld_only"] | None = "cis",
    ld_type: Literal["raw", "glm_wt", "no_glm_wt"] | None = None,
    platform: Literal["cpu", "gpu", "tpu"] | None = None,
    test_method: Literal["wald", "score"] | None = "score",
    window: int | None = 500000,
    nperm: int | None = 1000,
    max_iter: int | None = None,
    perm_seed: int | None = None,
    addpc: int | None = 2,
    prop_cutoff: float | None = None,
    express_percent: float | None = None,
    offset: str | None = None,
    indlist: str | None = None,
    cond_snp: str | None = None,
    robust: bool = False,
    rare_snp: bool = False,
    autosomal_only: bool = False,
    perm_pheno: bool = False,
    qvalue: bool = False,
    no_offset: bool = False,
    standardize: bool = True,
    statsmodel: bool = False,
    verbose: bool = False,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: list[str] | None = None,
    dtype: str = "float32",
    run: bool = True,
    read_results: bool = True,
    save_cmd_file: bool = False,
    plink_export_kwargs: dict | None = {},
    remove_intermediate_files: bool = True,
    overwrite_plink_export: bool = True,
) -> Union[pd.DataFrame, str]:
    """
    Run cis- or trans-eQTL mapping using jaxQTL on donor-level genotype and aggregated expression data.

    This function prepares input files from a `DonorData` object, builds a command to invoke the `jaxqtl` binary, and optionally executes it.
    Covariates such as age, sex, and additional user-specified variables are encoded and included in the model. Supports multiple modes
    including nominal testing, permutation-based cis-QTL mapping, trans-QTL mapping, and LD estimation.

    Parameters
    ----------
    dd : DonorData
        Object containing donor-level genotype (`dd.G`) and cell-level expression data (`dd.C`).
    prefix : str, optional
        Prefix for temporary and output files. If not provided, defaults to "jaxqtl_temp".
    out : str, optional
        Output file prefix for jaxQTL results.
    n_pcs : int, default=50
        Number of principal components to compute if not already present in `dd.C.obsm["X_pca"]`.
    add_covar : str, optional
        Path to file with additional covariates to include.
    covar_test : str, optional
        Covariate to test for inclusion in the model.
    rm_covar : str, optional
        Covariate to exclude from the model.
    model : {'gaussian', 'poisson', 'NB'}, default='NB'
        Statistical model used for QTL testing.
    mode : {'nominal', 'cis', 'cis_acat', 'fitnull', 'covar', 'trans', 'estimate_ld_only'}, default='cis'
        Analysis mode for jaxQTL.
    ld_type : {'raw', 'glm_wt', 'no_glm_wt'}, optional
        Type of linkage disequilibrium estimation to use.
    platform : {'cpu', 'gpu', 'tpu'}, optional
        Hardware backend for running jaxQTL.
    test_method : {'wald', 'score'}, default='score'
        Statistical test to use for association testing.
    window : int, optional
        Genomic window (in base pairs) for cis or trans testing. Default is 500,000.
    nperm : int, optional
        Number of permutations to perform for empirical FDR estimation.
    max_iter : int, optional
        Maximum number of iterations for model fitting.
    perm_seed : int, optional
        Seed for permutation reproducibility.
    addpc : int, optional
        Number of genotype PCs to include as covariates.
    prop_cutoff : float, optional
        Minimum proportion of cells required to include a gene.
    express_percent : float, optional
        Minimum percentage of donors expressing a gene for inclusion.
    offset : str, optional
        Path to offset vector for GLM models.
    indlist : str, optional
        File containing list of individual IDs to include.
    cond_snp : str, optional
        File with SNPs to condition on in the analysis.
    robust : bool, default=False
        If True, enables robust standard error estimation.
    rare_snp : bool, default=False
        If True, includes rare variants in the analysis.
    autosomal_only : bool, default=False
        If True, restricts analysis to autosomal chromosomes.
    perm_pheno : bool, default=False
        If True, permutes phenotypes instead of genotypes.
    qvalue : bool, default=False
        If True, calculates q-values for multiple testing correction.
    no_offset : bool, default=False
        If True, disables model offset.
    standardize : bool, default=True
        If True, standardizes phenotype and genotype data before analysis.
    statsmodel : bool, default=False
        If True, uses statsmodels GLM implementation.
    verbose : bool, default=False
        If True, prints detailed logging from jaxQTL.
    encode_sex : bool, default=True
        If True, adds sex as a categorical covariate using `dd.G.obs['sex']`.
    encode_age : bool, default=True
        If True, adds age as a numeric covariate from `dd.G.obs['age']`.
    additional_covariates : list of str, optional
        Additional covariates to extract from `dd.G.obs` or `dd.G.obsm` and include in the model.
    dtype : str, default='float32'
        Data type for numerical covariate matrices.
    run : bool, default=True
        If True, executes the jaxQTL command. If False, returns the constructed command as a string.
    read_results : bool, default=True
        If True, reads and returns the result files as a pandas DataFrame. If False, returns the path(s) to the output files.
    save_cmd_file : str, default=None
        If provided, saves the jaxQTL command to this file instead of printing it.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for `to_plink` function.
    remove_intermediate_files : bool, default=True
        If True, removes the intermediate files.
    overwrite_plink_export : bool, default=True
        If True, overwrites the plink export.

    Returns
    -------
    pd.DataFrame, str, or list[str]
        If run=True and read_results=True, returns a pandas DataFrame of QTL mapping results.
        If run=True and read_results=False, returns a list of output file paths.
        If run=False, returns the constructed jaxQTL command as a string.

    Raises
    ------
    ImportError
        If jaxqtl is not installed or not found in the system PATH.
    ValueError
        If required covariates are not found in the DonorData object.
    """
    if not prefix:
        prefix = "jaxqtl_temp"
    if not out:
        out = prefix

    if run and shutil.which("jaxqtl") is None:
        raise ImportError(
            "jaxqtl is required for `run_jaxqtl`. Please install it following the instructions on https://github.com/mancusolab/jaxqtl and ensure it is available in your system PATH."
        )

    if "X_pca" not in dd.C.obsm:
        logger.info("Calculating PCA.")
        sc.pp.pca(dd.C, n_comps=n_pcs)

    dd.aggregate(key_added="PB", sync_var=True, verbose=True)
    phenotype_df = dd.G.obsm["PB"].T
    phenotype_df.index.name = "Geneid"
    phenotype_pos_df = dd.C.var[["chrom", "start", "end"]].rename(columns={"chrom": "chr"})
    phenotype_pos_df["Geneid"] = phenotype_pos_df.index
    phenotype_write_df = pd.concat([phenotype_pos_df, phenotype_df], axis=1)
    with gzip.open(f"{prefix}_phenotype.bed.gz", "wt") as f:
        f.write("#" + "\t".join(phenotype_write_df.columns.tolist()) + "\n")
        phenotype_write_df.to_csv(f, sep="\t", header=False, index=False)

    covariate_list = []

    if encode_sex:
        sex_codes = dd.G.obs["sex"].astype("category").cat.codes
        covariate_list.append(pd.DataFrame(sex_codes.values, columns=["sex"], index=phenotype_df.columns))

    if encode_age:
        age_values = dd.G.obs[["age"]].values.astype("int")
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
    covariates_df.to_csv(f"{prefix}_donor_features.tsv", sep="\t")
    # genotype_df = pd.DataFrame(dd.G.X.T, index=dd.G.var.index, columns=dd.G.obs.index)
    
    if not os.path.isfile(f"{prefix}.bed") or overwrite_plink_export:
        to_plink(dd.G, prefix, **plink_export_kwargs)

    ###

    geno = prefix
    covar = f"{prefix}_donor_features.tsv"
    pheno = f"{prefix}_phenotype.bed.gz"

    cmd = ["jaxqtl", "--geno", geno, "--covar", covar, "--pheno", pheno]

    def add_opt(flag, value):
        if value is not None:
            cmd.extend([flag, str(value)])

    def add_flag(flag, enabled):
        if enabled:
            cmd.append(flag)

    add_opt("--add-covar", add_covar)
    add_opt("--covar-test", covar_test)
    add_opt("--rm-covar", rm_covar)
    add_opt("--model", model)
    add_opt("--offset", offset)
    add_opt("--indlist", indlist)
    add_opt("--mode", mode)
    add_opt("--ld-type", ld_type)
    add_opt("--platform", platform)
    add_opt("--test-method", test_method)
    add_opt("--window", window)
    add_opt("--nperm", nperm)
    add_opt("--max-iter", max_iter)
    add_opt("--perm-seed", perm_seed)
    add_opt("--addpc", addpc)
    add_opt("--prop-cutoff", prop_cutoff)
    add_opt("--express-percent", express_percent)
    add_opt("--cond-snp", cond_snp)
    add_opt("--out", out)

    add_flag("--robust", robust)
    add_flag("--rare-snp", rare_snp)
    add_flag("--autosomal-only", autosomal_only)
    add_flag("--perm-pheno", perm_pheno)
    add_flag("--qvalue", qvalue)
    add_flag("--no-offset", no_offset)
    add_flag("--standardize", standardize)
    add_flag("--statsmodel", statsmodel)
    add_flag("--verbose", verbose)

    if run:
        subprocess.run(" ".join(cmd), check=True, shell=True)

        if read_results:
            results = read_jaxqtl_results(prefix=prefix)

        if remove_intermediate_files:
            extensions = [".bim", ".fam", ".bed", "_donor_features.tsv", "_phenotype.bed.gz"]
            for ext in extensions:
                filename = prefix + ext
                if os.path.isfile(filename):
                    os.remove(filename)

        return results
    else:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(" ".join(cmd) + "\n")
        else:
            return " ".join(cmd)
