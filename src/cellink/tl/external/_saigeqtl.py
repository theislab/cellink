import logging
import os
from typing import Literal

import pandas as pd
import scanpy as sc
import yaml
from anndata.utils import asarray

from cellink._core import DAnn, DonorData
from cellink.io import to_plink
from cellink.tl._runner import BaseToolRunner

logger = logging.getLogger(__name__)


class SAIGEQTLRunner(BaseToolRunner):
    """SAIGE-QTL Runner with support for local, docker, and singularity"""

    def __init__(self, config_path: str | None = None, config_dict: dict | None = None):
        required_fields = [
            "execution_mode",
            "rscript_path",
            "step1_script",
            "step2_script",
            "step3_script",
            "makegroup_script",
        ]
        prefix_tokens = []
        super().__init__(config_path, config_dict, required_fields, prefix_tokens)

    def _load_config(self, config_path: str | None, config_dict: dict | None) -> dict:
        if config_dict:
            return config_dict
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            "execution_mode": "local",
            "rscript_path": "Rscript",
            "docker_image": "wzhou88/saigeqtl",
            "singularity_image": "docker://wzhou88/saigeqtl",
            "step1_script": "step1_fitNULLGLMM_qtl.R",
            "step2_script": "step2_tests_qtl.R",
            "step3_script": "step3_gene_pvalue_qtl.R",
            "makegroup_script": "makeGroupFile.R",
        }

    @property
    def step1_script(self) -> str:
        return self.config["step1_script"]

    @property
    def step2_script(self) -> str:
        return self.config["step2_script"]

    @property
    def step3_script(self) -> str:
        return self.config["step3_script"]

    @property
    def makegroup_script(self) -> str:
        return self.config["makegroup_script"]

    @property
    def execution_mode(self) -> str:
        return self.config["execution_mode"]


_saigeqtl_runner = None


def configure_saigeqtl_runner(config_path: str | None = None, config_dict: dict | None = None) -> SAIGEQTLRunner:
    global _saigeqtl_runner
    _saigeqtl_runner = SAIGEQTLRunner(config_path=config_path, config_dict=config_dict)
    return _saigeqtl_runner


def get_saigeqtl_runner() -> SAIGEQTLRunner:
    global _saigeqtl_runner
    if _saigeqtl_runner is None:
        _saigeqtl_runner = SAIGEQTLRunner()
    return _saigeqtl_runner


def read_saigeqtl_results(prefix: str, step: Literal["step1", "step2", "step3"] = "step2") -> pd.DataFrame | dict:
    """Read SAIGE-QTL results"""
    if step == "step1":
        variance_ratio_df = pd.read_csv(f"{prefix}.varianceRatio.txt", sep=" ", header=None)
        return {"model_file": f"{prefix}.rda", "variance_ratio": variance_ratio_df}
    elif step == "step2":
        return pd.read_csv(f"{prefix}_results.txt", sep="\t")
    elif step == "step3":
        return pd.read_csv(f"{prefix}_gene_pvalue.txt", sep="\t")


def make_group_file(
    dd: DonorData = None,
    region_file: str = None,
    output_prefix: str = "group_file",
    allele_order: str = "alt-first",
    run: bool = True,
    save_cmd_file: str | None = None,
    plink_export_kwargs: dict | None = None,
    remove_intermediate_files: bool = True,
    runner: SAIGEQTLRunner | None = None,
) -> str:
    """
    Create group file for set-based rare variant tests.

    Parameters
    ----------
    dd : DonorData, optional
        DonorData object. If provided, will export to PLINK format.
    region_file : str
        Path to region file (chr, start, end; no header)
    output_prefix : str
        Output file prefix
    allele_order : str, default="alt-first"
        Allele order for PLINK/BGEN
    run : bool, default=True
        Whether to execute command
    save_cmd_file : str, optional
        Save command to file
    plink_export_kwargs : dict, optional
        Additional kwargs for to_plink
    remove_intermediate_files : bool, default=True
        Remove intermediate PLINK files
    runner : SAIGEQTLRunner, optional
        Runner instance

    Returns
    -------
    str
        Path to group file or command string
    """
    if runner is None:
        runner = get_saigeqtl_runner()
    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    if dd is not None:
        temp_prefix = f"{output_prefix}_temp"
        to_plink(dd.G, temp_prefix, **plink_export_kwargs)
        bed_file = f"{temp_prefix}.bed"
        bim_file = f"{temp_prefix}.bim"
        fam_file = f"{temp_prefix}.fam"

    cmd = f"{runner.config['rscript_path']} {runner.makegroup_script} --regionFile={region_file} --outputPrefix={output_prefix}"

    cmd += f" --bedFile={bed_file} --bimFile={bim_file} --famFile={fam_file}"
    if allele_order != "alt-first":
        cmd += f" --AlleleOrder={allele_order}"

    file_paths = [region_file]
    file_paths.extend([bed_file, bim_file, fam_file])

    if not run:
        cmd = runner._build_container_command(cmd, file_paths)
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(cmd + "\n")
        return cmd

    logger.info("Creating group file")
    runner.run_command(cmd, file_paths=file_paths)

    if dd is not None and remove_intermediate_files:
        for ext in [".bed", ".bim", ".fam"]:
            if os.path.isfile(temp_prefix + ext):
                os.remove(temp_prefix + ext)

    return f"{output_prefix}"


def run_saigeqtl(
    dd: DonorData,
    gene_col: str,
    prefix: str = "saigeqtl_temp",
    steps: list[int] = [1, 2],
    mode: Literal["cis", "genome_wide"] = "cis",
    analysis_type: Literal["single_variant", "set_based"] = "single_variant",
    n_pcs: int = 50,
    # Step 1 parameters
    use_sparse_grm_to_fit_null: bool = False,
    use_grm_to_fit_null: bool = True,
    skip_variance_ratio: bool = False,
    overwrite_variance_ratio_file: bool = False,
    tol: float = 0.02,
    maxiter: int = 20,
    tol_pcg: float = 1e-5,
    maxiter_pcg: int = 500,
    n_threads: int = 1,
    spa_cutoff: float = 2,
    num_random_marker_variance_ratio: int = 30,
    skip_model_fitting: bool = False,
    memory_chunk: float = 2,
    tau_init: str = "0,0",
    loco_step1: bool = True,
    is_low_mem_loco: bool = False,
    trace_cv_cutoff: float = 0.0025,
    nrun: int = 30,
    ratio_cv_cutoff: float = 0.001,
    output_prefix_var_ratio: str = "",
    sparse_grm_file: str = "",
    sparse_grm_sample_id_file: str = "",
    is_cate_variance_ratio: bool = False,
    relatedness_cutoff: float = 0,
    cate_var_ratio_min_mac_vec_exclude: str = "10,20.5",
    cate_var_ratio_max_mac_vec_include: str = "20.5",
    is_covariate_transform: bool = True,
    is_diag_of_kin_set_as_one: bool = False,
    use_sparse_grm_for_var_ratio: bool = False,
    min_maf_for_grm: float = 0.01,
    max_missing_rate_for_grm: float = 0.15,
    min_covariate_count: int = -1,
    include_nonauto_markers_for_var_ratio: bool = False,
    female_only: bool = False,
    male_only: bool = False,
    sex_col: str = "",
    female_code: str = "1",
    male_code: str = "0",
    is_covariate_offset: bool = True,
    sample_id_include_file: str = "",
    # Step 2 parameters
    allele_order: str = "alt-first",
    ids_to_include_file: str = "",
    ranges_to_include_file: str = "",
    window: int = 500000,
    chrom: str | None = None,
    is_imputed_data: bool = False,
    min_maf: float = 0,
    min_mac: float = 20,
    min_info: float = 0,
    max_missing: float = 0.15,
    impute_method: str = "best_guess",
    loco_step2: bool = False,
    gmmat_model_file: str = "",
    variance_ratio_file: str = "",
    saige_output_file: str = "",
    markers_per_chunk: int = 10000,
    is_output_more_details: bool = False,
    is_overwrite_output: bool = True,
    # Set-based test parameters
    group_file: str | None = None,
    max_maf_in_group_test: str = "0.0001,0.001,0.01",
    min_maf_in_group_test_exclude: str | None = None,
    max_mac_in_group_test: str = "0",
    min_mac_in_group_test_exclude: str | None = None,
    annotation_in_group_test: str = "lof,missense;lof,missense;lof;synonymous",
    mac_cutoff_to_collapse_ultra_rare: int = 10,
    min_group_mac_in_burden_test: float = 5,
    weights_beta: str = "1;25",
    r_corr: float = 0,
    markers_per_chunk_in_group_test: int = 100,
    groups_per_chunk: int = 100,
    condition: str = "",
    weights_for_condition: str | None = None,
    dosage_zerod_cutoff: float = 0.2,
    dosage_zerod_mac_cutoff: int = 10,
    is_single_in_group_test: bool = False,
    is_skato: bool = False,
    is_equal_weight_in_group_test: bool = False,
    is_output_marker_list_in_group_test: bool = False,
    is_firth_beta: bool = False,
    p_cutoff_for_firth: float = 0.01,
    is_fast_test: bool = False,
    is_noadj_cov: bool = True,
    is_sparse_grm: bool = False,
    pval_cutoff_for_fast_test: float = 0.05,
    max_mac_for_er: int = 4,
    is_emp_spa: bool = False,
    # Step 3 parameters
    weight_file: str = "",
    gene_name: str = "",
    # Covariate parameters
    sample_covariates: list[str] | None = None,
    cell_covariates: list[str] | None = None,
    encode_sex: bool = True,
    encode_age: bool = True,
    dtype: str = "float32",
    # Execution parameters
    run: bool = True,
    read_results: bool = True,
    save_cmd_file: str | None = None,
    plink_export_kwargs: dict | None = None,
    remove_intermediate_files: bool = True,
    runner: SAIGEQTLRunner | None = None,
) -> pd.DataFrame | dict | str:
    """
    Run eQTL mapping using SAIGE-QTL with flexible step execution.

    This function can run Step 1 (null model fitting), Step 2 (association testing),
    and/or Step 3 (gene-level p-values) independently or in combination.

    Parameters
    ----------
    dd : DonorData
        DonorData object with genotype and expression data
    gene_col : str
        Gene/phenotype column name
    prefix : str, default="saigeqtl_temp"
        Output file prefix
    steps : list[int], default=[1, 2]
        Which steps to run: [1], [2], [3], [1,2], [1,2,3], [2,3], etc.
    mode : {'cis', 'genome_wide'}, default='cis'
        Analysis mode
    analysis_type : {'single_variant', 'set_based'}, default='single_variant'
        Type of analysis in Step 2
    n_pcs : int, default=50
        Number of PCs to compute

    Step 1 Parameters (Null Model Fitting)
    ---------------------------------------
    use_sparse_grm_to_fit_null : bool, default=False
        Use sparse GRM for null model
    use_grm_to_fit_null : bool, default=True
        Use full GRM for null model
    skip_variance_ratio : bool, default=False
        Skip variance ratio estimation
    overwrite_variance_ratio_file : bool, default=False
        Overwrite existing variance ratio file
    tol : float, default=0.02
        Convergence tolerance
    maxiter : int, default=20
        Maximum iterations
    tol_pcg : float, default=1e-5
        PCG tolerance
    maxiter_pcg : int, default=500
        Maximum PCG iterations
    n_threads : int, default=1
        Number of threads
    spa_cutoff : float, default=2
        SPA cutoff in SD units
    num_random_marker_variance_ratio : int, default=30
        Number of random markers for variance ratio
    skip_model_fitting : bool, default=False
        Skip model fitting (requires existing .rda)
    memory_chunk : float, default=2
        Memory chunk size in Gb
    tau_init : str, default="0,0"
        Initial tau values
    loco_step1 : bool, default=True
        Use LOCO in Step 1
    is_low_mem_loco : bool, default=False
        Low memory LOCO mode
    trace_cv_cutoff : float, default=0.0025
        Trace CV cutoff
    nrun : int, default=30
        Number of trace estimation runs
    ratio_cv_cutoff : float, default=0.001
        Ratio CV cutoff
    output_prefix_var_ratio : str, default=""
        Variance ratio file prefix
    sparse_grm_file : str, default=""
        Pre-calculated sparse GRM file
    sparse_grm_sample_id_file : str, default=""
        Sample IDs for sparse GRM
    is_cate_variance_ratio : bool, default=False
        Estimate categorical variance ratios
    relatedness_cutoff : float, default=0
        Kinship threshold for sparse GRM
    cate_var_ratio_min_mac_vec_exclude : str, default="10,20.5"
        Lower MAC bounds for categories
    cate_var_ratio_max_mac_vec_include : str, default="20.5"
        Upper MAC bounds for categories
    is_covariate_transform : bool, default=True
        Apply QR transformation to covariates
    is_diag_of_kin_set_as_one : bool, default=False
        Set GRM diagonal to 1
    use_sparse_grm_for_var_ratio : bool, default=False
        Use sparse GRM for variance ratio
    min_maf_for_grm : float, default=0.01
        Minimum MAF for GRM markers
    max_missing_rate_for_grm : float, default=0.15
        Maximum missing rate for GRM markers
    min_covariate_count : int, default=-1
        Minimum count for binary covariates
    include_nonauto_markers_for_var_ratio : bool, default=False
        Include non-autosomal markers
    female_only : bool, default=False
        Run for females only
    male_only : bool, default=False
        Run for males only
    sex_col : str, default=""
        Sex column name
    female_code : str, default="1"
        Female code in sex column
    male_code : str, default="0"
        Male code in sex column
    is_covariate_offset : bool, default=True
        Estimate fixed effect coefficients
    sample_id_include_file : str, default=""
        File with sample IDs to include

    Step 2 Parameters (Association Testing)
    ---------------------------------------

    allele_order : str, default="alt-first"
        Allele order for PLINK/BGEN
    ids_to_include_file : str, default=""
        File with variant IDs to include
    ranges_to_include_file : str, default=""
        File with genomic ranges
    window : int, default=500000
        Cis-window size in bp
    chrom : str, optional
        Chromosome
    is_imputed_data : bool, default=False
        Whether data is imputed
    min_maf : float, default=0
        Minimum MAF
    min_mac : float, default=20
        Minimum MAC
    min_info : float, default=0
        Minimum imputation INFO
    max_missing : float, default=0.15
        Maximum missing rate
    impute_method : str, default="best_guess"
        Imputation method
    loco_step2 : bool, default=False
        Use LOCO in Step 2
    gmmat_model_file : str, default=""
        Pre-computed model file
    variance_ratio_file : str, default=""
        Pre-computed variance ratio file
    saige_output_file : str, default=""
        Custom output file name
    markers_per_chunk : int, default=10000
        Markers per chunk
    is_output_more_details : bool, default=False
        Output additional details
    is_overwrite_output : bool, default=True
        Overwrite existing output

    Set-Based Test Parameters
    -------------------------
    group_file : str, optional
        Group file for set-based tests
    max_maf_in_group_test : str, default="0.0001,0.001,0.01"
        Maximum MAF for masks
    min_maf_in_group_test_exclude : str, optional
        Minimum MAF to exclude
    max_mac_in_group_test : str, default="0"
        Maximum MAC for masks
    min_mac_in_group_test_exclude : str, optional
        Minimum MAC to exclude
    annotation_in_group_test : str, default="lof,missense;lof,missense;lof;synonymous"
        Annotation categories
    mac_cutoff_to_collapse_ultra_rare : int, default=10
        MAC cutoff for collapsing
    min_group_mac_in_burden_test : float, default=5
        Minimum MAC for burden test
    weights_beta : str, default="1;25"
        Beta distribution parameters
    r_corr : float, default=0
        Correlation for SKAT-O (0=SKAT-O, 1=burden)
    markers_per_chunk_in_group_test : int, default=100
        Markers per chunk in group test
    groups_per_chunk : int, default=100
        Groups per chunk
    condition : str, default=""
        Conditioning markers
    weights_for_condition : str, optional
        Weights for conditioning
    dosage_zerod_cutoff : float, default=0.2
        Dosage zeroing cutoff
    dosage_zerod_mac_cutoff : int, default=10
        MAC cutoff for dosage zeroing
    is_single_in_group_test : bool, default=False
        Output single-variant results
    is_skato : bool, default=False
        Perform SKAT-O test
    is_equal_weight_in_group_test : bool, default=False
        Use equal weights
    is_output_marker_list_in_group_test : bool, default=False
        Output marker lists
    is_firth_beta : bool, default=False
        Use Firth correction
    p_cutoff_for_firth : float, default=0.01
        P-value cutoff for Firth
    is_fast_test : bool, default=False
        Use fast test mode
    is_noadj_cov : bool, default=True
        Don't adjust for covariates in genotypes
    is_sparse_grm : bool, default=False
        Use sparse GRM in Step 2
    pval_cutoff_for_fast_test : float, default=0.05
        P-value cutoff for fast test
    max_mac_for_er : int, default=4
        Maximum MAC for efficient resampling
    is_emp_spa : bool, default=False
        Use empirical SPA

    Step 3 Parameters (Gene-Level P-values)
    ---------------------------------------
    weight_file : str, default=""
        File with variant weights for ACAT
    gene_name : str, default=""
        Gene name for output

    Covariate Parameters
    -------------------
    sample_covariates : list[str], optional
        Individual-level covariates
    cell_covariates : list[str], optional
        Cell-level covariates
    encode_sex : bool, default=True
        Include sex covariate
    encode_age : bool, default=True
        Include age covariate
    dtype : str, default="float32"
        Data type for covariates

    Execution Parameters
    -------------------
    run : bool, default=True
        Execute commands
    read_results : bool, default=True
        Read and return results
    save_cmd_file : str, optional
        Save commands to file
    plink_export_kwargs : dict, optional
        Additional kwargs for to_plink
    remove_intermediate_files : bool, default=True
        Remove intermediate files
    runner : SAIGEQTLRunner, optional
        Runner instance

    Returns
    -------
    pd.DataFrame, dict, or str
        Results or commands depending on execution mode

    Examples
    --------
    # Run only Step 1
    >>> run_saigeqtl(dd, gene_col="GENE1", steps=[1], prefix="step1_only")

    # Run Steps 1 and 2
    >>> run_saigeqtl(dd, gene_col="GENE1", steps=[1, 2], prefix="full_analysis")

    # Run only Step 3 (requires existing Step 2 results)
    >>> run_saigeqtl(dd, gene_col="GENE1", steps=[3], prefix="existing_results")

    # Run all steps including gene-level p-values
    >>> run_saigeqtl(dd, gene_col="GENE1", steps=[1, 2, 3], prefix="complete")
    """
    if runner is None:
        runner = get_saigeqtl_runner()
    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    if steps == "all":
        steps = [1, 2, 3]

    if 1 in steps or 2 in steps:
        if "X_pca" not in dd.C.obsm:
            logger.info("Calculating PCA")
            sc.pp.pca(dd.C, n_comps=n_pcs)

        pheno_df = pd.DataFrame(dd.C.X.todense(), columns=dd.C.var.index, index=dd.C.obs[DAnn.donor])
        pheno_df.insert(0, "IND_ID", pheno_df.index)

        if gene_col not in pheno_df.columns:
            raise ValueError(f"Gene '{gene_col}' not found. Available: {pheno_df.columns[:10].tolist()}...")

        sample_cov_list = []
        sample_cov_names = []

        if encode_sex and "sex" in dd.G.obs.columns:
            sex_data = dd.G.obs["sex"].astype("category").cat.codes
            sample_cov_list.append(pd.DataFrame({"sex": sex_data.values}, index=dd.G.obs.index))
            sample_cov_names.append("sex")

        if encode_age and "age" in dd.G.obs.columns:
            age_data = dd.G.obs["age"].values.astype(dtype)
            sample_cov_list.append(pd.DataFrame({"age": age_data}, index=dd.G.obs.index))
            sample_cov_names.append("age")

        if sample_covariates:
            for cov in sample_covariates:
                if cov in dd.G.obs.columns:
                    cov_data = dd.G.obs[[cov]].astype(dtype)
                    sample_cov_list.append(cov_data)
                    sample_cov_names.append(cov)
                elif cov in dd.G.obsm:
                    cov_matrix = asarray(dd.G.obsm[cov]).astype(dtype)
                    if cov_matrix.ndim == 1:
                        sample_cov_list.append(pd.DataFrame({cov: cov_matrix}, index=dd.G.obs.index))
                        sample_cov_names.append(cov)
                    else:
                        cov_df = pd.DataFrame(
                            cov_matrix, columns=[f"{cov}_{i}" for i in range(cov_matrix.shape[1])], index=dd.G.obs.index
                        )
                        sample_cov_list.append(cov_df)
                        sample_cov_names.extend([f"{cov}_{i}" for i in range(cov_matrix.shape[1])])
                else:
                    raise ValueError(f"Sample covariate '{cov}' not found")

        if sample_cov_list:
            sample_cov_df = pd.concat(sample_cov_list, axis=1)
            donor_to_sample_cov = sample_cov_df.loc[pheno_df["IND_ID"]].reset_index(drop=True)
            donor_to_sample_cov.index = pheno_df.index

        cell_cov_list = []
        cell_cov_names = []

        if cell_covariates:
            for cov in cell_covariates:
                if cov in dd.C.obs.columns:
                    cov_data = dd.C.obs[[cov, DAnn.donor]]
                    cov_data.set_index(DAnn.donor, inplace=True)
                    cov_data = cov_data.astype(dtype)
                    cell_cov_list.append(cov_data)
                    cell_cov_names.append(cov)
                elif cov in dd.C.obsm:
                    cov_matrix = asarray(dd.C.obsm[cov]).astype(dtype)
                    if cov_matrix.ndim == 1:
                        cell_cov_list.append(pd.DataFrame({cov: cov_matrix}, index=dd.C.obs[DAnn.donor]))
                        cell_cov_names.append(cov)
                    else:
                        cov_df = pd.DataFrame(
                            cov_matrix,
                            columns=[f"{cov}_{i}" for i in range(cov_matrix.shape[1])],
                            index=dd.C.obs[DAnn.donor],
                        )
                        cell_cov_list.append(cov_df)
                        cell_cov_names.extend([f"{cov}_{i}" for i in range(cov_matrix.shape[1])])
                else:
                    raise ValueError(f"Cell covariate '{cov}' not found")

        if cell_cov_list:
            cell_cov_df = pd.concat(cell_cov_list, axis=1)

        if sample_cov_list and cell_cov_list:
            all_cov_df = pd.concat([donor_to_sample_cov, cell_cov_df], axis=1)
            pheno_df = pd.concat([pheno_df, all_cov_df], axis=1)
        elif sample_cov_list:
            pheno_df = pd.concat([pheno_df, donor_to_sample_cov], axis=1)
        elif cell_cov_list:
            pheno_df = pd.concat([pheno_df, cell_cov_df], axis=1)

        pheno_df.to_csv(f"{prefix}_phenotype.txt", sep="\t", index=False)

        all_cov_names = sample_cov_names + cell_cov_names
        covar_cols = ",".join(all_cov_names) if all_cov_names else ""
        sample_covar_cols = ",".join(sample_cov_names) if sample_cov_names else ""

        to_plink(dd.G, prefix, **plink_export_kwargs)
        fam = pd.read_csv(f"{prefix}.fam", delimiter="\t", header=None)
        fam.to_csv(f"{prefix}.fam", sep=" ", header=False, index=False)
        bim = pd.read_csv(f"{prefix}.bim", delimiter="\t", header=None)
        bim.to_csv(f"{prefix}.bim", sep=" ", header=False, index=False)

    commands = {}

    if 1 in steps:
        step1_cmd = f"{runner.config['rscript_path']} {runner.step1_script}"
        step1_cmd += f" --phenoFile={prefix}_phenotype.txt"
        step1_cmd += f" --phenoCol={gene_col}"
        step1_cmd += " --sampleIDColinphenoFile=IND_ID"
        step1_cmd += " --traitType=count"
        step1_cmd += f" --plinkFile={prefix}"
        step1_cmd += f" --outputPrefix={prefix}"

        if tol != 0.02:
            step1_cmd += f" --tol={tol}"
        if maxiter != 20:
            step1_cmd += f" --maxiter={maxiter}"
        if tol_pcg != 1e-5:
            step1_cmd += f" --tolPCG={tol_pcg}"
        if maxiter_pcg != 500:
            step1_cmd += f" --maxiterPCG={maxiter_pcg}"
        if n_threads != 1:
            step1_cmd += f" --nThreads={n_threads}"
        if spa_cutoff != 2:
            step1_cmd += f" --SPAcutoff={spa_cutoff}"
        if num_random_marker_variance_ratio != 30:
            step1_cmd += f" --numRandomMarkerforVarianceRatio={num_random_marker_variance_ratio}"
        if memory_chunk != 2:
            step1_cmd += f" --memoryChunk={memory_chunk}"
        if tau_init != "0,0":
            step1_cmd += f" --tauInit={tau_init}"
        if not loco_step1:
            step1_cmd += " --LOCO=FALSE"
        if trace_cv_cutoff != 0.0025:
            step1_cmd += f" --traceCVcutoff={trace_cv_cutoff}"
        if nrun != 30:
            step1_cmd += f" --nrun={nrun}"
        if ratio_cv_cutoff != 0.001:
            step1_cmd += f" --ratioCVcutoff={ratio_cv_cutoff}"
        if relatedness_cutoff != 0:
            step1_cmd += f" --relatednessCutoff={relatedness_cutoff}"
        if cate_var_ratio_min_mac_vec_exclude != "10,20.5":
            step1_cmd += f" --cateVarRatioMinMACVecExclude={cate_var_ratio_min_mac_vec_exclude}"
        if cate_var_ratio_max_mac_vec_include != "20.5":
            step1_cmd += f" --cateVarRatioMaxMACVecInclude={cate_var_ratio_max_mac_vec_include}"
        if not is_covariate_transform:
            step1_cmd += " --isCovariateTransform=FALSE"
        if min_maf_for_grm != 0.01:
            step1_cmd += f" --minMAFforGRM={min_maf_for_grm}"
        if max_missing_rate_for_grm != 0.15:
            step1_cmd += f" --maxMissingRateforGRM={max_missing_rate_for_grm}"
        if min_covariate_count != -1:
            step1_cmd += f" --minCovariateCount={min_covariate_count}"
        if not is_covariate_offset:
            step1_cmd += " --isCovariateOffset=FALSE"

        if use_sparse_grm_to_fit_null:
            step1_cmd += " --useSparseGRMtoFitNULL=TRUE"
        if not use_grm_to_fit_null:
            step1_cmd += " --useGRMtoFitNULL=FALSE"
        if skip_variance_ratio:
            step1_cmd += " --skipVarianceRatioEstimation=TRUE"
        if skip_model_fitting:
            step1_cmd += " --skipModelFitting=TRUE"
        if overwrite_variance_ratio_file:
            step1_cmd += " --IsOverwriteVarianceRatioFile=TRUE"
        if is_low_mem_loco:
            step1_cmd += " --isLowMemLOCO=TRUE"
        if is_cate_variance_ratio:
            step1_cmd += " --isCateVarianceRatio=TRUE"
        if is_diag_of_kin_set_as_one:
            step1_cmd += " --isDiagofKinSetAsOne=TRUE"
        if use_sparse_grm_for_var_ratio:
            step1_cmd += " --useSparseGRMforVarRatio=TRUE"
        if include_nonauto_markers_for_var_ratio:
            step1_cmd += " --includeNonautoMarkersforVarRatio=TRUE"
        if female_only:
            step1_cmd += " --FemaleOnly=TRUE"
        if male_only:
            step1_cmd += " --MaleOnly=TRUE"

        if covar_cols:
            step1_cmd += f" --covarColList={covar_cols}"
        if sample_covar_cols:
            step1_cmd += f" --sampleCovarColList={sample_covar_cols}"
        if output_prefix_var_ratio:
            step1_cmd += f" --outputPrefix_varRatio={output_prefix_var_ratio}"
        if sparse_grm_file:
            step1_cmd += f" --sparseGRMFile={sparse_grm_file}"
        if sparse_grm_sample_id_file:
            step1_cmd += f" --sparseGRMSampleIDFile={sparse_grm_sample_id_file}"
        if sex_col:
            step1_cmd += f" --sexCol={sex_col}"
        if female_code != "1":
            step1_cmd += f" --FemaleCode={female_code}"
        if male_code != "0":
            step1_cmd += f" --MaleCode={male_code}"
        if sample_id_include_file:
            step1_cmd += f" --SampleIDIncludeFile={sample_id_include_file}"

        commands["step1"] = step1_cmd

    if 2 in steps:
        if not gmmat_model_file:
            gmmat_model_file = f"{prefix}.rda"
        if not variance_ratio_file:
            variance_ratio_file = f"{prefix}.varianceRatio.txt"
        if not saige_output_file:
            saige_output_file = f"{prefix}_results.txt"

        step2_cmd = f"{runner.config['rscript_path']} {runner.step2_script}"

        step2_cmd += f" --bedFile={prefix}.bed"
        step2_cmd += f" --bimFile={prefix}.bim"
        step2_cmd += f" --famFile={prefix}.fam"

        step2_cmd += f" --GMMATmodelFile={gmmat_model_file}"
        step2_cmd += f" --varianceRatioFile={variance_ratio_file}"
        step2_cmd += f" --SAIGEOutputFile={saige_output_file}"

        if min_maf != 0:
            step2_cmd += f" --minMAF={min_maf}"
        if min_mac != 0.5:
            step2_cmd += f" --minMAC={min_mac}"
        if min_info != 0:
            step2_cmd += f" --minInfo={min_info}"
        if max_missing != 0.15:
            step2_cmd += f" --maxMissing={max_missing}"
        if impute_method != "best_guess":
            step2_cmd += f" --impute_method={impute_method}"
        if not loco_step2:
            step2_cmd += " --LOCO=FALSE"
        if spa_cutoff != 2:
            step2_cmd += f" --SPAcutoff={spa_cutoff}"
        if markers_per_chunk != 10000:
            step2_cmd += f" --markers_per_chunk={markers_per_chunk}"
        if not is_overwrite_output:
            step2_cmd += " --is_overwrite_output=FALSE"
        if allele_order != "alt-first":
            step2_cmd += f" --AlleleOrder={allele_order}"
        if dosage_zerod_cutoff != 0.2:
            step2_cmd += f" --dosage_zerod_cutoff={dosage_zerod_cutoff}"
        if dosage_zerod_mac_cutoff != 10:
            step2_cmd += f" --dosage_zerod_MAC_cutoff={dosage_zerod_mac_cutoff}"
        if p_cutoff_for_firth != 0.01:
            step2_cmd += f" --pCutoffforFirth={p_cutoff_for_firth}"
        if not is_noadj_cov:
            step2_cmd += " --is_noadjCov=FALSE"
        if pval_cutoff_for_fast_test != 0.05:
            step2_cmd += f" --pval_cutoff_for_fastTest={pval_cutoff_for_fast_test}"
        if max_mac_for_er != 4:
            step2_cmd += f" --max_MAC_for_ER={max_mac_for_er}"

        if is_imputed_data:
            step2_cmd += " --is_imputed_data=TRUE"
        if is_output_more_details:
            step2_cmd += " --is_output_moreDetails=TRUE"
        if is_firth_beta:
            step2_cmd += " --is_Firth_beta=TRUE"
        if is_fast_test:
            step2_cmd += " --is_fastTest=TRUE"
        if is_sparse_grm:
            step2_cmd += " --is_sparseGRM=TRUE"
        if is_emp_spa:
            step2_cmd += " --is_EmpSPA=TRUE"

        if chrom:
            step2_cmd += f" --chrom={chrom}"
        if ids_to_include_file:
            step2_cmd += f" --idstoIncludeFile={ids_to_include_file}"
        if condition:
            step2_cmd += f" --condition={condition}"

        if mode == "cis" and not ranges_to_include_file:
            ranges_file = f"{prefix}_ranges.txt"
            ranges_df = dd.C.var[["chrom", "start", "end"]].copy()
            ranges_df["start"] = ranges_df["start"] - window
            ranges_df["end"] = ranges_df["end"] + window
            ranges_df.to_csv(ranges_file, sep="\t", header=False, index=False)
            step2_cmd += f" --rangestoIncludeFile={ranges_file}"
        elif ranges_to_include_file:
            step2_cmd += f" --rangestoIncludeFile={ranges_to_include_file}"

        if analysis_type == "set_based" and group_file:
            step2_cmd += f" --groupFile={group_file}"
            if max_maf_in_group_test != "0.0001,0.001,0.01":
                step2_cmd += f" --maxMAF_in_groupTest={max_maf_in_group_test}"
            if min_maf_in_group_test_exclude:
                step2_cmd += f" --minMAF_in_groupTest_Exclude={min_maf_in_group_test_exclude}"
            if max_mac_in_group_test != "0":
                step2_cmd += f" --maxMAC_in_groupTest={max_mac_in_group_test}"
            if min_mac_in_group_test_exclude:
                step2_cmd += f" --minMAC_in_groupTest_Exclude={min_mac_in_group_test_exclude}"
            if annotation_in_group_test != "lof,missense;lof,missense;lof;synonymous":
                step2_cmd += f" --annotation_in_groupTest={annotation_in_group_test}"
            if mac_cutoff_to_collapse_ultra_rare != 10:
                step2_cmd += f" --MACCutoff_to_CollapseUltraRare={mac_cutoff_to_collapse_ultra_rare}"
            if min_group_mac_in_burden_test != 5:
                step2_cmd += f" --minGroupMAC_in_BurdenTest={min_group_mac_in_burden_test}"
            if weights_beta != "1;25":
                step2_cmd += f" --weights.beta={weights_beta}"
            if r_corr != 0:
                step2_cmd += f" --r.corr={r_corr}"
            if markers_per_chunk_in_group_test != 100:
                step2_cmd += f" --markers_per_chunk_in_groupTest={markers_per_chunk_in_group_test}"
            if groups_per_chunk != 100:
                step2_cmd += f" --groups_per_chunk={groups_per_chunk}"
            if weights_for_condition:
                step2_cmd += f" --weights_for_condition={weights_for_condition}"
            if is_single_in_group_test:
                step2_cmd += " --is_single_in_groupTest=TRUE"
            if is_skato:
                step2_cmd += " --is_SKATO=TRUE"
            if is_equal_weight_in_group_test:
                step2_cmd += " --is_equal_weight_in_groupTest=TRUE"
            if is_output_marker_list_in_group_test:
                step2_cmd += " --is_output_markerList_in_groupTest=TRUE"

        commands["step2"] = step2_cmd

    if 3 in steps:
        assoc_file = saige_output_file if saige_output_file else f"{prefix}_results.txt"
        gene_pval_output = f"{prefix}_gene_pvalue.txt"

        step3_cmd = f"{runner.config['rscript_path']} {runner.step3_script}"
        step3_cmd += f" --assocFile={assoc_file}"
        step3_cmd += f" --genePval_outputFile={gene_pval_output}"

        if gene_name:
            step3_cmd += f" --geneName={gene_name}"
        else:
            step3_cmd += f" --geneName={gene_col}"

        if weight_file:
            step3_cmd += f" --weightFile={weight_file}"

        commands["step3"] = step3_cmd

    file_paths = [f"{prefix}.bed", f"{prefix}_phenotype.txt"] if (1 in steps or 2 in steps) else []

    if not run:
        for step_num in sorted(steps):
            step_key = f"step{step_num}"
            if step_key in commands:
                commands[step_key] = runner._build_container_command(commands[step_key], file_paths)
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                for step_num in sorted(steps):
                    step_key = f"step{step_num}"
                    if step_key in commands:
                        f.write(f"# Step {step_num}\n{commands[step_key]}\n\n")
        return commands

    if 1 in steps:
        logger.info("Running Step 1: Fitting null model")
        runner.run_command(commands["step1"], file_paths=file_paths)

    if 2 in steps:
        logger.info("Running Step 2: Association tests")
        runner.run_command(commands["step2"], file_paths=file_paths)

    if 3 in steps:
        logger.info("Running Step 3: Gene-level p-values")
        runner.run_command(commands["step3"], file_paths=file_paths)

    if read_results:
        if 3 in steps:
            results = read_saigeqtl_results(prefix, "step3")
        elif 2 in steps:
            results = read_saigeqtl_results(prefix, "step2")
        elif 1 in steps:
            results = read_saigeqtl_results(prefix, "step1")
    else:
        results = {"commands": commands}

    if remove_intermediate_files and (1 in steps or 2 in steps):
        for ext in [".bed", ".bim", ".fam", "_phenotype.txt", "_ranges.txt"]:
            if os.path.isfile(prefix + ext):
                os.remove(prefix + ext)

    return results
