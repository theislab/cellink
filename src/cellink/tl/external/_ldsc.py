import logging
import os
import subprocess
from typing import Any

import yaml

from cellink._core import DonorData
from cellink.io import to_plink
from cellink.resources._utils import get_data_home

logger = logging.getLogger(__name__)


class LDSCRunner:
    """Enhanced LDSC Runner with YAML config and automatic path inference"""

    def __init__(self, config_path: str | None = None, config_dict: dict | None = None):
        """
        Initialize LDSC Runner

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        config_dict : dict, optional
            Configuration dictionary (takes precedence over config_path)
        """
        self.config = self._load_config(config_path, config_dict)
        self._validate_config()

    def _load_config(self, config_path: str | None, config_dict: dict | None) -> dict:
        """Load configuration from file or dictionary"""
        if config_dict:
            return config_dict

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            "execution_mode": "local",
            "docker_image": "zijingliu/ldsc",
            "singularity_image": None,
            "ldsc_command": "ldsc.py",
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }

    def _validate_config(self):
        """Validate configuration parameters"""
        required_fields = ["execution_mode", "ldsc_command", "make_annot_command", "munge_command"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if self.config["execution_mode"] not in ["local", "docker", "singularity"]:
            raise ValueError("execution_mode must be 'local', 'docker', or 'singularity'")

    def _infer_volumes_from_paths(self, *file_paths: str, data_home: str | None = None) -> dict[str, str]:
        """
        Automatically infer docker volumes or singularity binds from file paths

        Parameters
        ----------
        *file_paths : str
            Variable number of file paths to analyze

        Returns
        -------
        dict
            Dictionary mapping host paths to container paths
        """
        volumes = {}

        volumes[os.getcwd()] = "/data"

        cellink_data_path = get_data_home(data_home)
        if os.path.exists(cellink_data_path):
            volumes[cellink_data_path] = "/cellink_data"

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                parent_dir = os.path.dirname(abs_path)

                covered = False
                for host_path in volumes.keys():
                    host_path = str(host_path)
                    if abs_path.startswith(host_path):
                        covered = True
                        break

                if not covered:
                    container_path = f"/external_{len(volumes)}"
                    volumes[parent_dir] = container_path

        return volumes

    def _convert_path_to_container(self, file_path: str, volumes: dict[str, str]) -> str:
        """Convert host path to container path"""
        if not file_path:
            return file_path

        abs_path = os.path.abspath(file_path)

        for host_path, container_path in volumes.items():
            host_path = str(host_path)
            if abs_path.startswith(host_path):
                relative_path = os.path.relpath(abs_path, host_path)
                return os.path.join(container_path, relative_path).replace("\\", "/")

        return file_path

    def _build_container_command(self, base_command: str, file_paths: list[str] = None) -> str:
        """Build docker or singularity command with volumes"""
        if self.config["execution_mode"] == "local":
            return base_command
        
        if file_paths is None:
            file_paths = []
        
        volumes = self._infer_volumes_from_paths(*file_paths)

        container_command = base_command
        for host_path, container_path in volumes.items():
            container_command = str(container_command).replace(str(host_path), str(container_path))

        if self.config["execution_mode"] == "docker":
            volume_args = []
            for host_path, container_path in volumes.items():
                volume_args.extend(["-v", f"{host_path}:{container_path}"])

            cmd = ["docker", "run", "--rm", *volume_args, "-w", "/data", self.config["docker_image"], base_command]
            return " ".join(cmd)

        elif self.config["execution_mode"] == "singularity":
            bind_args = []
            for host_path, container_path in volumes.items():
                bind_args.extend(["-B", f"{host_path}:{container_path}"])

            cmd = ["singularity", "exec", *bind_args, self.config["singularity_image"], base_command]
            return " ".join(cmd)

        return container_command

    def run_command(self, base_command: str, file_paths: list[str] = None, check: bool = True):
        """
        Execute command with automatic path inference

        Parameters
        ----------
        base_command : str
            The base LDSC command
        file_paths : list, optional
            List of file paths involved in the command (for volume inference)
        check : bool
            Whether to raise exception on command failure
        """
        if file_paths is None:
            file_paths = []

        if self.config["execution_mode"] == "local":
            result = subprocess.run(base_command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        else:
            full_command = self._build_container_command(base_command, file_paths)

            logger.info(f"Executing: {full_command}")
            result = subprocess.run(full_command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)

    @property
    def ldsc_command(self) -> str:
        return self.config["ldsc_command"]

    @property
    def make_annot_command(self) -> str:
        return self.config["make_annot_command"]

    @property
    def munge_command(self) -> str:
        return self.config["munge_command"]

    @property
    def execution_mode(self) -> str:
        return self.config["execution_mode"]


_ldsc_runner = None


def configure_ldsc_runner(config_path: str | None = None, config_dict: dict | None = None) -> LDSCRunner:
    """
    Configure global LDSC runner

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file
    config_dict : dict, optional
        Configuration dictionary

    Returns
    -------
    LDSCRunner
        Configured runner instance
    """
    if config_path is not None and not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    global _ldsc_runner
    _ldsc_runner = LDSCRunner(config_path=config_path, config_dict=config_dict)
    return _ldsc_runner


def get_ldsc_runner() -> LDSCRunner:
    """Get the global LDSC runner instance"""
    global _ldsc_runner
    if _ldsc_runner is None:
        _ldsc_runner = LDSCRunner()
    return _ldsc_runner


def munge_sumstats(
    sumstats_file: str,
    out_prefix: str = "GWAS_summary_statistics_munged",
    n_samples: int | None = None,
    merge_alleles: str | None = None,
    snplist: str | None = None,
    info_min: float = 0.9,
    maf_min: float = 0.01,
    a1_inc: bool = False,
    signed_sumstats: tuple[str, float] | None = None,
    p_col: str | None = None,
    a1_col: str | None = None,
    a2_col: str | None = None,
    snp_col: str | None = None,
    n_col: str | None = None,
    info_col: str | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Munge (clean and standardize) GWAS summary statistics for LDSC analysis

    This function processes raw GWAS summary statistics files to prepare them for
    LD Score regression analysis. It performs quality control, standardizes column
    names, filters SNPs, and aligns alleles to a reference panel.

    Parameters
    ----------
    sumstats_file : str
        Path to input GWAS summary statistics file. Can be plain text or gzipped.
        Should contain columns for SNP ID, effect allele, other allele, and p-value.
    out_prefix : str, default "GWAS_summary_statistics_munged"
        Prefix for output files. Will create {out_prefix}.sumstats.gz
    n_samples : int, optional
        Total sample size. If the summary statistics file has a sample size column,
        this will be used to verify it. If there's no sample size column, this will
        be added to all SNPs.
    merge_alleles : str, optional
        Path to reference allele file (e.g., w_hm3.snplist) for aligning alleles
        and removing strand-ambiguous SNPs. Recommended for downstream analysis.
    snplist : str, optional
        Path to file with SNP IDs to keep. Only SNPs in this list will be retained.
    info_min : float, default 0.9
        Minimum INFO score for SNP inclusion. SNPs with INFO < info_min are removed.
    maf_min : float, default 0.01
        Minimum minor allele frequency for SNP inclusion. SNPs with MAF < maf_min
        are removed.
    a1_inc : bool, default False
        If True, A1 is the effect allele (increasing allele). If False, A1 is the
        other allele and the sign of the effect will be flipped.
    signed_sumstats : tuple[str, float], optional
        Tuple of (column_name, sign) for identifying the direction of effect.
        Example: ("OR", 1) means odds ratios where values >1 indicate positive effect.
        Example: ("BETA", 0) means betas where values >0 indicate positive effect.
    p_col : str, optional
        Name of the p-value column if non-standard (default: "P")
    a1_col : str, optional
        Name of the effect allele column if non-standard (default: "A1")
    a2_col : str, optional
        Name of the other allele column if non-standard (default: "A2")
    snp_col : str, optional
        Name of the SNP ID column if non-standard (default: "SNP")
    n_col : str, optional
        Name of the sample size column if non-standard (default: "N")
    info_col : str, optional
        Name of the INFO score column if non-standard (default: "INFO")
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to munge_sumstats.py
        Common options include:
        - ignore: List of columns to ignore
        - daner: Set if input is in daner format (PGC)
        - no-alleles: Don't require allele information
        - merge-alleles: Alternative way to specify reference alleles

    Returns
    -------
    dict
        Results dictionary containing:
        - 'sumstats_file': Path to the munged summary statistics file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Raises
    ------
    subprocess.CalledProcessError
        If the munging process fails (e.g., due to malformed input file)

    Examples
    --------
    Basic usage with standard column names:
    >>> result = munge_sumstats(
    ...     sumstats_file="height_gwas.txt.gz",
    ...     out_prefix="height_munged",
    ...     n_samples=253288,
    ...     merge_alleles="w_hm3.snplist",
    ... )

    With custom column names:
    >>> result = munge_sumstats(
    ...     sumstats_file="custom_gwas.txt",
    ...     out_prefix="custom_munged",
    ...     n_samples=50000,
    ...     snp_col="RSID",
    ...     a1_col="EFFECT_ALLELE",
    ...     a2_col="OTHER_ALLELE",
    ...     p_col="PVAL",
    ...     signed_sumstats=("BETA", 0),
    ... )

    Case-control study with odds ratios:
    >>> result = munge_sumstats(
    ...     sumstats_file="case_control_gwas.txt.gz",
    ...     out_prefix="case_control_munged",
    ...     n_samples=10000,
    ...     merge_alleles="w_hm3.snplist",
    ...     signed_sumstats=("OR", 1),
    ...     a1_inc=True,
    ... )

    Just generate the command without running:
    >>> result = munge_sumstats(
    ...     sumstats_file="height_gwas.txt.gz", out_prefix="height_munged", n_samples=253288, run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - The function expects summary statistics files to follow standard GWAS format
    - Strand-ambiguous SNPs (A/T or G/C) are removed when merge_alleles is used
    - The output file will be gzipped and named {out_prefix}.sumstats.gz
    - It's highly recommended to use merge_alleles with a reference panel (e.g., HapMap3)
      to ensure proper allele alignment
    - For binary traits, signed_sumstats should typically be ("OR", 1) or ("BETA", 0)
    - For quantitative traits, signed_sumstats is typically ("BETA", 0) or ("Z", 0)
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = f"{runner.munge_command} --sumstats {sumstats_file} --out {out_prefix}"

    if n_samples is not None:
        cmd += f" --N {n_samples}"
    if merge_alleles is not None:
        cmd += f" --merge-alleles {merge_alleles}"
    if snplist is not None:
        cmd += f" --merge {snplist}"
    if info_min != 0.9:
        cmd += f" --info-min {info_min}"
    if maf_min != 0.01:
        cmd += f" --maf-min {maf_min}"
    if a1_inc:
        cmd += " --a1-inc"

    if signed_sumstats is not None:
        col, min_val = signed_sumstats
        cmd += f" --signed-sumstats {col},{min_val}"
    if p_col is not None:
        cmd += f" --p {p_col}"
    if a1_col is not None:
        cmd += f" --a1 {a1_col}"
    if a2_col is not None:
        cmd += f" --a2 {a2_col}"
    if snp_col is not None:
        cmd += f" --snp {snp_col}"
    if n_col is not None:
        cmd += f" --N-col {n_col}"
    if info_col is not None:
        cmd += f" --info {info_col}"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file]
    if merge_alleles:
        file_paths.append(merge_alleles)
    if snplist:
        file_paths.append(snplist)

    if run:
        logger.info(f"Running munge_sumstats: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.sumstats.gz"
    else:
        return runner._build_container_command(cmd, file_paths)


def _run_ldsc_estimate_ld_scores(
    bfile_prefix: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Estimate LD Scores from genotype data
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = f"{runner.ldsc_command} --bfile {bfile_prefix} --l2 --out {out_prefix}"

    flags = [ld_wind_kb, ld_wind_snp, ld_wind_cm]
    non_null_flags = sum(f is not None for f in flags)

    if non_null_flags > 1:
        raise ValueError("Only one of ld_wind_kb, ld_wind_snp, or ld_wind_cm may be specified.")

    if ld_wind_kb is not None:
        cmd += f" --ld-wind-kb {ld_wind_kb}"
    elif ld_wind_snp is not None:
        cmd += f" --ld-wind-snp {ld_wind_snp}"
    else:
        cmd += f" --ld-wind-cm {ld_wind_cm}"

    if annot_file is not None:
        cmd += f" --annot {annot_file}"
        if thin_annot:
            cmd += " --thin-annot"

    if print_snps is not None:
        cmd += f" --print-snps {print_snps}"

    if maf_min != 0.01:
        cmd += f" --maf {maf_min}"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    cmd += " --yes-really"

    file_paths = [f"{bfile_prefix}.bed", f"{bfile_prefix}.bim", f"{bfile_prefix}.fam"]
    if annot_file:
        file_paths.append(annot_file)
    if print_snps:
        file_paths.append(print_snps)

    if run:
        logger.info(f"Estimating LD scores: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.l2.ldscore.gz"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_ld_scores_from_bimfile(
    bfile_prefix: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate LD scores from PLINK bfile (works with any bfile, including 1000G)

    Parameters
    ----------
    bfile_prefix : str
        Path to PLINK binary files (without .bed/.bim/.fam extension)
    out_prefix : str
        Prefix for output files
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    annot_file : str, optional
        Annotation file for computing category-specific LD scores
    thin_annot : bool, default False
        Thin the annot file by removing columns with <1% SNPs
    print_snps : str, optional
        File with SNP IDs to restrict LD score computation
    maf_min : float, default 0.01
        Minimum MAF threshold
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> # Using 1000G reference panel
    >>> result = estimate_ld_scores_from_bimfile(
    ...     bfile_prefix="1000G_EUR_Phase3_plink/1000G.EUR.QC.22",
    ...     out_prefix="my_ldscores_chr22",
    ...     annot_file="immune_genes.22.annot.gz",
    ...     print_snps="hm3_snps.txt",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    results = {}

    result_file = _run_ldsc_estimate_ld_scores(
        bfile_prefix=bfile_prefix,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        annot_file=annot_file,
        thin_annot=thin_annot,
        print_snps=print_snps,
        maf_min=maf_min,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["ld_scores_file"] = result_file
        results["files_created"] = [
            f"{out_prefix}.l2.ldscore.gz",
            f"{out_prefix}.l2.M",
            f"{out_prefix}.l2.M_5_50",
            f"{out_prefix}.log",
        ]
    else:
        results["command"] = result_file

    return results


def estimate_ld_scores_from_donor_data(
    dd: DonorData,
    out_prefix: str = "ldscores",
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate LD scores from DonorData object

    This convenience function exports genotype data from DonorData to PLINK format,
    then computes LD scores.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information
    out_prefix : str, default "ldscores"
        Prefix for output files (also used for temporary PLINK files)
    cleanup_files : bool, default True
        Whether to remove temporary PLINK files after computing LD scores
    plink_export_kwargs : dict, optional
        Additional keyword arguments to pass to to_plink()
    ... (other parameters as in estimate_ld_scores_from_bimfile)

    Returns
    -------
    dict
        Results dictionary (same as estimate_ld_scores_from_bimfile)

    Examples
    --------
    >>> result = estimate_ld_scores_from_donor_data(
    ...     dd=my_donor_data, out_prefix="my_ldscores", annot_file="immune_genes.annot.gz", ld_wind_cm=1.0
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    logger.info("Exporting genotype data to PLINK format for LD score estimation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)

    results = estimate_ld_scores_from_bimfile(
        bfile_prefix=out_prefix,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        annot_file=annot_file,
        thin_annot=thin_annot,
        print_snps=print_snps,
        maf_min=maf_min,
        run=run,
        runner=runner,
        **kwargs,
    )

    if cleanup_files and run:
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def _run_ldsc_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: float | None = None,
    pop_prev: float | None = None,
    intercept_h2: float | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Estimate SNP heritability using LD Score regression
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = (
        f"{runner.ldsc_command} --h2 {sumstats_file} --ref-ld-chr {ref_ld_chr} --w-ld-chr {w_ld_chr} --out {out_prefix}"
    )

    if overlap_annot:
        cmd += " --overlap-annot"
        if frqfile_chr is None:
            logger.warning("--overlap-annot requires --frqfile-chr")

    if frqfile_chr is not None:
        cmd += f" --frqfile-chr {frqfile_chr}"

    if not_m_5_50:
        cmd += " --not-M-5-50"

    if print_coefficients:
        cmd += " --print-coefficients"

    if print_delete_vals:
        cmd += " --print-delete-vals"

    if samp_prev is not None:
        cmd += f" --samp-prev {samp_prev}"

    if pop_prev is not None:
        cmd += f" --pop-prev {pop_prev}"

    if intercept_h2 is not None:
        cmd += f" --intercept-h2 {intercept_h2}"

    if no_intercept:
        cmd += " --no-intercept"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file, ref_ld_chr, w_ld_chr]
    if frqfile_chr:
        file_paths.append(frqfile_chr)

    if run:
        logger.info(f"Estimating heritability: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: float | None = None,
    pop_prev: float | None = None,
    intercept_h2: float | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate SNP heritability using LD Score regression

    Convenience wrapper around run_ldsc_heritability with validation and
    structured output.

    Parameters
    ----------
    sumstats_file : str
        Path to munged summary statistics file (.sumstats.gz)
    ref_ld_chr : str
        Prefix for reference LD scores (with @, e.g., "baseline.")
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights.")
    out_prefix : str
        Prefix for output files
    overlap_annot : bool, default False
        Use overlapping annotation model
    frqfile_chr : str, optional
        Prefix for allele frequency files (required with overlap_annot)
    not_m_5_50 : bool, default False
        Don't restrict to common SNPs for estimating h2
    print_coefficients : bool, default False
        Print coefficient estimates
    print_delete_vals : bool, default False
        Print delete values
    samp_prev : float, optional
        Sample prevalence (for binary traits)
    pop_prev : float, optional
        Population prevalence (for binary traits)
    intercept_h2 : float, optional
        Constrain the LD Score regression intercept
    no_intercept : bool, default False
        Force intercept to 1
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> result = estimate_heritability(
    ...     sumstats_file="height_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     out_prefix="height_h2",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_file:
        raise ValueError("sumstats_file is required")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")

    results = {}

    result_file = _run_ldsc_heritability(
        sumstats_file=sumstats_file,
        ref_ld_chr=ref_ld_chr,
        w_ld_chr=w_ld_chr,
        out_prefix=out_prefix,
        overlap_annot=overlap_annot,
        frqfile_chr=frqfile_chr,
        not_m_5_50=not_m_5_50,
        print_coefficients=print_coefficients,
        print_delete_vals=print_delete_vals,
        samp_prev=samp_prev,
        pop_prev=pop_prev,
        intercept_h2=intercept_h2,
        no_intercept=no_intercept,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["log_file"] = result_file
        results["files_created"] = [f"{out_prefix}.log"]
    else:
        results["command"] = result_file

    return results


def _run_ldsc_genetic_correlation(
    sumstats_files: list[str],
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: list[float] | None = None,
    pop_prev: list[float] | None = None,
    intercept_h2: list[float] | None = None,
    intercept_gencov: list[float] | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Estimate genetic correlation using LD Score regression
    """
    if runner is None:
        runner = get_ldsc_runner()

    sumstats_str = ",".join(sumstats_files)
    cmd = (
        f"{runner.ldsc_command} --rg {sumstats_str} --ref-ld-chr {ref_ld_chr} --w-ld-chr {w_ld_chr} --out {out_prefix}"
    )

    if overlap_annot:
        cmd += " --overlap-annot"
        if frqfile_chr is None:
            logger.warning("--overlap-annot requires --frqfile-chr")

    if frqfile_chr is not None:
        cmd += f" --frqfile-chr {frqfile_chr}"

    if not_m_5_50:
        cmd += " --not-M-5-50"

    if print_coefficients:
        cmd += " --print-coefficients"

    if print_delete_vals:
        cmd += " --print-delete-vals"

    if samp_prev is not None:
        samp_prev_str = ",".join([str(x) if x is not None else "nan" for x in samp_prev])
        cmd += f" --samp-prev {samp_prev_str}"

    if pop_prev is not None:
        pop_prev_str = ",".join([str(x) if x is not None else "nan" for x in pop_prev])
        cmd += f" --pop-prev {pop_prev_str}"

    if intercept_h2 is not None:
        intercept_h2_str = ",".join([str(x) for x in intercept_h2])
        cmd += f" --intercept-h2 {intercept_h2_str}"

    if intercept_gencov is not None:
        intercept_gencov_str = ",".join([str(x).replace("-", "N") for x in intercept_gencov])
        cmd += f" --intercept-gencov {intercept_gencov_str}"

    if no_intercept:
        cmd += " --no-intercept"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = sumstats_files + [ref_ld_chr, w_ld_chr]
    if frqfile_chr:
        file_paths.append(frqfile_chr)

    if run:
        logger.info(f"Estimating genetic correlation: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_genetic_correlation(
    sumstats_files: list[str],
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: list[float] | None = None,
    pop_prev: list[float] | None = None,
    intercept_h2: list[float] | None = None,
    intercept_gencov: list[float] | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate genetic correlation using LD Score regression

    Convenience wrapper around run_ldsc_genetic_correlation with validation
    and structured output.

    Parameters
    ----------
    sumstats_files : list[str]
        List of paths to munged summary statistics files (.sumstats.gz)
    ref_ld_chr : str
        Prefix for reference LD scores (with @, e.g., "baseline.")
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights.")
    out_prefix : str
        Prefix for output files
    overlap_annot : bool, default False
        Use overlapping annotation model
    frqfile_chr : str, optional
        Prefix for allele frequency files (required with overlap_annot)
    not_m_5_50 : bool, default False
        Don't restrict to common SNPs
    print_coefficients : bool, default False
        Print coefficient estimates
    print_delete_vals : bool, default False
        Print delete values
    samp_prev : list[float], optional
        Sample prevalences for each trait (use None for quantitative traits)
    pop_prev : list[float], optional
        Population prevalences for each trait
    intercept_h2 : list[float], optional
        Constrain h2 intercepts for each trait
    intercept_gencov : list[float], optional
        Constrain genetic covariance intercepts
    no_intercept : bool, default False
        Force intercepts to 1 and 0
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> result = estimate_genetic_correlation(
    ...     sumstats_files=["height_munged.sumstats.gz", "bmi_munged.sumstats.gz"],
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     out_prefix="height_bmi_rg",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_files or len(sumstats_files) < 2:
        raise ValueError("sumstats_files must contain at least 2 files for genetic correlation")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")

    results = {}

    result_file = _run_ldsc_genetic_correlation(
        sumstats_files=sumstats_files,
        ref_ld_chr=ref_ld_chr,
        w_ld_chr=w_ld_chr,
        out_prefix=out_prefix,
        overlap_annot=overlap_annot,
        frqfile_chr=frqfile_chr,
        not_m_5_50=not_m_5_50,
        print_coefficients=print_coefficients,
        print_delete_vals=print_delete_vals,
        samp_prev=samp_prev,
        pop_prev=pop_prev,
        intercept_h2=intercept_h2,
        intercept_gencov=intercept_gencov,
        no_intercept=no_intercept,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["log_file"] = result_file
        results["files_created"] = [f"{out_prefix}.log"]
    else:
        results["command"] = result_file

    return results


def _run_ldsc_make_annot(
    bimfile: str,
    annot_file: str,
    gene_set_file: str | None = None,
    gene_coord_file: str | None = None,
    windowsize: int | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Internal function to create annotation files using make_annot.py

    Either gene_set_file or bed_file must be provided.
    Returns annot_file path if run=True, otherwise command string.
    """
    if runner is None:
        runner = get_ldsc_runner()

    if gene_set_file is None and bed_file is None:
        raise ValueError("Either gene_set_file or bed_file must be provided")

    cmd = f"{runner.make_annot_command} --bimfile {bimfile} --annot-file {annot_file}"

    if gene_set_file is not None:
        cmd += f" --gene-set-file {gene_set_file}"

        if gene_coord_file is not None:
            cmd += f" --gene-coord-file {gene_coord_file}"

        if windowsize is not None:
            cmd += f" --windowsize {windowsize}"

    if bed_file is not None:
        cmd += f" --bed-file {bed_file}"

        if nomerge:
            cmd += " --nomerge"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [bimfile]
    if gene_set_file:
        file_paths.append(gene_set_file)
    if gene_coord_file:
        file_paths.append(gene_coord_file)
    if bed_file:
        file_paths.append(bed_file)

    if run:
        logger.info(f"Creating annotation file: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return annot_file
    else:
        return runner._build_container_command(cmd, file_paths)


def make_annot_from_bimfile(
    bimfile: str,
    annot_file: str,
    gene_set_file: str | None = None,
    gene_coord_file: str | None = None,
    windowsize: int | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create annotation file from a PLINK bimfile

    This function creates binary annotation files that indicate which SNPs belong to
    specific genomic regions or gene sets. These annotations can be used with LDSC
    to compute category-specific LD scores. Works with any PLINK bimfile, including
    standard reference panels like 1000 Genomes.

    Parameters
    ----------
    bimfile : str
        Path to PLINK .bim file (e.g., from 1000 Genomes reference panel).
        This defines the SNPs for which annotations will be created.
    annot_file : str
        The name of the annot file to output. Should typically end in .annot or .annot.gz
    gene_set_file : str, optional
        A file of gene names, one line per gene. Used for gene-set based annotations.
        Either this or bed_file must be provided.
    gene_coord_file : str, optional
        A file with columns GENE, CHR, START, and END, where START and END are
        base pair coordinates of TSS and TES. This file can contain more genes
        than are in the gene set. Default ENSG_coord.txt is provided by LDSC.
        Only used with gene_set_file.
    windowsize : int, optional
        How many base pairs to add around the transcribed region to make the annotation.
        Only used with gene-set based annotations. Typical values: 0-500000 (0-500kb).
    bed_file : str, optional
        The UCSC bed file with the regions that make up your annotation.
        Used for region-based annotations. Either this or gene_set_file must be provided.
    nomerge : bool, default False
        Don't merge the bed file; make an annot file with values proportional to
        the number of intervals in the bedfile overlapping the SNP. Only used with bed_file.
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to make_annot.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'annot_file': Path to the created annotation file
        - 'files_created': List of files created (if run=True)
        - 'command': Command string (if run=False)

    Raises
    ------
    ValueError
        If neither gene_set_file nor bed_file is provided

    Examples
    --------
    Gene-set based annotation for chromosome 22:
    >>> result = make_annot_from_bimfile(
    ...     bimfile="1000G_EUR_Phase3_plink/1000G.EUR.QC.22.bim",
    ...     annot_file="immune_genes.22.annot.gz",
    ...     gene_set_file="immune_genes.txt",
    ...     gene_coord_file="ENSG_coord.txt",
    ...     windowsize=100000,
    ... )

    BED-file based annotation for enhancer regions:
    >>> result = make_annot_from_bimfile(
    ...     bimfile="1000G.EUR.QC.1.bim", annot_file="enhancers.1.annot.gz", bed_file="enhancers.bed"
    ... )

    Generate command without running:
    >>> result = make_annot_from_bimfile(
    ...     bimfile="1000G.EUR.QC.22.bim",
    ...     annot_file="my_annot.22.annot.gz",
    ...     gene_set_file="my_genes.txt",
    ...     gene_coord_file="ENSG_coord.txt",
    ...     windowsize=50000,
    ...     run=False,
    ... )
    >>> print(result["command"])

    Notes
    -----
    - Either gene_set_file or bed_file must be provided, but not both
    - gene_coord_file and windowsize are only used with gene_set_file
    - nomerge is only used with bed_file
    - The output annotation file has one row per SNP in the bimfile, with 1
      indicating the SNP is in the annotation and 0 otherwise
    - For whole-genome analyses, this should be run separately for each chromosome
    - Typical workflow: Create annotations for chr 1-22, then compute LD scores
      for each chromosome using these annotations

    See Also
    --------
    make_annot_from_donor_data : Create annotations from DonorData object
    estimate_ld_scores_from_bimfile : Compute LD scores using annotations
    """
    if runner is None:
        runner = get_ldsc_runner()

    results = {"annot_file": annot_file, "files_created": []}

    result_file = _run_ldsc_make_annot(
        bimfile=bimfile,
        annot_file=annot_file,
        gene_set_file=gene_set_file,
        gene_coord_file=gene_coord_file,
        windowsize=windowsize,
        bed_file=bed_file,
        nomerge=nomerge,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["annot_file"] = result_file
        results["files_created"].append(annot_file)
    else:
        results["command"] = result_file

    return results


def make_annot_from_donor_data(
    dd: DonorData,
    annot_file: str,
    gene_set_file: str | None = None,
    gene_coord_file: str | None = None,
    windowsize: int | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    out_prefix: str = "ldsc_annot",
    run: bool = True,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create annotation file from DonorData object

    This convenience function exports genotype data from a DonorData object to PLINK
    format, then creates binary annotation files that indicate which SNPs belong to
    specific genomic regions or gene sets. These annotations can be used with LDSC
    to compute category-specific LD scores.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information
    annot_file : str
        The name of the annot file to output. Should typically end in .annot or .annot.gz
    gene_set_file : str, optional
        A file of gene names, one line per gene. Used for gene-set based annotations.
        Either this or bed_file must be provided.
    gene_coord_file : str, optional
        A file with columns GENE, CHR, START, and END, where START and END are
        base pair coordinates of TSS and TES. This file can contain more genes
        than are in the gene set. Default ENSG_coord.txt is provided by LDSC.
        Only used with gene_set_file.
    windowsize : int, optional
        How many base pairs to add around the transcribed region to make the annotation.
        Only used with gene-set based annotations. Typical values: 0-500000 (0-500kb).
        Common choices:
        - 0: Only SNPs within gene body
        - 10000: ±10kb around gene
        - 100000: ±100kb around gene (default in many studies)
    bed_file : str, optional
        The UCSC bed file with the regions that make up your annotation.
        Used for region-based annotations. Either this or gene_set_file must be provided.
    nomerge : bool, default False
        Don't merge the bed file; make an annot file with values proportional to
        the number of intervals in the bedfile overlapping the SNP. Only used with bed_file.
    out_prefix : str, default "ldsc_annot"
        Prefix for temporary PLINK files created during export
    run : bool, default True
        Whether to execute the command or just return it
    cleanup_files : bool, default True
        Whether to remove temporary PLINK files after creating annotations.
        If True, removes {out_prefix}.bed, .bim, and .fam files.
    plink_export_kwargs : dict, optional
        Additional keyword arguments to pass to to_plink()
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to make_annot.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'annot_file': Path to the created annotation file
        - 'files_created': List of files created (if run=True)
        - 'command': Command string (if run=False)

    Raises
    ------
    ValueError
        If neither gene_set_file nor bed_file is provided

    Examples
    --------
    Create gene-set annotation from DonorData:
    >>> result = make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="immune_genes.annot.gz",
    ...     gene_set_file="immune_genes.txt",
    ...     gene_coord_file="ENSG_coord.txt",
    ...     windowsize=100000,
    ... )

    Create BED-file annotation for regulatory regions:
    >>> result = make_annot_from_donor_data(dd=my_donor_data, annot_file="enhancers.annot.gz", bed_file="enhancers.bed")

    Keep temporary PLINK files for inspection:
    >>> result = make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="my_annot.annot.gz",
    ...     gene_set_file="my_genes.txt",
    ...     gene_coord_file="ENSG_coord.txt",
    ...     windowsize=50000,
    ...     cleanup_files=False,
    ... )

    Generate command without running:
    >>> result = make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="my_annot.annot.gz",
    ...     gene_set_file="my_genes.txt",
    ...     gene_coord_file="ENSG_coord.txt",
    ...     windowsize=100000,
    ...     run=False,
    ... )
    >>> print(result["command"])

    Notes
    -----
    - This function exports dd.G to PLINK format, creates the annotation,
      then optionally cleans up the temporary PLINK files
    - Either gene_set_file or bed_file must be provided, but not both
    - The output annotation file has one row per SNP, with 1 indicating the SNP
      is in the annotation and 0 otherwise
    - gene_coord_file should contain coordinates for all genes you might annotate,
      not just those in your specific gene set
    - For gene-based annotations, the annotation includes SNPs within windowsize bp
      of the transcribed region (TSS to TES)
    - Temporary PLINK files are created in the current directory and cleaned up by
      default, but you can set cleanup_files=False to keep them

    See Also
    --------
    make_annot_from_bimfile : Create annotations from existing PLINK bimfile
    estimate_ld_scores_from_donor_data : Compute LD scores from DonorData
    """
    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    logger.info("Exporting genotype data to PLINK format for annotation creation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)
    bimfile = f"{out_prefix}.bim"

    results = _run_ldsc_make_annot(
        bimfile=bimfile,
        annot_file=annot_file,
        gene_set_file=gene_set_file,
        gene_coord_file=gene_coord_file,
        windowsize=windowsize,
        bed_file=bed_file,
        nomerge=nomerge,
        run=run,
        runner=runner,
        **kwargs,
    )

    if cleanup_files and run:
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def compute_ld_scores_with_annotations_from_bimfile(
    bfile_prefix: str,
    annot_file: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    print_snps: str | None = None,
    thin_annot: bool = True,
    maf_min: float = 0.01,
    yes_really: bool = True,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Compute LD scores with cell-type-specific annotations from PLINK bfile

    This is the first step in cell-type-specific LDSC analysis. It computes
    LD scores for SNPs while incorporating cell-type-specific gene annotations.
    This function should be run for each chromosome and each cell type.

    Works with any PLINK bfile, including standard reference panels like 1000 Genomes.

    Parameters
    ----------
    bfile_prefix : str
        Path to PLINK binary files (without .bed/.bim/.fam extension).
        Typically from 1000 Genomes reference panel, e.g.,
        "1000G_EUR_Phase3_plink/1000G.EUR.QC.22"
    annot_file : str
        Path to the annotation file created by make_annot_from_donor_data()
        or make_annot_from_bimfile(). Should end in .annot.gz
        Example: "CD8_Naive.22.annot.gz"
    out_prefix : str
        Prefix for output files. Will create:
        - {out_prefix}.l2.ldscore.gz (LD scores)
        - {out_prefix}.l2.M (number of SNPs)
        - {out_prefix}.l2.M_5_50 (number of common SNPs)
        - {out_prefix}.log (log file)
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans. Only one of ld_wind_cm, ld_wind_kb,
        or ld_wind_snp can be specified.
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    print_snps : str, optional
        Path to file with SNP IDs (one per row) to restrict LD score computation.
        Commonly used with HapMap3 SNPs (e.g., "hapmap3_snps/hm.22.snp").
        The sum r^2 will still include all SNPs, but only listed SNPs will
        have LD scores computed.
    thin_annot : bool, default True
        Assume annotation files only have annotations (no SNP, CM, CHR, BP columns).
        Should typically be True for annotations created by make_annot functions.
    maf_min : float, default 0.01
        Minimum minor allele frequency threshold
    yes_really : bool, default True
        Required flag for computing whole-chromosome LD scores
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage for chromosome 22:
    >>> result = compute_ld_scores_with_annotations_from_bimfile(
    ...     bfile_prefix="1000G_EUR_Phase3_plink/1000G.EUR.QC.22",
    ...     annot_file="CD8_Naive.22.annot.gz",
    ...     out_prefix="CD8_Naive.22",
    ...     print_snps="hapmap3_snps/hm.22.snp",
    ... )

    For all chromosomes (in a loop):
    >>> for chrom in range(1, 23):
    ...     result = compute_ld_scores_with_annotations_from_bimfile(
    ...         bfile_prefix=f"1000G_EUR/1000G.EUR.QC.{chrom}",
    ...         annot_file=f"CD8_Naive.{chrom}.annot.gz",
    ...         out_prefix=f"CD8_Naive.{chrom}",
    ...         print_snps=f"hapmap3_snps/hm.{chrom}.snp",
    ...     )

    Just generate command without running:
    >>> result = compute_ld_scores_with_annotations_from_bimfile(
    ...     bfile_prefix="1000G.EUR.QC.22", annot_file="CD8_Naive.22.annot.gz", out_prefix="CD8_Naive.22", run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - This function is specifically for cell-type-specific analysis workflow
    - Should be run separately for each chromosome (1-22)
    - The annotation file should be created first using make_annot_from_donor_data()
      or make_annot_from_bimfile()
    - print_snps is typically used to restrict to HapMap3 SNPs for better
      matching with standard reference LD scores
    - After computing LD scores for all chromosomes, use
      estimate_celltype_specific_heritability() for the actual analysis

    See Also
    --------
    compute_ld_scores_with_annotations_from_donor_data : Compute from DonorData
    make_annot_from_donor_data : Create annotations from DonorData
    estimate_celltype_specific_heritability : Run cell-type-specific analysis
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = f"{runner.ldsc_command} --l2 --bfile {bfile_prefix} --annot {annot_file} --out {out_prefix}"

    flags = [ld_wind_kb, ld_wind_snp, ld_wind_cm]
    non_null_flags = sum(f is not None for f in flags)

    if non_null_flags > 1:
        raise ValueError("Only one of ld_wind_kb, ld_wind_snp, or ld_wind_cm may be specified.")

    if ld_wind_kb is not None:
        cmd += f" --ld-wind-kb {ld_wind_kb}"
    elif ld_wind_snp is not None:
        cmd += f" --ld-wind-snp {ld_wind_snp}"
    else:
        cmd += f" --ld-wind-cm {ld_wind_cm}"

    if thin_annot:
        cmd += " --thin-annot"

    if print_snps is not None:
        cmd += f" --print-snps {print_snps}"

    if maf_min != 0.01:
        cmd += f" --maf {maf_min}"

    if yes_really:
        cmd += " --yes-really"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [f"{bfile_prefix}.bed", f"{bfile_prefix}.bim", f"{bfile_prefix}.fam", annot_file]
    if print_snps:
        file_paths.append(print_snps)

    if run:
        logger.info(f"Computing LD scores with annotations: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)

        return {
            "ld_scores_file": f"{out_prefix}.l2.ldscore.gz",
            "files_created": [
                f"{out_prefix}.l2.ldscore.gz",
                f"{out_prefix}.l2.M",
                f"{out_prefix}.l2.M_5_50",
                f"{out_prefix}.log",
            ],
        }
    else:
        return {"command": runner._build_container_command(cmd, file_paths)}


def compute_ld_scores_with_annotations_from_donor_data(
    dd: DonorData,
    annot_file: str,
    out_prefix: str = "ldscores_annot",
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    print_snps: str | None = None,
    thin_annot: bool = True,
    maf_min: float = 0.01,
    yes_really: bool = True,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Compute LD scores with cell-type-specific annotations from DonorData object

    This convenience function exports genotype data from DonorData to PLINK format,
    then computes LD scores with cell-type-specific annotations. This is useful when
    you want to compute LD scores from your own genotype data rather than using a
    reference panel like 1000 Genomes.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information
    annot_file : str
        Path to the annotation file created by make_annot_from_donor_data()
        or make_annot_from_bimfile(). Should end in .annot.gz
        Example: "CD8_Naive.annot.gz"
    out_prefix : str, default "ldscores_annot"
        Prefix for output files (also used for temporary PLINK files).
        Will create:
        - {out_prefix}.l2.ldscore.gz (LD scores)
        - {out_prefix}.l2.M (number of SNPs)
        - {out_prefix}.l2.M_5_50 (number of common SNPs)
        - {out_prefix}.log (log file)
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans. Only one of ld_wind_cm, ld_wind_kb,
        or ld_wind_snp can be specified.
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    print_snps : str, optional
        Path to file with SNP IDs (one per row) to restrict LD score computation.
        Commonly used with HapMap3 SNPs (e.g., "hapmap3_snps/hm.22.snp").
    thin_annot : bool, default True
        Assume annotation files only have annotations (no SNP, CM, CHR, BP columns).
        Should typically be True for annotations created by make_annot functions.
    maf_min : float, default 0.01
        Minimum minor allele frequency threshold
    yes_really : bool, default True
        Required flag for computing whole-chromosome LD scores
    cleanup_files : bool, default True
        Whether to remove temporary PLINK files after computing LD scores.
        If True, removes {out_prefix}.bed, .bim, and .fam files.
    plink_export_kwargs : dict, optional
        Additional keyword arguments to pass to to_plink()
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     out_prefix="CD8_Naive_ldscores",
    ...     print_snps="hapmap3_snps.txt",
    ... )

    Complete workflow for cell-type analysis:
    >>> # 1. Create annotation from DonorData
    >>> annot_result = make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     gene_set_file="CD8_Naive.GeneSet",
    ...     gene_coord_file="gene_coords.txt",
    ...     windowsize=100000,
    ... )

    >>> # 2. Compute LD scores with annotations
    >>> ldsc_result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="CD8_Naive.annot.gz", out_prefix="CD8_Naive_ldscores"
    ... )

    Keep temporary PLINK files:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="immune_genes.annot.gz", out_prefix="my_ldscores", cleanup_files=False
    ... )

    Just generate command:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="CD8_Naive.annot.gz", out_prefix="CD8_Naive_ldscores", run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - This function exports dd.G to PLINK format, computes LD scores with annotations,
      then optionally cleans up the temporary PLINK files
    - The annotation file must match the SNPs in the DonorData object
    - Typically used when you have your own genotype data and want to compute
      custom LD scores rather than using pre-computed reference LD scores
    - For standard cell-type-specific heritability analysis, it's more common to use
      compute_ld_scores_with_annotations_from_bimfile() with 1000 Genomes data
    - Temporary PLINK files are created in the current directory and cleaned up by
      default, but you can set cleanup_files=False to keep them

    See Also
    --------
    compute_ld_scores_with_annotations_from_bimfile : Compute from existing PLINK files
    make_annot_from_donor_data : Create annotations from DonorData
    estimate_celltype_specific_heritability : Run cell-type-specific analysis
    """
    if runner is None:
        runner = get_ldsc_runner()

    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    logger.info("Exporting genotype data to PLINK format for LD score computation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)

    results = compute_ld_scores_with_annotations_from_bimfile(
        bfile_prefix=out_prefix,
        annot_file=annot_file,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        print_snps=print_snps,
        thin_annot=thin_annot,
        maf_min=maf_min,
        yes_really=yes_really,
        run=run,
        runner=runner,
        **kwargs,
    )

    if cleanup_files and run:
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def estimate_celltype_specific_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    ref_ld_chr_cts: str,
    out_prefix: str,
    print_all_cts: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate cell-type-specific heritability using LD Score regression

    This is the second step in cell-type-specific LDSC analysis. It tests whether
    SNP heritability is enriched in specific cell types by regressing GWAS summary
    statistics against cell-type-specific LD scores.

    This function requires that LD scores with cell-type annotations have already
    been computed using compute_ld_scores_with_annotations() for all chromosomes.

    Parameters
    ----------
    sumstats_file : str
        Path to munged summary statistics file (.sumstats.gz) from munge_sumstats()
    ref_ld_chr : str
        Prefix for baseline reference LD scores (with @, e.g., "baseline_v1.2/baseline.").
        These are the standard LD scores used for controlling confounders.
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights_hm3_no_hla/weights.").
        These are standard weights files from the LDSC resources.
    ref_ld_chr_cts : str
        Path to control file listing cell-type-specific LD score prefixes.
        This file should have two tab-separated columns per line:
        - Cell type name
        - Prefix for that cell type's LD scores (with @ for chromosome)

        Example file content:
        ```
        CD8_Naive    cts_ldscores/CD8_Naive.
        CD4_Memory   cts_ldscores/CD4_Memory.
        B_cells      cts_ldscores/B_cells.
        ```

        LDSC will look for files like:
        cts_ldscores/CD8_Naive.1.l2.ldscore.gz through
        cts_ldscores/CD8_Naive.22.l2.ldscore.gz
    out_prefix : str
        Prefix for output files. Will create:
        - {out_prefix}.cell_type_results.txt (main results)
        - {out_prefix}.log (log file)
    print_all_cts : bool, default False
        Print results for all cell types (not just significant ones)
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'results_file': Path to cell type results file (if run=True)
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage after computing LD scores:
    >>> # First create control file
    >>> with open("celltype_ldscores.txt", "w") as f:
    ...     f.write("CD8_Naive\\tcts_ldscores/CD8_Naive.\\n")
    ...     f.write("CD4_Memory\\tcts_ldscores/CD4_Memory.\\n")
    ...     f.write("B_cells\\tcts_ldscores/B_cells.\\n")

    >>> # Run cell-type-specific analysis
    >>> result = estimate_celltype_specific_heritability(
    ...     sumstats_file="height_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     ref_ld_chr_cts="celltype_ldscores.txt",
    ...     out_prefix="height_celltype_results",
    ... )

    Complete workflow example:
    >>> # 1. Prepare annotations for each cell type and chromosome
    >>> for cell_type in ["CD8_Naive", "CD4_Memory"]:
    ...     for chrom in range(1, 23):
    ...         make_annot_from_donor_data(
    ...             dd=dd_chr,
    ...             annot_file=f"annots/{cell_type}.{chrom}.annot.gz",
    ...             gene_set_file=f"genesets/{cell_type}.GeneSet",
    ...             gene_coord_file="gene_coords.txt",
    ...         )

    >>> # 2. Compute LD scores for each cell type and chromosome
    >>> for cell_type in ["CD8_Naive", "CD4_Memory"]:
    ...     for chrom in range(1, 23):
    ...         compute_ld_scores_with_annotations(
    ...             bfile_prefix=f"1000G/1000G.EUR.QC.{chrom}",
    ...             annot_file=f"annots/{cell_type}.{chrom}.annot.gz",
    ...             out_prefix=f"cts_ldscores/{cell_type}.{chrom}",
    ...             print_snps=f"hapmap3/hm.{chrom}.snp",
    ...         )

    >>> # 3. Create control file
    >>> with open("celltype_ldscores.txt", "w") as f:
    ...     f.write("CD8_Naive\\tcts_ldscores/CD8_Naive.\\n")
    ...     f.write("CD4_Memory\\tcts_ldscores/CD4_Memory.\\n")

    >>> # 4. Run cell-type-specific analysis
    >>> result = estimate_celltype_specific_heritability(
    ...     sumstats_file="disease_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     ref_ld_chr_cts="celltype_ldscores.txt",
    ...     out_prefix="disease_celltype",
    ... )

    Notes
    -----
    - This function performs the final cell-type-specific heritability analysis
    - Requires baseline LD scores and weights (can be downloaded from LDSC resources)
    - The ref_ld_chr_cts file format is critical: tab-separated, cell type name
      then prefix with @ or chromosome numbers appended
    - Tests whether heritability is enriched in genes specific to each cell type
    - Results show coefficient estimates and p-values for each cell type
    - Significant positive coefficients indicate heritability enrichment in that cell type

    See Also
    --------
    compute_ld_scores_with_annotations : Compute LD scores with annotations
    make_annot_from_donor_data : Create cell-type-specific annotations
    munge_sumstats : Prepare GWAS summary statistics
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_file:
        raise ValueError("sumstats_file is required")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")
    if not ref_ld_chr_cts:
        raise ValueError("ref_ld_chr_cts is required")

    cmd = (
        f"{runner.ldsc_command} --h2-cts {sumstats_file} "
        f"--ref-ld-chr {ref_ld_chr} "
        f"--w-ld-chr {w_ld_chr} "
        f"--ref-ld-chr-cts {ref_ld_chr_cts} "
        f"--out {out_prefix}"
    )

    if print_all_cts:
        cmd += " --print-all-cts"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file, ref_ld_chr, w_ld_chr, ref_ld_chr_cts]

    if run:
        logger.info(f"Running cell-type-specific heritability analysis: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)

        return {
            "results_file": f"{out_prefix}.cell_type_results.txt",
            "log_file": f"{out_prefix}.log",
            "files_created": [f"{out_prefix}.cell_type_results.txt", f"{out_prefix}.log"],
        }
    else:
        return {"command": runner._build_container_command(cmd, file_paths)}
