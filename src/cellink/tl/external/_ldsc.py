import logging
import os
import subprocess
from typing import Any, Literal

import yaml

from cellink._core import DonorData
from cellink.io import to_plink

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

        # Default configuration
        return {
            "execution_mode": "local",
            "docker_image": "zijingliu/ldsc",
            "singularity_image": None,
            "ldsc_command": "ldsc.py",
            "munge_command": "munge_sumstats.py",
        }

    def _validate_config(self):
        """Validate configuration parameters"""
        required_fields = ["execution_mode", "ldsc_command", "munge_command"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if self.config["execution_mode"] not in ["local", "docker", "singularity"]:
            raise ValueError("execution_mode must be 'local', 'docker', or 'singularity'")

    def _infer_volumes_from_paths(self, *file_paths: str) -> dict[str, str]:
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

        # Always include current working directory
        volumes[os.getcwd()] = "/data"

        # Standard cellink data path (if exists)
        cellink_data_path = "/Users/larnoldt/cellink_data"
        if os.path.exists(cellink_data_path):
            volumes[cellink_data_path] = "/cellink_data"

        # Analyze provided file paths
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                # Get absolute path and its parent directory
                abs_path = os.path.abspath(file_path)
                parent_dir = os.path.dirname(abs_path)

                # Check if this path is already covered by existing volumes
                covered = False
                for host_path in volumes.keys():
                    if abs_path.startswith(host_path):
                        covered = True
                        break

                if not covered:
                    # Create a new volume mapping
                    container_path = f"/external_{len(volumes)}"
                    volumes[parent_dir] = container_path

        return volumes

    def _convert_path_to_container(self, file_path: str, volumes: dict[str, str]) -> str:
        """Convert host path to container path"""
        if not file_path:
            return file_path

        abs_path = os.path.abspath(file_path)

        # Find the matching volume
        for host_path, container_path in volumes.items():
            if abs_path.startswith(host_path):
                relative_path = os.path.relpath(abs_path, host_path)
                return os.path.join(container_path, relative_path).replace("\\", "/")

        return file_path  # Return original if no match found

    def _build_container_command(self, base_command: str, volumes: dict[str, str]) -> str:
        """Build docker or singularity command with volumes"""
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

        return base_command

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
            # For local execution, just run the command directly
            result = subprocess.run(base_command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        else:
            # For containerized execution, infer volumes and build container command
            volumes = self._infer_volumes_from_paths(*file_paths)

            # Convert file paths in base_command to container paths
            container_command = base_command
            for host_path, container_path in volumes.items():
                # Simple replacement - might need more sophisticated path handling
                container_command = container_command.replace(host_path, container_path)

            full_command = self._build_container_command(container_command, volumes)

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
    global _ldsc_runner
    _ldsc_runner = LDSCRunner(config_path=config_path, config_dict=config_dict)
    return _ldsc_runner


def get_ldsc_runner() -> LDSCRunner:
    """Get the global LDSC runner instance"""
    global _ldsc_runner
    if _ldsc_runner is None:
        _ldsc_runner = LDSCRunner()  # Use default config
    return _ldsc_runner


def run_ldsc_munge_sumstats(
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

    if run:
        file_paths = [sumstats_file]
        if merge_alleles:
            file_paths.append(merge_alleles)
        if snplist:
            file_paths.append(snplist)

        logger.info(f"Running munge_sumstats: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.sumstats.gz"
    else:
        volumes = runner._infer_volumes_from_paths(sumstats_file, merge_alleles or "", snplist or "")
        return runner._build_container_command(cmd, volumes) if runner.execution_mode != "local" else cmd


def run_ldsc_estimate_ld_scores(
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

    if run:
        file_paths = [f"{bfile_prefix}.bed", f"{bfile_prefix}.bim", f"{bfile_prefix}.fam"]
        if annot_file:
            file_paths.append(annot_file)
        if print_snps:
            file_paths.append(print_snps)

        logger.info(f"Estimating LD scores: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.l2.ldscore.gz"
    else:
        file_paths = [f"{bfile_prefix}.bed", annot_file or "", print_snps or ""]
        volumes = runner._infer_volumes_from_paths(*file_paths)
        return runner._build_container_command(cmd, volumes) if runner.execution_mode != "local" else cmd


def run_ldsc_heritability(
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

    if run:
        file_paths = [sumstats_file, ref_ld_chr, w_ld_chr]
        if frqfile_chr:
            file_paths.append(frqfile_chr)

        logger.info(f"Estimating heritability: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        file_paths = [sumstats_file, ref_ld_chr, w_ld_chr, frqfile_chr or ""]
        volumes = runner._infer_volumes_from_paths(*file_paths)
        return runner._build_container_command(cmd, volumes) if runner.execution_mode != "local" else cmd


def run_ldsc_genetic_correlation(
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

    if run:
        file_paths = sumstats_files + [ref_ld_chr, w_ld_chr]
        if frqfile_chr:
            file_paths.append(frqfile_chr)

        logger.info(f"Estimating genetic correlation: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        file_paths = sumstats_files + [ref_ld_chr, w_ld_chr, frqfile_chr or ""]
        volumes = runner._infer_volumes_from_paths(*file_paths)
        return runner._build_container_command(cmd, volumes) if runner.execution_mode != "local" else cmd


def run_ldsc_cell_type_specific(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    ref_ld_chr_cts: str,
    out_prefix: str,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Run cell-type specific analysis using LDSC
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = f"{runner.ldsc_command} --h2-cts {sumstats_file} --ref-ld-chr {ref_ld_chr} --w-ld-chr {w_ld_chr} --ref-ld-chr-cts {ref_ld_chr_cts} --out {out_prefix}"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    if run:
        file_paths = [sumstats_file, ref_ld_chr, w_ld_chr, ref_ld_chr_cts]

        logger.info(f"Running cell-type specific analysis: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.cell_type_results.txt"
    else:
        file_paths = [sumstats_file, ref_ld_chr, w_ld_chr, ref_ld_chr_cts]
        volumes = runner._infer_volumes_from_paths(*file_paths)
        return runner._build_container_command(cmd, volumes) if runner.execution_mode != "local" else cmd


def run_ldsc_from_donor_data(
    dd: DonorData,
    analysis_type: Literal["heritability", "genetic_correlation", "cell_type_specific", "estimate_ld"] = "heritability",
    sumstats_files: str | list[str] | None = None,
    ref_ld_chr: str | None = None,
    w_ld_chr: str | None = None,
    out_prefix: str = "ldsc_analysis",
    # Heritability/genetic correlation specific
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    samp_prev: float | list[float] | None = None,
    pop_prev: float | list[float] | None = None,
    no_intercept: bool = False,
    # Cell-type specific
    ref_ld_chr_cts: str | None = None,
    # LD estimation specific
    ld_wind_cm: float = 1.0,
    annot_file: str | None = None,
    print_snps: str | None = None,
    # General options
    run: bool = True,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run LDSC analysis from DonorData object
    """
    if runner is None:
        runner = get_ldsc_runner()

    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    results = {"analysis_type": analysis_type, "files_created": []}

    if analysis_type == "estimate_ld":
        # Export genotype data to PLINK format
        logger.info("Exporting genotype data to PLINK format for LD estimation")
        to_plink(dd.G, out_prefix, **plink_export_kwargs)

        # Estimate LD scores
        result_file = run_ldsc_estimate_ld_scores(
            bfile_prefix=out_prefix,
            out_prefix=out_prefix,
            ld_wind_cm=ld_wind_cm,
            annot_file=annot_file,
            print_snps=print_snps,
            run=run,
            runner=runner,
            **kwargs,
        )

        if run:
            results["ld_scores_file"] = result_file
            results["files_created"].extend(
                [f"{out_prefix}.l2.ldscore.gz", f"{out_prefix}.l2.M", f"{out_prefix}.l2.M_5_50", f"{out_prefix}.log"]
            )
        else:
            results["command"] = result_file

    elif analysis_type == "heritability":
        if sumstats_files is None:
            raise ValueError("sumstats_files required for heritability analysis")

        if isinstance(sumstats_files, list):
            sumstats_file = sumstats_files[0]
        else:
            sumstats_file = sumstats_files

        result_file = run_ldsc_heritability(
            sumstats_file=sumstats_file,
            ref_ld_chr=ref_ld_chr,
            w_ld_chr=w_ld_chr,
            out_prefix=out_prefix,
            overlap_annot=overlap_annot,
            frqfile_chr=frqfile_chr,
            samp_prev=samp_prev if not isinstance(samp_prev, list) else samp_prev[0],
            pop_prev=pop_prev if not isinstance(pop_prev, list) else pop_prev[0],
            no_intercept=no_intercept,
            run=run,
            runner=runner,
            **kwargs,
        )

        if run:
            results["log_file"] = result_file
            results["files_created"].extend(
                [
                    f"{out_prefix}.log",
                ]
            )
        else:
            results["command"] = result_file

    elif analysis_type == "genetic_correlation":
        if sumstats_files is None:
            raise ValueError("sumstats_files required for genetic correlation analysis")

        if not isinstance(sumstats_files, list):
            raise ValueError("sumstats_files must be a list for genetic correlation")

        result_file = run_ldsc_genetic_correlation(
            sumstats_files=sumstats_files,
            ref_ld_chr=ref_ld_chr,
            w_ld_chr=w_ld_chr,
            out_prefix=out_prefix,
            overlap_annot=overlap_annot,
            frqfile_chr=frqfile_chr,
            samp_prev=samp_prev,
            pop_prev=pop_prev,
            no_intercept=no_intercept,
            run=run,
            runner=runner,
            **kwargs,
        )

        if run:
            results["log_file"] = result_file
            results["files_created"].extend([f"{out_prefix}.log"])
        else:
            results["command"] = result_file

    elif analysis_type == "cell_type_specific":
        if sumstats_files is None:
            raise ValueError("sumstats_files required for cell-type specific analysis")
        if ref_ld_chr_cts is None:
            raise ValueError("ref_ld_chr_cts required for cell-type specific analysis")

        if isinstance(sumstats_files, list):
            sumstats_file = sumstats_files[0]
        else:
            sumstats_file = sumstats_files

        result_file = run_ldsc_cell_type_specific(
            sumstats_file=sumstats_file,
            ref_ld_chr=ref_ld_chr,
            w_ld_chr=w_ld_chr,
            ref_ld_chr_cts=ref_ld_chr_cts,
            out_prefix=out_prefix,
            run=run,
            runner=runner,
            **kwargs,
        )

        if run:
            results["cell_type_results_file"] = result_file
            results["files_created"].extend([f"{out_prefix}.log", f"{out_prefix}.cell_type_results.txt"])
        else:
            results["command"] = result_file

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    if cleanup_files and analysis_type == "estimate_ld":
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def estimate_heritability_from_sumstats(
    sumstats_file: str,
    out_prefix: str = "heritability",
    ref_ld_chr: str = "eur_w_ld_chr/",
    w_ld_chr: str = "eur_w_ld_chr/",
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for heritability estimation"""
    return {
        "analysis_type": "heritability",
        "command": run_ldsc_heritability(
            sumstats_file=sumstats_file,
            ref_ld_chr=ref_ld_chr,
            w_ld_chr=w_ld_chr,
            out_prefix=out_prefix,
            runner=runner,
            **kwargs,
        ),
    }


def estimate_genetic_correlation(
    sumstats_files: list[str],
    out_prefix: str = "genetic_correlation",
    ref_ld_chr: str = "eur_w_ld_chr/",
    w_ld_chr: str = "eur_w_ld_chr/",
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for genetic correlation estimation"""
    return {
        "analysis_type": "genetic_correlation",
        "command": run_ldsc_genetic_correlation(
            sumstats_files=sumstats_files,
            ref_ld_chr=ref_ld_chr,
            w_ld_chr=w_ld_chr,
            out_prefix=out_prefix,
            runner=runner,
            **kwargs,
        ),
    }


if __name__ == "__main__":
    """
    # =============================================================================
    # 1. SETUP: Create and Use Configuration Files
    # =============================================================================

    # Create configuration files for different execution modes

    # Local execution config
    local_config = create_ldsc_config(
        execution_mode="local",
        ldsc_command="ldsc.py",
        munge_command="munge_sumstats.py",
        config_path="ldsc_local.yaml"
    )

    # Docker execution config
    docker_config = create_ldsc_config(
        execution_mode="docker",
        docker_image="zijingliu/ldsc",
        ldsc_command="/ldsc/ldsc.py",
        munge_command="/ldsc/munge_sumstats.py",
        config_path="ldsc_docker.yaml"
    )

    # Singularity execution config
    singularity_config = create_ldsc_config(
        execution_mode="singularity",
        singularity_image="/path/to/ldsc.sif",
        ldsc_command="./ldsc/ldsc.py",
        munge_command="./ldsc/munge_sumstats.py",
        config_path="ldsc_singularity.yaml"
    )
    """

    # =============================================================================
    # 2. CONFIGURE RUNNER FROM YAML FILE
    # =============================================================================

    # Option 1: Configure from YAML file
    configure_ldsc_runner(config_path="./cellink/tl/external/config/ldsc_docker.yaml")

    """
    # Option 2: Configure from dictionary
    config_dict = {
        'execution_mode': 'docker',
        'docker_image': 'zijingliu/ldsc',
        'ldsc_command': '/ldsc/ldsc.py',
        'munge_command': '/ldsc/munge_sumstats.py'
    }
    configure_ldsc_runner(config_dict=config_dict)


    # Option 3: Create a custom runner for specific analysis
    custom_runner = LDSCRunner(config_path="ldsc_docker.yaml")
    """

    from cellink._core import DonorData

    # =============================================================================
    # 3. LOAD DATA AND SETUP PATHS
    # =============================================================================
    from cellink.io import read_h5_dd

    # Load your DonorData object
    dd = read_h5_dd("onek1k_chr22_hackathon.dd.h5")
    dd.G.obs["donor_id"] = dd.G.obs.index

    # Set up file paths (these will be automatically inferred for volume mounting)

    gwas_summary_statistic_path = "/Users/larnoldt/cellink_data/GCST90018690_summary_stats.csv"

    # =============================================================================
    # 4. MUNGE SUMMARY STATISTICS (with automatic path inference)
    # =============================================================================

    munged_file = run_ldsc_munge_sumstats(
        sumstats_file=gwas_summary_statistic_path,
        out_prefix="GWAS_summary_statistics_munged",
        info_min=0.9,
        maf_min=0.01,
        signed_sumstats=("beta", 0),
        run=True,
        p_col="p_value",
        snp_col="variant_id",
        a1_col="effect_allele",
        a2_col="other_allele",
        n_samples=50000,  # Sample size if not in file
    )

    print(f"Munged summary statistics: {munged_file}")

    # =============================================================================
    # 5. ESTIMATE SNP HERITABILITY (with automatic path inference)
    # =============================================================================

    from cellink.resources import get_1000genomes_ld_scores, get_1000genomes_ld_weights

    ldscores_path, ldscores_prefix = get_1000genomes_ld_scores(population="EUR", return_path=True)
    ldweights_path, ldweights_prefix = get_1000genomes_ld_weights(population="EUR", return_path=True)

    gwas_summary_statistic_path = "/Users/larnoldt/sc-genetics/src/PASS_AgeOfInitiation_Liu2019.sumstats.gz"
    h2_results = run_ldsc_from_donor_data(
        dd=dd,
        analysis_type="heritability",
        sumstats_files=gwas_summary_statistic_path,
        ref_ld_chr=os.path.join(ldscores_path, ldscores_prefix),
        w_ld_chr=os.path.join(ldweights_path, ldweights_prefix),
        out_prefix="PASS_AgeOfInitiation_Liu2019_DSC_ANALYSIS",
        run=True,
    )

    print(f"Heritability results: {h2_results['log_file']}")

    # =============================================================================
    # 7. GENETIC CORRELATIONS (with automatic path inference)
    # =============================================================================

    gwas_summary_statistic_path_2 = "/Users/larnoldt/sc-genetics/src/PASS_LargeArteryStroke_Malik2018.sumstats.gz"

    # Genetic correlation between two traits
    rg_results = run_ldsc_from_donor_data(
        dd=dd,
        analysis_type="genetic_correlation",
        sumstats_files=[
            gwas_summary_statistic_path,
            gwas_summary_statistic_path_2,
        ],
        ref_ld_chr=os.path.join(ldscores_path, ldscores_prefix),
        w_ld_chr=os.path.join(ldweights_path, ldweights_prefix),
        out_prefix="twoGWAS_rg",
        run=True,
    )

    print(f"Genetic correlation results: {rg_results['log_file']}")

    # =============================================================================
    # 9. ESTIMATE LD SCORES FROM YOUR OWN DATA
    # =============================================================================

    # Estimate LD scores from your genotype data - automatic volume inference
    ld_results = run_ldsc_from_donor_data(
        dd=dd,
        analysis_type="estimate_ld",
        ld_wind_cm=1.0,
        print_snps="hm3_no_MHC.list.txt",
        run=True,
        cleanup_files=True,
    )

    print(f"LD scores saved to: {ld_results['ld_scores_file']}")

    """
    22.log
    22.l2.M
    22.l2.M_5_50
    22.l2.ldscore.gz
    """

    # =============================================================================
    # 8. CELL-TYPE SPECIFIC ANALYSIS
    # =============================================================================

    # Cell-type specific analysis - paths automatically inferred
    cts_results = run_ldsc_from_donor_data(
        dd=dd,
        analysis_type="cell_type_specific",
        sumstats_files="schizophrenia_munged.sumstats.gz",
        ref_ld_chr=ldscores_path,
        w_ld_chr=ldweights_path,
        ref_ld_chr_cts="brain_celltype.ldcts",
        out_prefix="schizophrenia_brain_celltypes",
        run=True,
    )

    print(f"Cell-type results: {cts_results['cell_type_results_file']}")
