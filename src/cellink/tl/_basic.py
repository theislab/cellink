import os
import sys
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

Chromosome = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def calculate_genetic_pc(
    #G=None,
    #genodir=DATA / "OneK1K_imputation_post_qc_r2_08",
    output_prefix: str = None,
    outputdir: str = None,
    #prunedir=None,
    #pcdir=None,
    chromosome: Chromosome | list[Chromosome] | None = None, #chroms=np.arange(1, 23),
    maf_threshold=0.05,
    geno_threshold=0.1,
    hwe_threshold=1e-6,
    indep_pairwise_params=("1000", "50", "0.1"),
    pca_components=30,
    threads=10,
    #output_prefix="wgs",
):
    """
    Processes genotype data by filtering variants, pruning, merging, and performing PCA using PLINK.

    Parameters
    ----------
    genodir : Path, optional
        The directory where genotype data is located. Defaults to `DATA / "OneK1K_imputation_post_qc_r2_08"`.
    plinkdir : Path, optional
        The directory containing the PLINK binary files. If not provided, defaults to `genodir / "plink_v2"`.
    prunedir : Path, optional
        The directory where pruned data will be stored. If not provided, defaults to `genodir / "prunedir_v2"`.
    pcdir : Path, optional
        The directory where PCA results will be stored. If not provided, defaults to `genodir / "pcdir_v2"`.
    chroms : array-like, optional
        The chromosomes to process. Defaults to `np.arange(1, 23)`.
    maf_threshold : float, optional
        The minimum minor allele frequency threshold. Defaults to 0.05.
    geno_threshold : float, optional
        The genotype missingness threshold. Defaults to 0.1.
    hwe_threshold : float, optional
        The Hardy-Weinberg equilibrium threshold. Defaults to 1e-6.
    indep_pairwise_params : tuple, optional
        Parameters for the `--indep-pairwise` pruning step. Defaults to ("1000", "50", "0.1").
    pca_components : int, optional
        The number of principal components to compute. Defaults to 30.
    threads : int, optional
        The number of threads to use. Defaults to 10.
    output_prefix : str, optional
        Prefix for the output filenames. Defaults to "wgs".

    Returns
    -------
    None
        This function processes the data and saves results in the specified directories.
    """

    if outputdir is None:
        outputdir = output_prefix

    os.makedirs(outputdir, exist_ok=True)
    
    fname = "chr%d.dose.filtered.R2_0.8" #TODO CHANGE THIS
    plink_base_cmd = ["plink", "--threads", str(threads), "--keep-allele-order"]

    # Filter variants and prune
    for chr in chromosome:
        _fname = fname % chr

        # Filtering step
        cmd = plink_base_cmd + [
            "--bfile", str(_fname), #output_prefix
            "--geno", str(geno_threshold),
            "--maf", str(maf_threshold),
            "--hwe", str(hwe_threshold),
            #"--include-nonctrl", #TODO WHY REMOVE THIS "include-nonctrl"
            "--make-bed",
            "--out", str(f"{_fname}.filtered"), #prunedir
        ]
        subprocess.run(cmd, check=True)

        # Prune step 1
        cmd = plink_base_cmd + [
            "--bfile", str(prunedir / f"{_fname}.filtered"),
            "--indep-pairwise", *indep_pairwise_params,
            "--out", str(pcdir / f"{_fname}.filtered"),
        ]
        subprocess.run(cmd, check=True)

        # Prune step 2
        cmd = plink_base_cmd + [
            "--bfile", str(prunedir / f"{_fname}.filtered"),
            "--extract", str(pcdir / f"{_fname}.filtered.prune.in"),
            "--make-bed",
            "--out", str(pcdir / f"{_fname}.filtered.pruned"),
        ]
        subprocess.run(cmd, check=True)

    # Merge pruned files
    ofname = fname.replace("chr%d", output_prefix) + ".filtered.pruned"
    bfiles = np.array([str(pcdir / f"{fname % x}.filtered.pruned") for x in chroms])
    bfiles_txt = pcdir / "bfiles.txt"
    pd.DataFrame({0: bfiles[1:]}).to_csv(bfiles_txt, index=False, header=False)

    cmd = plink_base_cmd + [
        "--bfile", bfiles[0],
        "--merge-list", str(bfiles_txt),
        "--make-bed",
        "--out", str(pcdir / ofname),
    ]
    subprocess.run(cmd, check=True)

    # Compute PCA
    cmd = plink_base_cmd + [
        "--bfile", str(pcdir / ofname),
        "--pca", str(pca_components),
        "--out", str(pcdir / ofname),
    ]
    subprocess.run(cmd, check=True)


def run_prune_and_kinship(
    genodir: Path,
    chroms=range(1, 23),
    fname_template="chr%d.dose.filtered.R2_0.8",
    threads=10,
    overwrite=False
):
    """
    Performs variant pruning and kinship matrix computation using PLINK.

    Parameters
    ----------
    genodir : Path
        Base directory where genotype data is stored. Should contain `plink_v2` subdirectory
        with pre-filtered chromosome files.
    chroms : iterable of int, optional
        Chromosomes to process. Defaults to autosomes (1-22).
    fname_template : str, optional
        Template for chromosome-specific filenames. Defaults to "chr%d.dose.filtered.R2_0.8".
    threads : int, optional
        Number of threads to use for PLINK. Defaults to 10.
    overwrite : bool, optional
        If True, existing files will be overwritten. Defaults to False.

    Returns
    -------
    Path
        Path to the final merged and pruned PLINK file prefix (without extensions).
    """
    plinkdir = genodir / "plink_v2"
    prunedir = genodir / "prunedir_v2"
    kinshipdir = genodir / "kinship_v2"

    prunedir.mkdir(parents=True, exist_ok=True)
    kinshipdir.mkdir(parents=True, exist_ok=True)

    plink_base_cmd = ["plink", "--threads", str(threads), "--keep-allele-order"]

    for chrom in chroms:
        fname = fname_template % chrom
        pruned_out = kinshipdir / f"{fname}.filtered"
        pruned_final = kinshipdir / f"{fname}.filtered.pruned"

        if not overwrite and pruned_final.with_suffix(".bed").exists():
            continue

        # Prune step 1
        cmd1 = plink_base_cmd + [
            "--bfile", str(prunedir / f"{fname}.filtered"),
            "--indep-pairwise", "250", "50", "0.2",
            "--out", str(pruned_out)
        ]
        subprocess.run(cmd1, check=True)

        # Prune step 2
        cmd2 = plink_base_cmd + [
            "--bfile", str(prunedir / f"{fname}.filtered"),
            "--extract", str(pruned_out.with_suffix(".prune.in")),
            "--make-bed",
            "--out", str(pruned_final)
        ]
        subprocess.run(cmd2, check=True)

    # Merge chromosomes
    merged_fname = fname_template.replace("chr%d", "wgs") + ".filtered.pruned"
    bfiles = np.array([str(kinshipdir / f"{fname_template % x}.filtered.pruned") for x in chroms])
    bfiles_txt = kinshipdir / "bfiles.txt"
    pd.DataFrame({0: bfiles[1:]}).to_csv(bfiles_txt, index=False, header=False)

    merged_prefix = kinshipdir / merged_fname

    if overwrite or not merged_prefix.with_suffix(".bed").exists():
        merge_cmd = plink_base_cmd + [
            "--bfile", bfiles[0],
            "--merge-list", str(bfiles_txt),
            "--make-bed",
            "--out", str(merged_prefix)
        ]
        subprocess.run(merge_cmd, check=True)

    # Compute kinship matrix
    rel_cmd = plink_base_cmd + [
        "--bfile", str(merged_prefix),
        "--make-rel", "square",
        "--out", str(merged_prefix)
    ]
    subprocess.run(rel_cmd, check=True)

    return merged_prefix
