import functools
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@functools.cache
def _get_hg38_to_hg19_converter():
    """Load the hg38 to hg19 liftover chain on first use (None if ``liftover`` is unavailable).

    The chain is cached in ``$CELLINK_LIFTOVER_CACHE``, by default ``~/.liftover``.
    """
    try:
        from liftover import get_lifter

        cache = os.environ.get("CELLINK_LIFTOVER_CACHE", os.path.expanduser("~/.liftover"))
        return get_lifter("hg38", "hg19", one_based=True, cache=cache)
    except Exception as e:  # noqa: BLE001
        logger.warning("liftover unavailable (%s); hg38 to hg19 coordinate translation is disabled.", e)
        return None


def preprocess_vcf_to_plink(vcf_filename: str = None, DATA: str | Path = None) -> None:
    """
    Convert a VCF file to PLINK binary format (BED/BIM/FAM).

    This function uses PLINK to filter SNPs with minor allele frequency (MAF) < 0.01,
    keeps the original allele order, and writes the resulting PLINK files to the specified
    data directory. The BIM file is additionally updated to include a combined variant
    identifier in the format `chrom:pos_ref_alt`.

    Parameters
    ----------
    vcf_filename : str
        Path to the input VCF file (.vcf.gz) to convert.
    DATA : str or Path
        Directory to store the output PLINK files.

    Returns
    -------
    None
        Outputs PLINK `.bed`, `.bim`, and `.fam` files to the specified directory.
    """
    DATA = Path(DATA)

    plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]

    cmd = plink_base_cmd + [
        "--vcf",
        str(DATA / vcf_filename),
        "--maf",
        "0.01",
        "--const-fid",
        "--make-bed",
        "--out",
        str(DATA / vcf_filename.replace(".vcf.gz", "")),
    ]
    subprocess.run(cmd, check=True)

    bim = pd.read_csv(DATA / f"{vcf_filename.replace('.vcf.gz', '')}.bim", sep="\t", header=None)
    bim[1] = bim[0].astype(str) + ":" + bim[3].astype(str) + "_" + bim[4] + "_" + bim[5]
    bim.to_csv(DATA / f"{vcf_filename.replace('.vcf.gz', '')}.bim", sep="\t", header=None, index=None)


def plink_filter_prune(fname: str = None, DATA: str | Path = None) -> None:
    """
    Perform quality control filtering and LD-based pruning on a PLINK dataset.

    Steps:
    1. Filter SNPs by genotype missingness (>10%), MAF (<5%), and Hardy-Weinberg equilibrium (p < 1e-6).
    2. Perform LD-based pruning in two steps using sliding windows.
    3. Compute the top 30 principal components (PCs) on the pruned dataset.

    Parameters
    ----------
    fname : str
        Base filename of the input PLINK dataset (without file extensions).
    DATA : str or Path
        Directory containing the PLINK dataset and where outputs will be written.

    Returns
    -------
    None
        Filtered, pruned PLINK files and PCA results are saved under `prunedir` and `pcdir`.
    """
    DATA = Path(DATA)

    prunedir = DATA / "prunedir"
    pcdir = DATA / "pcdir"

    prunedir.mkdir(parents=True, exist_ok=True)
    pcdir.mkdir(parents=True, exist_ok=True)

    plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]

    cmd = plink_base_cmd + [
        "--bfile",
        str(DATA / fname),
        "--geno",
        "0.1",
        "--maf",
        "0.05",
        "--hwe",
        "1e-6",
        "--make-bed",
        "--out",
        str(prunedir / f"{fname}.filtered"),
    ]
    subprocess.run(cmd, check=True)

    # Step 2: Prune step 1
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{fname}.filtered"),
        "--indep-pairwise",
        "1000",
        "50",
        "0.1",
        "--out",
        str(pcdir / f"{fname}.filtered"),
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Prune step 2
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{fname}.filtered"),
        "--extract",
        str(pcdir / f"{fname}.filtered.prune.in"),
        "--make-bed",
        "--out",
        str(pcdir / f"{fname}.filtered.pruned"),
    ]
    subprocess.run(cmd, check=True)

    # Step 4: Compute Principal Components (PCs)
    cmd = plink_base_cmd + [
        "--bfile",
        str(pcdir / f"{fname}.filtered.pruned"),
        "--pca",
        "30",
        "--out",
        str(pcdir / f"{fname}.filtered.pruned"),
    ]
    subprocess.run(cmd, check=True)


def plink_kinship(fname: str = None, DATA: str | Path = None) -> None:
    """
    Compute a kinship matrix from a PLINK dataset.

    This function performs pruning of SNPs to reduce LD, then calculates a pairwise
    kinship matrix using the `--make-rel square` option in PLINK. Outputs are stored
    in a dedicated `kinship` directory.

    Parameters
    ----------
    fname : str
        Base filename of the input PLINK dataset (without file extensions).
    DATA : str or Path
        Directory containing the filtered PLINK files and where the kinship matrix will be stored.

    Returns
    -------
    None
        Produces a kinship matrix file (`.rel`) along with pruned PLINK files in the `kinship` directory.
    """
    # DATA = Path(cl.__file__).parent.parent.parent / "data" if data_home is None else Path(data_home)
    # genodir = DATA / "eqtl_cat_genotypes"
    # plinkdir = genodir / "plink"
    DATA = Path(DATA)

    prunedir = DATA / "prunedir"
    kinshipdir = DATA / "kinship"

    prunedir.mkdir(parents=True, exist_ok=True)
    kinshipdir.mkdir(parents=True, exist_ok=True)

    plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]

    # Step 1: Prune for kinship
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{fname}.filtered"),
        "--indep-pairwise",
        "250",
        "50",
        "0.2",
        "--out",
        str(kinshipdir / f"{fname}.filtered"),
    ]
    subprocess.run(cmd, check=True)

    # Step 2: Apply pruning
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{fname}.filtered"),
        "--extract",
        str(kinshipdir / f"{fname}.filtered.prune.in"),
        "--make-bed",
        "--out",
        str(kinshipdir / f"{fname}.filtered.pruned"),
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Compute kinship matrix
    cmd = plink_base_cmd + [
        "--bfile",
        str(kinshipdir / f"{fname}.filtered.pruned"),
        "--make-rel",
        "square",
        "--out",
        str(kinshipdir / f"{fname}.filtered.pruned"),
    ]
    subprocess.run(cmd, check=True)


def try_liftover(row) -> int | float:
    """
    Attempt to lift over a genomic coordinate from hg38 (GRCh38) to hg19 (GRCh37).

    Uses the `liftover` module to convert a chromosome and position from hg38 to
    hg19, as needed for the GRCh38 OneK1K genotypes. Returns NaN if conversion
    fails, if the position maps to another chromosome, or if the liftover
    converter is unavailable.

    Parameters
    ----------
    row : pandas.Series
        A row containing `chrom` and `pos` fields representing the hg38 coordinate.

    Returns
    -------
    int or float
        The converted hg19 position if successful, or `np.nan` if conversion fails.
    """
    converter = _get_hg38_to_hg19_converter()
    if converter is None:
        return np.nan
    chrom = str(row.chrom).removeprefix("chr")
    try:
        hits = converter[chrom][int(row.pos)]
    except (KeyError, TypeError, ValueError):
        return np.nan
    if not hits or str(hits[0][0]).removeprefix("chr") != chrom:
        return np.nan
    return int(hits[0][1])
