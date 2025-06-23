import subprocess

import numpy as np
import pandas as pd
from liftover import get_lifter


def preprocess_vcf_to_plink(vcf_filename: str = None, DATA: str = None):
    """Convert VCF file to PLINK format."""
    plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]

    cmd = plink_base_cmd + [
        "--vcf",
        vcf_filename,
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


def plink_filter_prune(fname: str = None, DATA: str = None):  # ="OneK1K.noGP"
    """Filter and prune PLINK dataset."""
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


def plink_kinship(fname: str = None, DATA: str = None):  # ="OneK1K.noGP"
    """Calculate kinship matrix from PLINK dataset."""
    # DATA = Path(cl.__file__).parent.parent.parent / "data" if data_home is None else Path(data_home)
    # genodir = DATA / "eqtl_cat_genotypes"
    # plinkdir = genodir / "plink"
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


converter = get_lifter("hg19", "hg38", one_based=True)

def try_liftover(row):
    try:
        return int(converter[str(row.chrom)][row.pos][0][1])
    except IndexError:
        return np.nan
