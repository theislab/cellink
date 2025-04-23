import subprocess

import numpy as np
import pandas as pd

from pysrc.paths import DATA

# Setup
genodir = DATA / "OneK1K_imputation_post_qc_r2_08"
plinkdir = genodir / "plink_v2"
prunedir = genodir / "prunedir_v2"
pcdir = genodir / "pcdir_v2"
prunedir.mkdir(parents=True, exist_ok=True)
pcdir.mkdir(parents=True, exist_ok=True)

fname = "chr%d.dose.filtered.R2_0.8"
plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]

# Filter variants
chroms = np.arange(1, 23)
for chrom in chroms:
    _fname = fname % chrom

    cmd = plink_base_cmd + [
        "--bfile",
        str(plinkdir / _fname),
        "--geno",
        "0.1",
        "--maf",
        "0.05",
        "--hwe",
        "1e-6",
        "include-nonctrl",
        "--make-bed",
        "--out",
        str(prunedir / f"{_fname}.filtered"),
    ]
    subprocess.run(cmd, check=True)

    # Prune step 1
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{_fname}.filtered"),
        "--indep-pairwise",
        "1000",
        "50",
        "0.1",
        "--out",
        str(pcdir / f"{_fname}.filtered"),
    ]
    subprocess.run(cmd, check=True)

    # Prune step 2
    cmd = plink_base_cmd + [
        "--bfile",
        str(prunedir / f"{_fname}.filtered"),
        "--extract",
        str(pcdir / f"{_fname}.filtered.prune.in"),
        "--make-bed",
        "--out",
        str(pcdir / f"{_fname}.filtered.pruned"),
    ]
    subprocess.run(cmd, check=True)

ofname = fname.replace("chr%d", "wgs") + ".filtered.pruned"
bfiles = np.array([str(pcdir / f"{fname % x}.filtered.pruned") for x in chroms])
bfiles_txt = pcdir / "bfiles.txt"
pd.DataFrame({0: bfiles[1:]}).to_csv(bfiles_txt, index=False, header=False)

cmd = plink_base_cmd + [
    "--bfile",
    bfiles[0],
    "--merge-list",
    str(bfiles_txt),
    "--make-bed",
    "--out",
    str(pcdir / ofname),
]
subprocess.run(cmd, check=True)

# Compute PCs
cmd = plink_base_cmd + [
    "--bfile",
    str(pcdir / ofname),
    "--pca",
    "30",
    "--out",
    str(pcdir / ofname),
]
subprocess.run(cmd, check=True)
