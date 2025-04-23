import subprocess

from pathlib import Path
import cellink as cl

# Setup
DATA = Path(cl.__file__).parent.parent.parent / "data"
genodir = DATA / "eqtl_cat_genotypes"

plinkdir = genodir / "plink"
prunedir = genodir / "prunedir"
pcdir = genodir / "pcdir"
prunedir.mkdir(parents=True, exist_ok=True)
pcdir.mkdir(parents=True, exist_ok=True)

fname = "OneK1K.noGP"
_fname = fname
plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]


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
print(cmd)

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
print(cmd)

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
print(cmd)

ofname = fname + ".filtered.pruned"


# Compute PCs
cmd = plink_base_cmd + [
    "--bfile",
    str(pcdir / ofname),
    "--pca",
    "30",
    "--out",
    str(pcdir / ofname),
]
print(cmd)
subprocess.run(cmd, check=True)
