import subprocess

from pathlib import Path
import cellink as cl

# Setup
DATA = Path(cl.__file__).parent.parent.parent / "data"
genodir = DATA / "eqtl_cat_genotypes"


plinkdir = genodir / "plink"
prunedir = genodir / "prunedir"
kinshipdir = genodir / "kinship"
prunedir.mkdir(parents=True, exist_ok=True)
kinshipdir.mkdir(parents=True, exist_ok=True)

fname = "OneK1K.noGP"
_fname = fname
plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]


_fname = fname

# Prune step 1
cmd = plink_base_cmd + [
    "--bfile",
    str(prunedir / f"{_fname}.filtered"),
    "--indep-pairwise",
    "250",
    "50",
    "0.2",
    "--out",
    str(kinshipdir / f"{_fname}.filtered"),
]
subprocess.run(cmd, check=True)

# Prune step 2
cmd = plink_base_cmd + [
    "--bfile",
    str(prunedir / f"{_fname}.filtered"),
    "--extract",
    str(kinshipdir / f"{_fname}.filtered.prune.in"),
    "--make-bed",
    "--out",
    str(kinshipdir / f"{_fname}.filtered.pruned"),
]
subprocess.run(cmd, check=True)

ofname = fname + ".filtered.pruned"

# Compute PCs
cmd = plink_base_cmd + [
    "--bfile",
    str(kinshipdir / ofname),
    "--make-rel",
    "square",
    "--out",
    str(kinshipdir / ofname),
]
subprocess.run(cmd, check=True)
