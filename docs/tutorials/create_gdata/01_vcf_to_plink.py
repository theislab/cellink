import subprocess

import pandas as pd
from pathlib import Path
import cellink as cl

# from pysrc.paths import DATA
DATA = Path(cl.__file__).parent.parent.parent / "data"
genodir = DATA / "eqtl_cat_genotypes"

vcfdir = genodir
plinkdir = genodir / "plink"
plinkdir.mkdir(parents=True, exist_ok=True)

plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]
fname = "OneK1K.noGP"


cmd = plink_base_cmd + [
    "--vcf",
    fname + "vcf.gz",
    "--maf",
    "0.01",
    "--const-fid",
    "--make-bed",
    "--out",
    str(plinkdir / fname),
]
print(cmd)
subprocess.run(cmd, check=True)

bim = pd.read_csv(plinkdir / f"{fname}.bim", sep="\t", header=None)
bim[1] = bim[0].astype(str) + ":" + bim[3].astype(str) + "_" + bim[4] + "_" + bim[5]
bim.to_csv(plinkdir / f"{fname}.bim", sep="\t", header=None, index=None)
