import subprocess

import numpy as np
import pandas as pd

from pysrc.paths import DATA

genodir = DATA / "OneK1K_imputation_post_qc_r2_08"

vcfdir = genodir / "filter_vcf_r08"
plinkdir = genodir / "plink_v2"
plinkdir.mkdir(parents=True, exist_ok=True)

plink_base_cmd = ["plink", "--threads", "10", "--keep-allele-order"]
fname = "chr%d.dose.filtered.R2_0.8"

chroms = np.arange(1, 23)


for chrom in chroms:
    _fname = fname % chrom

    cmd = plink_base_cmd + [
        "--vcf",
        str(vcfdir / _fname) + ".vcf.gz",
        "--maf",
        "0.01",
        "--const-fid",
        "--make-bed",
        "--out",
        str(plinkdir / _fname),
    ]
    subprocess.run(cmd, check=True)

    bim = pd.read_csv(plinkdir / f"{_fname}.bim", sep="\t", header=None)
    bim[1] = bim[0].astype(str) + ":" + bim[3].astype(str) + "_" + bim[4] + "_" + bim[5]
    bim.to_csv(plinkdir / f"{_fname}.bim", sep="\t", header=None, index=None)
