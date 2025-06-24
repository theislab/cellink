import logging
import shutil
import subprocess

import pandas as pd

from cellink._core import DonorData
from cellink.io import to_plink

logger = logging.getLogger(__name__)


def calculate_pcs(
    dd: DonorData,
    prefix: str,
    out: str = None,
    num_pcs: int = 10,
    run: bool = True,
    save_cmd_file: str | None = None,
    plink_export_kwargs: dict | None = {},
) -> pd.DataFrame | str:
    """
    Run PLINK to calculate genetic PCs using the `--pca` option.

    Parameters
    ----------
    dd : DonorData
        Object containing donor-level genotype (`dd.G`) and cell-level expression data (`dd.C`).
    prefix : str
        Prefix to PLINK binary files (.bed/.bim/.fam).
    out : str, optional
        Output file prefix. Defaults to <prefix>_pca.
    num_pcs : int, default=10
        Number of principal components to compute.
    run : bool, default=True
        If True, run the command. If False, return the command string.
    save_cmd_file : str, optional
        If provided, writes the command to a file.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for `to_plink` function.

    Returns
    -------
    pd.DataFrame or str
        PC dataframe if run, or command string if run=False.
    """
    if run and shutil.which("plink") is None:
        raise ImportError("plink is required for `calculate_pcs`. Please install it.")

    if out is None:
        out = f"{prefix}_pca"

    to_plink(dd.G, prefix, **plink_export_kwargs)

    cmd = ["plink", "--bfile", prefix, "--pca", str(num_pcs), "--out", out]

    if run:
        subprocess.run(cmd, check=True)
        pc_df = pd.read_csv(f"{out}.eigenvec", delim_whitespace=True, header=None)
        pc_df.columns = ["FID", "IID"] + [f"PC{i+1}" for i in range(num_pcs)]
        return pc_df
    else:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(" ".join(cmd) + "\n")
        else:
            return " ".join(cmd)
