import logging
import shutil
import subprocess

import pandas as pd

from cellink._core import DonorData
from cellink.io import to_plink

logger = logging.getLogger(__name__)


def calculate_ld(
    dd: DonorData,
    prefix: str,
    out: str = None,
    window_kb: int = 500,
    ld_window: int = 99999,
    r2_threshold: float = 0.2,
    run: bool = True,
    save_cmd_file: str | None = None,
    plink_export_kwargs: dict | None = {},
) -> pd.DataFrame | str:
    """
    Estimate LD patterns using PLINK.

    Parameters
    ----------
    dd : DonorData
        Object containing donor-level genotype (`dd.G`) and cell-level expression data (`dd.C`).
    prefix : str
        PLINK binary file prefix (.bed/.bim/.fam).
    out : str, optional
        Output prefix. Defaults to <prefix>_ld.
    window_kb : int, default=500
        Window size in kilobases.
    ld_window : int, default=99999
        LD window size in number of variants.
    r2_threshold : float, default=0.2
        Minimum r² value to report.
    run : bool, default=True
        Whether to run the command.
    save_cmd_file : str, optional
        Path to save the command string.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for `to_plink` function.

    Returns
    -------
    pd.DataFrame or str
        LD dataframe or command string.
    """
    if run and shutil.which("plink") is None:
        raise ImportError("plink is required for `calculate_pcs`. Please install it.")

    if out is None:
        out = f"{prefix}_ld"

    to_plink(dd.G, prefix, **plink_export_kwargs)

    cmd = [
        "plink",
        "--bfile",
        prefix,
        "--r2",
        "--ld-window-kb",
        str(window_kb),
        "--ld-window",
        str(ld_window),
        "--ld-window-r2",
        str(r2_threshold),
        "--out",
        out,
    ]

    if run:
        subprocess.run(cmd, check=True)
        ld_df = pd.read_csv(f"{out}.ld", delim_whitespace=True)
        return ld_df
    else:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(" ".join(cmd) + "\n")
        else:
            return " ".join(cmd)
