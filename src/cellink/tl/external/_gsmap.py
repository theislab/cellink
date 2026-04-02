import logging
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def format_gsmap_sumstats(
    sumstats: str | Path | pd.DataFrame,
    out_prefix: str | Path,
    snp: str | None = None,
    a1: str | None = None,
    a2: str | None = None,
    beta: str | None = None,
    se: str | None = None,
    p: str | None = None,
    z: str | None = None,
    n: str | None = None,
    chr_col: str | None = None,
    pos: str | None = None,
    info: str | None = None,
    frq: str | None = None,
    info_min: float = 0.9,
    maf_min: float = 0.01,
    keep_chr_pos: bool = False,
    dbsnp: str | Path | None = None,
    tmp_dir: str | Path | None = None,
    cleanup_tmp: bool = False,
) -> Path:
    """
    Convert GWAS summary statistics into gsMap-compatible format.

    Thin wrapper around ``gsmap format_sumstats`` that accepts a pandas
    DataFrame as input in addition to a file path, handling the temporary
    file creation and column remapping automatically. The output is a
    gzip-compressed file with columns SNP, A1, A2, Z, N, ready to pass
    as ``--sumstats_file`` to any ``gsmap`` subcommand.

    Parameters
    ----------
    sumstats : str, Path, or pd.DataFrame
        Input GWAS summary statistics. If a DataFrame, it is written to a
        temporary file in ``tmp_dir`` before being passed to the CLI.
    out_prefix : str or Path
        Output prefix. The formatted file is written as
        ``{out_prefix}.sumstats.gz``.
    snp : str, optional
        Column name for SNP rs-identifiers.
    a1 : str, optional
        Column name for effect allele.
    a2 : str, optional
        Column name for non-effect allele.
    beta : str, optional
        Column name for GWAS beta coefficient.
    se : str, optional
        Column name for standard error of beta.
    p : str, optional
        Column name for p-value.
    z : str, optional
        Column name for Z-statistic.
    n : str, optional
        Column name for sample size.
    chr_col : str, optional
        Column name for chromosome.
    pos : str, optional
        Column name for base-pair position.
    info : str, optional
        Column name for INFO imputation quality score.
    frq : str, optional
        Column name for allele frequency.
    info_min : float, default=0.9
        Minimum INFO score threshold.
    maf_min : float, default=0.01
        Minimum minor allele frequency threshold.
    keep_chr_pos : bool, default=False
        Retain chromosome and position columns in the output.
    dbsnp : str or Path, optional
        Path to a dbSNP reference file for rs-ID matching.
    tmp_dir : str or Path, optional
        Directory for the temporary text file written when ``sumstats`` is a
        DataFrame. Defaults to the current working directory.
    cleanup_tmp : bool, default=False
        Delete the temporary file after ``gsmap format_sumstats`` finishes.

    Returns
    -------
    Path
        Path to the formatted ``.sumstats.gz`` output file.

    Examples
    --------
    >>> # From a file path
    >>> sumstats_path = format_gsmap_sumstats(
    ...     "GIANT_HEIGHT.txt", out_prefix="height", beta="BETA", se="SE", n="N"
    ... )

    >>> # From a GWAS Catalog DataFrame
    >>> gwas_df['hm_beta'] = pd.to_numeric(gwas_df['hm_beta'], errors='coerce')
    >>> sumstats_path = format_gsmap_sumstats(
    ...     sumstats=gwas_df,
    ...     out_prefix="IQ",
    ...     snp="hm_rsid",
    ...     a1="hm_effect_allele",
    ...     a2="hm_other_allele",
    ...     beta="hm_beta",
    ...     p="p_value",
    ...     n="n",
    ...     tmp_dir="./tmp",
    ...     cleanup_tmp=True,
    ... )
    """
    out_prefix = Path(out_prefix)

    _tmp_to_clean: Path | None = None
    sumstats_path: str

    if isinstance(sumstats, pd.DataFrame):
        _tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path.cwd()
        _tmp_dir.mkdir(parents=True, exist_ok=True)

        # Build the set of columns we actually need based on the flags provided.
        # Writing the full DataFrame risks exposing multiple beta-like columns that gsMap
        # auto-detects, causing "Found 2 different BETA columns" errors. A common example:
        # GWAS Catalog harmonised files contain both 'hm_beta' and a plain 'beta' column
        # with identical values — gsMap sees both as BETA and raises ValueError.
        col_map = {
            snp: snp, a1: a1, a2: a2, beta: beta, se: se,
            p: p, z: z, n: n, chr_col: chr_col, pos: pos, info: info, frq: frq,
        }
        cols_to_keep = [v for v in col_map.values() if v is not None and v in sumstats.columns]
        sumstats = sumstats[cols_to_keep].copy()

        # Coerce beta-like columns to numeric — gsMap computes Z = sign(BETA) * sqrt(chi2.isf(P,1))
        # and fails with a TypeError if BETA contains string values.
        for col in sumstats.columns:
            if col.lower() in {"beta", "hm_beta", "effect_size", "or", "odds_ratio"}:
                sumstats[col] = pd.to_numeric(sumstats[col], errors="coerce")

        _tmp_path = _tmp_dir / f"_gsmap_sumstats_{out_prefix.name}_tmp.txt"
        sumstats.to_csv(_tmp_path, sep="\t", index=False)
        logger.info(f"Wrote temporary sumstats to {_tmp_path}")
        sumstats_path = str(_tmp_path)
        if cleanup_tmp:
            _tmp_to_clean = _tmp_path
    else:
        sumstats_path = str(sumstats)

    cmd = [
        "gsmap", "format_sumstats",
        "--sumstats", sumstats_path,
        "--out", str(out_prefix),
        "--info_min", str(info_min),
        "--maf_min", str(maf_min),
    ]
    for flag, val in {
        "--snp": snp, "--a1": a1, "--a2": a2, "--beta": beta,
        "--se": se, "--p": p, "--z": z, "--n": n,
        "--chr": chr_col, "--pos": pos, "--info": info, "--frq": frq,
    }.items():
        if val is not None:
            cmd += [flag, val]
    if keep_chr_pos:
        cmd.append("--keep_chr_pos")
    if dbsnp is not None:
        cmd += ["--dbsnp", str(dbsnp)]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise RuntimeError(
                f"gsmap format_sumstats failed (exit {result.returncode}):\n{result.stderr}"
            )
    finally:
        if _tmp_to_clean is not None and _tmp_to_clean.exists():
            _tmp_to_clean.unlink()
            logger.info(f"Removed temporary file {_tmp_to_clean}")

    output_file = out_prefix.parent / f"{out_prefix.name}.sumstats.gz"
    logger.info(f"Formatted sumstats written to {output_file}")
    return output_file


def load_gsmap_results(
    workdir: str | Path,
    sample_name: str,
    trait_name: str,
    annotation: str | None = None,
) -> dict:
    """
    Load gsMap output files into DataFrames.

    Reads the spot-level spatial LDSC results and the region-level Cauchy
    combination results written by ``gsmap quick_mode`` or the individual
    pipeline steps.

    Parameters
    ----------
    workdir : str or Path
        Working directory used when running gsMap (``--workdir``).
    sample_name : str
        Sample name used when running gsMap (``--sample_name``).
    trait_name : str
        Trait name used when running gsMap (``--trait_name``).
    annotation : str, optional
        Annotation column name. Required to locate Cauchy combination results.

    Returns
    -------
    dict with keys:
        - ``"spatial_ldsc"`` : pd.DataFrame or None
            Per-spot LDSC results with columns spot, beta, se, z, p.
        - ``"cauchy_combination"`` : pd.DataFrame or None
            Region-level aggregated p-values (p_cauchy, p_median).
        - ``"workdir"`` : Path
        - ``"report_path"`` : Path or None

    Examples
    --------
    >>> results = load_gsmap_results(
    ...     workdir="./gsmap_output",
    ...     sample_name="V1_Adult_Mouse_Brain_Coronal",
    ...     trait_name="IQ",
    ...     annotation="domain",
    ... )
    >>> spot_df   = results["spatial_ldsc"]
    >>> cauchy_df = results["cauchy_combination"]
    """
    workdir = Path(workdir)
    sample_dir = workdir / sample_name

    results: dict = {
        "spatial_ldsc": None,
        "cauchy_combination": None,
        "workdir": workdir,
        "report_path": None,
    }

    # Spatial LDSC — gsMap writes comma-separated .csv.gz
    ldsc_dir = sample_dir / "spatial_ldsc"
    if ldsc_dir.exists():
        ldsc_files = list(ldsc_dir.glob(f"*{trait_name}*.gz")) or list(ldsc_dir.glob("*.gz"))
        if ldsc_files:
            try:
                _df = pd.read_csv(ldsc_files[0], sep=",", compression="gzip")
                # Fall back to tab if the file turns out to be tab-separated
                if _df.shape[1] == 1:
                    _df = pd.read_csv(ldsc_files[0], sep="\t", compression="gzip")
                results["spatial_ldsc"] = _df
                logger.info(f"Loaded spatial LDSC results from {ldsc_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load spatial LDSC results: {e}")

    # Cauchy combination
    cauchy_dir = sample_dir / "cauchy_combination"
    if cauchy_dir.exists() and annotation is not None:
        cauchy_files = (
            list(cauchy_dir.glob(f"*{trait_name}*.csv*")) or list(cauchy_dir.glob("*.csv*"))
        )
        if cauchy_files:
            try:
                results["cauchy_combination"] = pd.read_csv(cauchy_files[0], index_col=0)
                logger.info(f"Loaded Cauchy combination results from {cauchy_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load Cauchy combination results: {e}")

    # Report
    report_dir = sample_dir / "report"
    if report_dir.exists():
        results["report_path"] = report_dir
        logger.info(f"Report available at: {report_dir}")

    return results