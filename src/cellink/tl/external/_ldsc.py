import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import yaml

from cellink._core import DonorData
from cellink.io import to_plink
from cellink.tl._runner import BaseToolRunner

logger = logging.getLogger(__name__)


class LDSCRunner(BaseToolRunner):
    """
    LDSC Runner with support for local, Docker, and Singularity.

    Configuration keys
    ------------------
    execution_mode : str
        One of ``"local"``, ``"docker"``, ``"singularity"``.
    ldsc_command : str
        Command name / path for ``ldsc.py``.
    make_annot_command : str
        Command name / path for ``make_annot.py``.
    munge_command : str
        Command name / path for ``munge_sumstats.py``.
    docker_image : str, optional
        Docker image name (required when ``execution_mode="docker"``).
    singularity_image : str, optional
        Path to Singularity SIF image (required when
        ``execution_mode="singularity"``).
    """

    def __init__(self, config_path: str | None = None, config_dict: dict | None = None):
        required_fields = ["execution_mode", "ldsc_command", "make_annot_command", "munge_command"]
        prefix_tokens = ["--annot-file", "--out", "--bfile", "--ref-ld-chr", "--w-ld-chr", "--frqfile-chr"]
        super().__init__(config_path, config_dict, required_fields, prefix_tokens)

    def _load_config(self, config_path: str | None, config_dict: dict | None) -> dict:
        if config_dict:
            return config_dict
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            "execution_mode": "local",
            "docker_image": "zijingliu/ldsc",
            "singularity_image": None,
            "ldsc_command": "ldsc.py",
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }

    @property
    def ldsc_command(self) -> str:
        return self.config["ldsc_command"]

    @property
    def make_annot_command(self) -> str:
        return self.config["make_annot_command"]

    @property
    def munge_command(self) -> str:
        return self.config["munge_command"]

    @property
    def execution_mode(self) -> str:
        return self.config["execution_mode"]


_ldsc_runner = None


def configure_ldsc_runner(config_path: str | None = None, config_dict: dict | None = None) -> LDSCRunner:
    global _ldsc_runner
    _ldsc_runner = LDSCRunner(config_path=config_path, config_dict=config_dict)
    return _ldsc_runner


def get_ldsc_runner() -> LDSCRunner:
    global _ldsc_runner
    if _ldsc_runner is None:
        _ldsc_runner = LDSCRunner()
    return _ldsc_runner


def _ensure_out_prefix_dir(out_prefix: str) -> None:
    """Create out_prefix's parent directory if it doesn't exist yet.

    Every ldsc.py/munge_sumstats.py/make_annot.py call writes to `{out_prefix}...`
    and fails with an IOError (or, for singularity/docker, a permission error
    surfaced from inside the container) if that directory doesn't exist. cellink
    builds the container command itself, so it creates the directory upfront.
    """
    parent = os.path.dirname(os.path.abspath(out_prefix))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sniff_delimiter(sumstats_file: str) -> str:
    """
    Detect the field delimiter of a sumstats file from its header line.

    Prefers an explicit single-character delimiter (tab, comma, semicolon)
    over a whitespace-regex fallback, because only an explicit delimiter can
    represent an empty field (two delimiters with nothing between them).
    Whitespace-splitting collapses consecutive separators.
    """
    import csv
    import gzip

    opener = gzip.open if str(sumstats_file).endswith(".gz") else open
    with opener(sumstats_file, "rt") as f:
        header = f.readline()

    try:
        dialect = csv.Sniffer().sniff(header, delimiters="\t,; ")
        if dialect.delimiter in ("\t", ",", ";"):
            return dialect.delimiter
    except csv.Error:
        pass
    return r"\s+"


def _drop_allna_columns(sumstats_file: str, sanitized_path: str, chunksize: int) -> str:
    """
    Write a copy of ``sumstats_file`` with any 100%-missing columns dropped.

    Streams the file in chunks (never holds the whole file in memory) to
    find columns that are entirely NaN/empty, then streams it again to write
    a tab-separated copy without them. 
    """
    sep = _sniff_delimiter(sumstats_file)

    has_value: pd.Series | None = None
    reader = pd.read_csv(sumstats_file, sep=sep, engine="python" if sep == r"\s+" else "c", chunksize=chunksize)
    for chunk in reader:
        chunk_has_value = chunk.notna().any(axis=0)
        has_value = chunk_has_value if has_value is None else (has_value | chunk_has_value)

    if has_value is None:
        raise ValueError(f"'{sumstats_file}' contains no data rows.")

    allna_cols = [c for c in has_value.index if not has_value[c]]
    if not allna_cols:
        return sumstats_file

    logger.warning(
        f"munge_sumstats: dropping {len(allna_cols)} all-missing column(s) {allna_cols} from "
        f"'{sumstats_file}' before munging. munge_sumstats.py's blanket dropna(how='any') "
        "would otherwise drop every row because of these, zeroing out the entire result "
        f"('No objects to concatenate'). Writing sanitized copy to '{sanitized_path}'. "
        "Pass drop_allna_columns=False to disable."
    )

    first = True
    reader = pd.read_csv(sumstats_file, sep=sep, engine="python" if sep == r"\s+" else "c", chunksize=chunksize)
    for chunk in reader:
        chunk.drop(columns=allna_cols).to_csv(sanitized_path, sep="\t", index=False, mode="w" if first else "a", header=first)
        first = False

    return sanitized_path


def _validate_sumstats_pre_munge(
    sumstats_file: str,
    p_col: str | None,
    snp_col: str | None,
    merge_alleles: str | None,
    sample_rows: int = 200_000,
) -> None:
    """
    Pre-flight checks on the input file before invoking the container.

    Raises with a specific message for two input problems that munge_sumstats.py
    itself either "succeeds" through with corrupted output or reports opaquely:

    - a p-value column that is actually -log10(p) (the standard output format of
      REGENIE step 2, which has no untransformed p-value column at all)
    - a SNP-ID column that is not rsIDs while --merge-alleles is requested against
      an rsID-keyed reference (e.g. a UKB/GWAS-Catalog-style chr:pos:ref:alt-only
      file), which otherwise fails with a bare "ValueError: No objects to
      concatenate"

    Also logs upfront counts of indels and duplicate-SNP-ID (multiallelic) rows,
    which munge_sumstats.py folds into generic QC counters ("variants that were
    not SNPs or were strand-ambiguous", "SNPs with duplicated rs numbers").
    """
    sep = _sniff_delimiter(sumstats_file)
    df = pd.read_csv(sumstats_file, sep=sep, engine="python" if sep == r"\s+" else "c", nrows=sample_rows)

    resolved_p_col = p_col if p_col in df.columns else next((c for c in df.columns if c.upper() == "P"), None)
    if resolved_p_col is not None:
        p_vals = pd.to_numeric(df[resolved_p_col], errors="coerce").dropna()
        if len(p_vals) > 0 and (p_vals > 1).mean() > 0.5:
            raise ValueError(
                f"'{resolved_p_col}' does not look like a p-value column: {(p_vals > 1).mean():.0%} of "
                f"sampled values are > 1 (e.g. {p_vals[p_vals > 1].iloc[0]!r}). munge_sumstats.py's "
                "own filter_pvals just checks 0 < P <= 1 and only logs a buried one-line warning rather "
                "than raising, so it would proceed and drop nearly every SNP. This is the standard "
                "shape of a -log10(p) column (e.g. REGENIE step 2's LOG10P, which has no untransformed "
                "p-value column at all). Transform it first with `df[col] = 10 ** -df[col]`, or point "
                "p_col at an untransformed p-value column."
            )

    resolved_snp_col = snp_col if snp_col in df.columns else next((c for c in df.columns if c.upper() == "SNP"), None)
    if resolved_snp_col is not None:
        snp_vals = df[resolved_snp_col].dropna().astype(str)
        if len(snp_vals) > 0:
            looks_like_rsid = snp_vals.str.match(r"^rs\d+$")
            rsid_frac = looks_like_rsid.mean()
            if merge_alleles is not None and rsid_frac < 0.5:
                raise ValueError(
                    f"--merge-alleles was requested against '{merge_alleles}' (an rsID-keyed reference), "
                    f"but only {rsid_frac:.0%} of sampled values in '{resolved_snp_col}' look like "
                    f"rsIDs (e.g. {snp_vals.iloc[0]!r}). munge_sumstats.py's own error for this is "
                    "an opaque 'ValueError: No objects to concatenate' with no indication the actual cause "
                    "is an ID-scheme mismatch. If this file only has chr:pos(:ref:alt)-style variant IDs "
                    "(common for UKB/GWAS-Catalog-format files with no rsID column), you need to look up "
                    "rsIDs for these variants first (e.g. against the reference panel's own bim file by "
                    "position) before merge_alleles can work, or omit merge_alleles and accept the "
                    "reduced strand-ambiguous-SNP filtering."
                )
            has_chr_prefix = snp_vals.str.contains(r"^chr", case=False, regex=True)
            if has_chr_prefix.any() and not has_chr_prefix.all():
                logger.warning(
                    f"munge_sumstats: '{resolved_snp_col}' mixes 'chr'-prefixed and bare IDs "
                    f"({has_chr_prefix.mean():.0%} prefixed in the sampled rows); these will not match "
                    "each other or a differently-conventioned reference panel. Normalize to one "
                    "convention before munging."
                )
            allele_cols = [c for c in df.columns if c.upper() in ("A1", "A2")]
            n_indel = 0
            if allele_cols:
                allele_lens = pd.concat([df[c].dropna().astype(str).str.len() for c in allele_cols])
                n_indel = int((allele_lens > 1).sum())
            n_dup = int(snp_vals.duplicated().sum())
            if n_indel or n_dup:
                logger.info(
                    f"munge_sumstats pre-flight: sampled {len(df)} rows from '{sumstats_file}', found "
                    f"~{n_indel} row(s) with a multi-character allele (indel) and {n_dup} duplicated "
                    f"'{resolved_snp_col}' value(s) (likely multiallelic sites). munge_sumstats.py "
                    "folds indels into its generic 'variants that were not SNPs or were strand-ambiguous' "
                    "count and keeps only one row per duplicated ID (`drop_duplicates(keep='first')`, "
                    "not by significance); these numbers are not broken out separately in its own log."
                )


_MATCH_ALLELES = frozenset(
    {
        "ACAC", "ACCA", "ACGT", "ACTG", "AGAG", "AGCT", "AGGA", "AGTC",
        "CAAC", "CACA", "CAGT", "CATG", "CTAG", "CTCT", "CTGA", "CTTC",
        "GAAG", "GACT", "GAGA", "GATC", "GTAC", "GTCA", "GTGT", "GTTG",
        "TCAG", "TCCT", "TCGA", "TCTC", "TGAC", "TGCA", "TGGT", "TGTG",
    }
)  


def _resolve_col(col: str | None, header_cols: list[str], default_upper: str) -> str:
    if col is not None and col in header_cols:
        return col
    return next(c for c in header_cols if c.upper() == default_upper)


def filter_sumstats_by_merge_alleles(
    sumstats_file: str,
    merge_alleles: str,
    out_path: str,
    snp_col: str | None = None,
    a1_col: str | None = None,
    a2_col: str | None = None,
) -> str:
    """
    Fast, native equivalent of ``munge_sumstats.py --merge-alleles``.

    Filters ``sumstats_file`` down to exactly the rows
    ``munge_sumstats.py --merge-alleles`` keeps: SNPs present in
    ``merge_alleles`` (a reference file with columns ``SNP``, ``A1``, ``A2``)
    whose (A1, A2) allele pair has a valid, non-strand-ambiguous
    correspondence to the reference's pair. Replicates ldsc 1.0.1's
    ``allele_merge`` logic using the same 32-entry valid-pair table
    (:data:`_MATCH_ALLELES`), so running :func:`munge_sumstats` (with
    ``merge_alleles=None``) on this function's output yields the same final SNP
    set as ``munge_sumstats(merge_alleles=...)`` on the original file in
    seconds.

    Parameters
    ----------
    sumstats_file : str
        Path to the sumstats file (raw or already partially processed).
    merge_alleles : str
        Path to a reference file with columns ``SNP``, ``A1``, ``A2`` (e.g.
        from :func:`cellink.resources.get_1000genomes_merge_alleles`).
    out_path : str
        Where to write the filtered file (plain tab-separated text, every
        original column preserved, values unmodified, including A1/A2 case,
        which ``munge_sumstats.py`` uppercases itself downstream).
    snp_col, a1_col, a2_col : str, optional
        Column names in ``sumstats_file`` if not ldsc's defaults (``SNP``,
        ``A1``, ``A2``, resolved case-insensitively).

    Returns
    -------
    str
        ``out_path``.
    """
    sep = _sniff_delimiter(sumstats_file)
    ref_sep = _sniff_delimiter(merge_alleles)

    if sep == r"\s+" or ref_sep == r"\s+":
        ref = pd.read_csv(merge_alleles, sep=ref_sep, engine="python" if ref_sep == r"\s+" else "c")
        ref["__ma_ma__"] = (ref["A1"].astype(str).str.upper() + ref["A2"].astype(str).str.upper())
        ref = ref[["SNP", "__ma_ma__"]]

        header_cols = list(pd.read_csv(sumstats_file, sep=sep, engine="python" if sep == r"\s+" else "c", nrows=0).columns)
        resolved_snp = _resolve_col(snp_col, header_cols, "SNP")
        resolved_a1 = _resolve_col(a1_col, header_cols, "A1")
        resolved_a2 = _resolve_col(a2_col, header_cols, "A2")

        first = True
        for chunk in pd.read_csv(sumstats_file, sep=sep, engine="python" if sep == r"\s+" else "c", chunksize=1_000_000):
            merged = chunk.merge(ref, left_on=resolved_snp, right_on="SNP", how="inner", suffixes=("", "_ref"))
            key = (
                merged[resolved_a1].astype(str).str.upper()
                + merged[resolved_a2].astype(str).str.upper()
                + merged["__ma_ma__"]
            )
            keep = merged[key.isin(_MATCH_ALLELES)][header_cols]
            keep.to_csv(out_path, sep="\t", index=False, mode="w" if first else "a", header=first)
            first = False
        return out_path

    header_cols = pl.scan_csv(sumstats_file, separator=sep, n_rows=0).collect_schema().names()
    resolved_snp = _resolve_col(snp_col, header_cols, "SNP")
    resolved_a1 = _resolve_col(a1_col, header_cols, "A1")
    resolved_a2 = _resolve_col(a2_col, header_cols, "A2")

    ref = pl.scan_csv(merge_alleles, separator=ref_sep).select(
        pl.col("SNP").alias("__ma_snp__"),
        (pl.col("A1").str.to_uppercase() + pl.col("A2").str.to_uppercase()).alias("__ma_ma__"),
    )
    (
        pl.scan_csv(sumstats_file, separator=sep, infer_schema_length=0)
        .join(ref, left_on=resolved_snp, right_on="__ma_snp__", how="inner")
        .filter(
            (
                pl.col(resolved_a1).str.to_uppercase()
                + pl.col(resolved_a2).str.to_uppercase()
                + pl.col("__ma_ma__")
            ).is_in(list(_MATCH_ALLELES))
        )
        .select(header_cols)
        .sink_csv(out_path, separator="\t")
    )
    return out_path


def munge_sumstats(
    sumstats_file: str,
    out_prefix: str = "GWAS_summary_statistics_munged",
    n_samples: int | None = None,
    merge_alleles: str | None = None,
    snplist: str | None = None,
    info_min: float = 0.9,
    maf_min: float = 0.01,
    a1_inc: bool = False,
    signed_sumstats: tuple[str, float] | None = None,
    p_col: str | None = None,
    a1_col: str | None = None,
    a2_col: str | None = None,
    snp_col: str | None = None,
    n_col: str | None = None,
    info_col: str | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    drop_allna_columns: bool = True,
    _allna_chunksize: int = 1_000_000,
    **kwargs,
) -> str | None:
    """
    Munge (clean and standardize) GWAS summary statistics for LDSC analysis

    This function processes raw GWAS summary statistics files to prepare them for
    LD Score regression analysis. It performs quality control, standardizes column
    names, filters SNPs, and aligns alleles to a reference panel.

    Parameters
    ----------
    sumstats_file : str
        Path to input GWAS summary statistics file. Can be plain text or gzipped.
        Should contain columns for SNP ID, effect allele, other allele, and p-value.
    out_prefix : str, default "GWAS_summary_statistics_munged"
        Prefix for output files. Will create {out_prefix}.sumstats.gz
    n_samples : int, optional
        Total sample size. If the summary statistics file has a sample size column,
        this will be used to verify it. If there's no sample size column, this will
        be added to all SNPs.
    merge_alleles : str, optional
        Path to a reference allele file with columns ``SNP A1 A2`` (e.g. the
        classic ``w_hm3.snplist``). This is a filter, not an alignment
        step: ``munge_sumstats.py`` drops any SNP whose (A1, A2) pair
        has no valid correspondence to the reference's pair (including
        strand-ambiguous ones), but every SNP it keeps has its A1/A2/Z
        taken verbatim from ``sumstats_file``.
        cellink never invokes ldsc's own ``--merge-alleles`` flag. Passing ``merge_alleles`` here instead
        runs :func:`filter_sumstats_by_merge_alleles` first (a from-scratch
        reimplementation of the same filtering logic, keeping the identical
        final SNP set), then calls plain ``munge_sumstats.py`` without the
        flag on the result. 
    snplist : str, optional
        Non-functional in the pinned ldsc version (1.0.1); passing it
        raises ``ValueError``. Pass a 3-column file via ``merge_alleles``
        instead.
    info_min : float, default 0.9
        Minimum INFO score for SNP inclusion. SNPs with INFO < info_min are removed.
    maf_min : float, default 0.01
        Minimum minor allele frequency for SNP inclusion. SNPs with MAF < maf_min
        are removed.
    a1_inc : bool, default False
        If True, A1 is the effect allele (increasing allele). If False, A1 is the
        other allele and the sign of the effect will be flipped.
    signed_sumstats : tuple[str, float], optional
        Tuple of (column_name, sign) for identifying the direction of effect.
        Example: ("OR", 1) means odds ratios where values >1 indicate positive effect.
        Example: ("BETA", 0) means betas where values >0 indicate positive effect.
    p_col : str, optional
        Name of the p-value column if non-standard (default: "P")
    a1_col : str, optional
        Name of the effect allele column if non-standard (default: "A1")
    a2_col : str, optional
        Name of the other allele column if non-standard (default: "A2")
    snp_col : str, optional
        Name of the SNP ID column if non-standard (default: "SNP")
    n_col : str, optional
        Name of the sample size column if non-standard (default: "N")
    info_col : str, optional
        Name of the INFO score column if non-standard (default: "INFO")
    drop_allna_columns : bool, default True
        Set to False to pass the file through unmodified.
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to munge_sumstats.py
        Common options include:
        - ignore: List of columns to ignore
        - daner: Set if input is in daner format (PGC)
        - no-alleles: Don't require allele information
        - merge-alleles: Alternative way to specify reference alleles

    Returns
    -------
    dict
        Results dictionary containing:
        - 'sumstats_file': Path to the munged summary statistics file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Raises
    ------
    subprocess.CalledProcessError
        If the munging process fails (e.g., due to malformed input file)

    Examples
    --------
    Basic usage with standard column names:
    >>> result = munge_sumstats(
    ...     sumstats_file="height_gwas.txt.gz",
    ...     out_prefix="height_munged",
    ...     n_samples=253288,
    ...     merge_alleles="w_hm3.snplist",
    ... )

    With custom column names:
    >>> result = munge_sumstats(
    ...     sumstats_file="custom_gwas.txt",
    ...     out_prefix="custom_munged",
    ...     n_samples=50000,
    ...     snp_col="RSID",
    ...     a1_col="EFFECT_ALLELE",
    ...     a2_col="OTHER_ALLELE",
    ...     p_col="PVAL",
    ...     signed_sumstats=("BETA", 0),
    ... )

    Case-control study with odds ratios:
    >>> result = munge_sumstats(
    ...     sumstats_file="case_control_gwas.txt.gz",
    ...     out_prefix="case_control_munged",
    ...     n_samples=10000,
    ...     merge_alleles="w_hm3.snplist",
    ...     signed_sumstats=("OR", 1),
    ...     a1_inc=True,
    ... )

    Just generate the command without running:
    >>> result = munge_sumstats(
    ...     sumstats_file="height_gwas.txt.gz", out_prefix="height_munged", n_samples=253288, run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - The function expects summary statistics files to follow standard GWAS format
    - Strand-ambiguous SNPs (A/T or G/C) are always removed, with or without merge_alleles.
      merge_alleles adds an *additional* reference-correspondence filter on top of this;
      it is not what enables strand-ambiguous filtering in the first place
    - The output file will be gzipped and named {out_prefix}.sumstats.gz
    - merge_alleles only *filters out* SNPs with no valid allele correspondence to the
      reference; it does not rewrite/align the alleles or flip signs of the SNPs it keeps
      (see the merge_alleles parameter above). Using it reduces false strand-ambiguous
      hits but does not by itself guarantee two munged files are in consistent orientation
    - merge_alleles never invokes ldsc's own --merge-alleles flag; see the merge_alleles
      parameter above for what cellink does instead
    - For binary traits, signed_sumstats should typically be ("OR", 1) or ("BETA", 0)
    - For quantitative traits, signed_sumstats is typically ("BETA", 0) or ("Z", 0)
    """
    if snplist is not None:
        raise ValueError(
            "`snplist` does not do a lenient SNP-ID-only filter: munge_sumstats.py (ldsc 1.0.1) has "
            "no standalone '--merge' flag, only '--merge-alleles'. Passing '--merge <file>' on the "
            "command line is resolved by argparse as an abbreviation of '--merge-alleles', so this "
            "parameter would inherit that flag's 3-column 'SNP A1 A2' requirement rather "
            "than filtering by ID alone. Pass a 3-column file via `merge_alleles=` instead."
        )

    if runner is None:
        runner = get_ldsc_runner()

    _ensure_out_prefix_dir(out_prefix)

    if drop_allna_columns:
        sumstats_file = _drop_allna_columns(sumstats_file, f"{out_prefix}.sanitized_input.tsv", _allna_chunksize)

    _validate_sumstats_pre_munge(
        sumstats_file, p_col=p_col, snp_col=snp_col, merge_alleles=merge_alleles
    )

    if merge_alleles is not None:
        sumstats_file = filter_sumstats_by_merge_alleles(
            sumstats_file,
            merge_alleles,
            f"{out_prefix}.merge_alleles_filtered.tsv",
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
        )

    cmd = f"{runner.munge_command} --sumstats {sumstats_file} --out {out_prefix}"

    if n_samples is not None:
        cmd += f" --N {n_samples}"
    if info_min != 0.9:
        cmd += f" --info-min {info_min}"
    if maf_min != 0.01:
        cmd += f" --maf-min {maf_min}"
    if a1_inc:
        cmd += " --a1-inc"

    if signed_sumstats is not None:
        col, min_val = signed_sumstats
        cmd += f" --signed-sumstats {col},{min_val}"
    if p_col is not None:
        cmd += f" --p {p_col}"
    if a1_col is not None:
        cmd += f" --a1 {a1_col}"
    if a2_col is not None:
        cmd += f" --a2 {a2_col}"
    if snp_col is not None:
        cmd += f" --snp {snp_col}"
    if n_col is not None:
        cmd += f" --N-col {n_col}"
    if info_col is not None:
        cmd += f" --info {info_col}"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file]
    if merge_alleles:
        file_paths.append(merge_alleles)
    if snplist:
        file_paths.append(snplist)

    if run:
        logger.info(f"Running munge_sumstats: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.sumstats.gz"
    else:
        return runner._build_container_command(cmd, file_paths)


def _run_ldsc_estimate_ld_scores(
    bfile_prefix: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """Estimate LD Scores from genotype data."""
    if runner is None:
        runner = get_ldsc_runner()

    _ensure_out_prefix_dir(out_prefix)

    cmd = f"{runner.ldsc_command} --bfile {bfile_prefix} --l2 --out {out_prefix}"

    flags = [ld_wind_kb, ld_wind_snp, ld_wind_cm]
    non_null_flags = sum(f is not None for f in flags)

    if non_null_flags > 1:
        raise ValueError("Only one of ld_wind_kb, ld_wind_snp, or ld_wind_cm may be specified.")

    if ld_wind_kb is not None:
        cmd += f" --ld-wind-kb {ld_wind_kb}"
    elif ld_wind_snp is not None:
        cmd += f" --ld-wind-snp {ld_wind_snp}"
    else:
        cmd += f" --ld-wind-cm {ld_wind_cm}"

    if annot_file is not None:
        cmd += f" --annot {annot_file}"
        if thin_annot:
            cmd += " --thin-annot"

    if print_snps is not None:
        cmd += f" --print-snps {print_snps}"

    if maf_min != 0.01:
        cmd += f" --maf {maf_min}"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    cmd += " --yes-really"

    file_paths = [f"{bfile_prefix}.bed", f"{bfile_prefix}.bim", f"{bfile_prefix}.fam"]
    if annot_file:
        file_paths.append(annot_file)
    if print_snps:
        file_paths.append(print_snps)

    if run:
        logger.info(f"Estimating LD scores: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.l2.ldscore.gz"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_ld_scores_from_bimfile(
    bfile_prefix: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate LD scores from PLINK bfile (works with any bfile, including 1000G)

    Parameters
    ----------
    bfile_prefix : str
        Path to PLINK binary files (without .bed/.bim/.fam extension)
    out_prefix : str
        Prefix for output files
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    annot_file : str, optional
        Annotation file for computing category-specific LD scores
    thin_annot : bool, default False
        Thin the annot file by removing columns with <1% SNPs
    print_snps : str, optional
        File with SNP IDs to restrict LD score computation
    maf_min : float, default 0.01
        Minimum MAF threshold
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> # Using 1000G reference panel
    >>> result = estimate_ld_scores_from_bimfile(
    ...     bfile_prefix="1000G_EUR_Phase3_plink/1000G.EUR.QC.22",
    ...     out_prefix="my_ldscores_chr22",
    ...     annot_file="immune_genes.22.annot.gz",
    ...     print_snps="hm3_snps.txt",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    results = {}

    result_file = _run_ldsc_estimate_ld_scores(
        bfile_prefix=bfile_prefix,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        annot_file=annot_file,
        thin_annot=thin_annot,
        print_snps=print_snps,
        maf_min=maf_min,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["ld_scores_file"] = result_file
        results["files_created"] = [
            f"{out_prefix}.l2.ldscore.gz",
            f"{out_prefix}.l2.M",
            f"{out_prefix}.l2.M_5_50",
            f"{out_prefix}.log",
        ]
    else:
        results["command"] = result_file

    return results


def estimate_ld_scores_from_donor_data(
    dd: DonorData,
    out_prefix: str = "ldscores",
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    annot_file: str | None = None,
    thin_annot: bool = False,
    print_snps: str | None = None,
    maf_min: float = 0.01,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate LD scores from DonorData object

    This convenience function exports genotype data from DonorData to PLINK format,
    then computes LD scores.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information
    out_prefix : str, default "ldscores"
        Prefix for output files (also used for temporary PLINK files)
    cleanup_files : bool, default True
        Whether to remove temporary PLINK files after computing LD scores
    plink_export_kwargs : dict, optional
        Additional keyword arguments to pass to to_plink()
    ... (other parameters as in estimate_ld_scores_from_bimfile)

    Returns
    -------
    dict
        Results dictionary (same as estimate_ld_scores_from_bimfile)

    Examples
    --------
    >>> result = estimate_ld_scores_from_donor_data(
    ...     dd=my_donor_data, out_prefix="my_ldscores", annot_file="immune_genes.annot.gz", ld_wind_cm=1.0
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    logger.info("Exporting genotype data to PLINK format for LD score estimation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)

    results = estimate_ld_scores_from_bimfile(
        bfile_prefix=out_prefix,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        annot_file=annot_file,
        thin_annot=thin_annot,
        print_snps=print_snps,
        maf_min=maf_min,
        run=run,
        runner=runner,
        **kwargs,
    )

    if cleanup_files and run:
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def _read_M_line(path: str) -> list[float]:
    with open(path) as f:
        return [float(z) for z in f.readline().split()]


def _combine_one_chr_prefix(
    chr_prefix: str,
    out_prefix: str,
    num_chr: int,
    combine_annot: bool,
    combine_frq: bool,
    combine_ldscore: bool = True,
) -> None:
    """
    Combine one chromosome-split LD score prefix into a single non-chr prefix.
    """
    _ensure_out_prefix_dir(out_prefix)

    if combine_ldscore:
        ld_frames = [pd.read_csv(f"{chr_prefix}{i}.l2.ldscore.gz", sep=r"\s+") for i in range(1, num_chr + 1)]
        ld_combined = pd.concat(ld_frames)[ld_frames[0].columns]
        ld_combined.to_csv(f"{out_prefix}.l2.ldscore.gz", sep="\t", index=False, compression="gzip")

        for suffix in [".l2.M", ".l2.M_5_50"]:
            if not os.path.isfile(f"{chr_prefix}1{suffix}"):
                continue
            total = None
            for i in range(1, num_chr + 1):
                values = _read_M_line(f"{chr_prefix}{i}{suffix}")
                total = values if total is None else [a + b for a, b in zip(total, values)]
            with open(f"{out_prefix}{suffix}", "w") as f:
                f.write("\t".join(str(v) for v in total) + "\n")

    if combine_annot:
        annot_frames = [pd.read_csv(f"{chr_prefix}{i}.annot.gz", sep=r"\s+") for i in range(1, num_chr + 1)]
        annot_combined = pd.concat(annot_frames)[annot_frames[0].columns]
        annot_combined.to_csv(f"{out_prefix}.annot.gz", sep="\t", index=False, compression="gzip")

    if combine_frq:
        frq_frames = [pd.read_csv(f"{chr_prefix}{i}.frq", sep=r"\s+") for i in range(1, num_chr + 1)]
        frq_combined = pd.concat(frq_frames)[frq_frames[0].columns]
        frq_combined.to_csv(f"{out_prefix}.frq", sep="\t", index=False)


def combine_chr_ld_scores(
    chr_prefix: str,
    out_dir: str,
    num_chr: int = 22,
    combine_annot: bool = False,
    combine_frq: bool = False,
    combine_ldscore: bool = True,
) -> str:
    """
    Combine chromosome-split LDSC LD score files into single non-chr-split files.

    Parameters
    ----------
    chr_prefix : str
        Prefix(es) as passed to --ref-ld-chr / --w-ld-chr / --frqfile-chr.
        May be a single prefix or multiple comma-separated prefixes (as used
        with --overlap-annot to combine a baseline panel with a custom
        annotation, e.g. "baselineLD.,my_annotation.").
    out_dir : str
        Directory to write the combined files into. Each comma-separated
        input prefix gets its own combined output file, named after the
        last path component of that prefix.
    num_chr : int, default 22
        Number of chromosomes the input is split across.
    combine_annot : bool, default False
        Also combine the matching .annot.gz files (needed for --overlap-annot).
    combine_frq : bool, default False
        Also combine the matching .frq files (needed for --overlap-annot's
        --frqfile / --frqfile-chr).
    combine_ldscore : bool, default True
        Combine the .l2.ldscore.gz (+ .l2.M/.l2.M_5_50 if present) files.
        Set False when ``chr_prefix`` is a frqfile_chr prefix: allele-frequency
        files have no .l2.ldscore.gz companion, so combining it would try to
        read a file that does not exist for that prefix.

    Returns
    -------
    str
        The new, comma-separated (if the input was) non-chr prefix string,
        ready to pass to --ref-ld / --w-ld / --frqfile.

    Examples
    --------
    >>> combined = combine_chr_ld_scores("baselineLD.,my_annot.", out_dir="combined_ld", combine_annot=True)
    >>> combined
    'combined_ld/baselineLD,combined_ld/my_annot'
    """
    os.makedirs(out_dir, exist_ok=True)
    combined_prefixes = []
    for one_prefix in chr_prefix.split(","):
        name = Path(one_prefix.rstrip(".")).name or "ld"
        out_prefix = os.path.join(out_dir, name)
        _combine_one_chr_prefix(one_prefix, out_prefix, num_chr, combine_annot, combine_frq, combine_ldscore)
        combined_prefixes.append(out_prefix)
    return ",".join(combined_prefixes)


def _run_ldsc_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    combine_chromosomes: bool = False,
    num_chr: int = 22,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: float | None = None,
    pop_prev: float | None = None,
    intercept_h2: float | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """Estimate SNP heritability using LD Score regression."""
    if runner is None:
        runner = get_ldsc_runner()

    _ensure_out_prefix_dir(out_prefix)

    ref_ld_flag, w_ld_flag, frqfile_flag = "--ref-ld-chr", "--w-ld-chr", "--frqfile-chr"
    if combine_chromosomes:
        combined_dir = f"{out_prefix}_combined_ld"
        ref_ld_chr = combine_chr_ld_scores(
            ref_ld_chr, combined_dir, num_chr=num_chr, combine_annot=overlap_annot
        )
        w_ld_chr = combine_chr_ld_scores(w_ld_chr, combined_dir, num_chr=num_chr)
        if frqfile_chr is not None:
            frqfile_chr = combine_chr_ld_scores(
                frqfile_chr, combined_dir, num_chr=num_chr, combine_frq=True, combine_ldscore=False
            )
        ref_ld_flag, w_ld_flag, frqfile_flag = "--ref-ld", "--w-ld", "--frqfile"

    cmd = f"{runner.ldsc_command} --h2 {sumstats_file} {ref_ld_flag} {ref_ld_chr} {w_ld_flag} {w_ld_chr} --out {out_prefix}"

    if overlap_annot:
        cmd += " --overlap-annot"
        if frqfile_chr is None:
            logger.warning(f"--overlap-annot requires {frqfile_flag}")

    if frqfile_chr is not None:
        cmd += f" {frqfile_flag} {frqfile_chr}"

    if not_m_5_50:
        cmd += " --not-M-5-50"

    if print_coefficients:
        cmd += " --print-coefficients"

    if print_delete_vals:
        cmd += " --print-delete-vals"

    if samp_prev is not None:
        cmd += f" --samp-prev {samp_prev}"

    if pop_prev is not None:
        cmd += f" --pop-prev {pop_prev}"

    if intercept_h2 is not None:
        cmd += f" --intercept-h2 {intercept_h2}"

    if no_intercept:
        cmd += " --no-intercept"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file, ref_ld_chr, w_ld_chr]
    if frqfile_chr:
        file_paths.append(frqfile_chr)

    if run:
        logger.info(f"Estimating heritability: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    combine_chromosomes: bool = False,
    num_chr: int = 22,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: float | None = None,
    pop_prev: float | None = None,
    intercept_h2: float | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate SNP heritability using LD Score regression

    Convenience wrapper around run_ldsc_heritability with validation and
    structured output.

    Parameters
    ----------
    sumstats_file : str
        Path to munged summary statistics file (.sumstats.gz)
    ref_ld_chr : str
        Prefix for reference LD scores (with @, e.g., "baseline.")
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights.")
    out_prefix : str
        Prefix for output files
    overlap_annot : bool, default False
        Use overlapping annotation model
    frqfile_chr : str, optional
        Prefix for allele frequency files (required with overlap_annot)
    combine_chromosomes : bool, default False
        Combine the per-chromosome ref_ld_chr/w_ld_chr/frqfile_chr files into
        single non-chr-split files ourselves.
    num_chr : int, default 22
        Number of chromosomes ref_ld_chr/w_ld_chr/frqfile_chr are split across.
        Only used when combine_chromosomes=True.
    not_m_5_50 : bool, default False
        Don't restrict to common SNPs for estimating h2
    print_coefficients : bool, default False
        Print coefficient estimates
    print_delete_vals : bool, default False
        Print delete values
    samp_prev : float, optional
        Sample prevalence (for binary traits)
    pop_prev : float, optional
        Population prevalence (for binary traits)
    intercept_h2 : float, optional
        Constrain the LD Score regression intercept
    no_intercept : bool, default False
        Force intercept to 1
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> result = estimate_heritability(
    ...     sumstats_file="height_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     out_prefix="height_h2",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_file:
        raise ValueError("sumstats_file is required")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")

    results = {}

    result_file = _run_ldsc_heritability(
        sumstats_file=sumstats_file,
        ref_ld_chr=ref_ld_chr,
        w_ld_chr=w_ld_chr,
        out_prefix=out_prefix,
        overlap_annot=overlap_annot,
        frqfile_chr=frqfile_chr,
        combine_chromosomes=combine_chromosomes,
        num_chr=num_chr,
        not_m_5_50=not_m_5_50,
        print_coefficients=print_coefficients,
        print_delete_vals=print_delete_vals,
        samp_prev=samp_prev,
        pop_prev=pop_prev,
        intercept_h2=intercept_h2,
        no_intercept=no_intercept,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["log_file"] = result_file
        results["files_created"] = [f"{out_prefix}.log"]
    else:
        results["command"] = result_file

    return results


def _run_ldsc_genetic_correlation(
    sumstats_files: list[str],
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: list[float] | None = None,
    pop_prev: list[float] | None = None,
    intercept_h2: list[float] | None = None,
    intercept_gencov: list[float] | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """Estimate genetic correlation using LD Score regression."""
    if runner is None:
        runner = get_ldsc_runner()

    _ensure_out_prefix_dir(out_prefix)

    sumstats_str = ",".join(sumstats_files)
    cmd = (
        f"{runner.ldsc_command} --rg {sumstats_str} --ref-ld-chr {ref_ld_chr} --w-ld-chr {w_ld_chr} --out {out_prefix}"
    )

    if overlap_annot:
        cmd += " --overlap-annot"
        if frqfile_chr is None:
            logger.warning("--overlap-annot requires --frqfile-chr")

    if frqfile_chr is not None:
        cmd += f" --frqfile-chr {frqfile_chr}"

    if not_m_5_50:
        cmd += " --not-M-5-50"

    if print_coefficients:
        cmd += " --print-coefficients"

    if print_delete_vals:
        cmd += " --print-delete-vals"

    if samp_prev is not None:
        samp_prev_str = ",".join([str(x) if x is not None else "nan" for x in samp_prev])
        cmd += f" --samp-prev {samp_prev_str}"

    if pop_prev is not None:
        pop_prev_str = ",".join([str(x) if x is not None else "nan" for x in pop_prev])
        cmd += f" --pop-prev {pop_prev_str}"

    if intercept_h2 is not None:
        intercept_h2_str = ",".join([str(x) for x in intercept_h2])
        cmd += f" --intercept-h2 {intercept_h2_str}"

    if intercept_gencov is not None:
        intercept_gencov_str = ",".join([str(x).replace("-", "N") for x in intercept_gencov])
        cmd += f" --intercept-gencov {intercept_gencov_str}"

    if no_intercept:
        cmd += " --no-intercept"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = sumstats_files + [ref_ld_chr, w_ld_chr]
    if frqfile_chr:
        file_paths.append(frqfile_chr)

    if run:
        logger.info(f"Estimating genetic correlation: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return f"{out_prefix}.log"
    else:
        return runner._build_container_command(cmd, file_paths)


def estimate_genetic_correlation(
    sumstats_files: list[str],
    ref_ld_chr: str,
    w_ld_chr: str,
    out_prefix: str,
    overlap_annot: bool = False,
    frqfile_chr: str | None = None,
    not_m_5_50: bool = False,
    print_coefficients: bool = False,
    print_delete_vals: bool = False,
    samp_prev: list[float] | None = None,
    pop_prev: list[float] | None = None,
    intercept_h2: list[float] | None = None,
    intercept_gencov: list[float] | None = None,
    no_intercept: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Estimate genetic correlation using LD Score regression

    Convenience wrapper around run_ldsc_genetic_correlation with validation
    and structured output.

    Parameters
    ----------
    sumstats_files : list[str]
        List of paths to munged summary statistics files (.sumstats.gz)
    ref_ld_chr : str
        Prefix for reference LD scores (with @, e.g., "baseline.")
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights.")
    out_prefix : str
        Prefix for output files
    overlap_annot : bool, default False
        Use overlapping annotation model
    frqfile_chr : str, optional
        Prefix for allele frequency files (required with overlap_annot)
    not_m_5_50 : bool, default False
        Don't restrict to common SNPs
    print_coefficients : bool, default False
        Print coefficient estimates
    print_delete_vals : bool, default False
        Print delete values
    samp_prev : list[float], optional
        Sample prevalences for each trait (use None for quantitative traits)
    pop_prev : list[float], optional
        Population prevalences for each trait
    intercept_h2 : list[float], optional
        Constrain h2 intercepts for each trait
    intercept_gencov : list[float], optional
        Constrain genetic covariance intercepts
    no_intercept : bool, default False
        Force intercepts to 1 and 0
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use
    **kwargs
        Additional arguments passed to ldsc.py

    Returns
    -------
    dict
        Results dictionary with:
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    >>> result = estimate_genetic_correlation(
    ...     sumstats_files=["height_munged.sumstats.gz", "bmi_munged.sumstats.gz"],
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     out_prefix="height_bmi_rg",
    ... )
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_files or len(sumstats_files) < 2:
        raise ValueError("sumstats_files must contain at least 2 files for genetic correlation")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")

    results = {}

    result_file = _run_ldsc_genetic_correlation(
        sumstats_files=sumstats_files,
        ref_ld_chr=ref_ld_chr,
        w_ld_chr=w_ld_chr,
        out_prefix=out_prefix,
        overlap_annot=overlap_annot,
        frqfile_chr=frqfile_chr,
        not_m_5_50=not_m_5_50,
        print_coefficients=print_coefficients,
        print_delete_vals=print_delete_vals,
        samp_prev=samp_prev,
        pop_prev=pop_prev,
        intercept_h2=intercept_h2,
        intercept_gencov=intercept_gencov,
        no_intercept=no_intercept,
        run=run,
        runner=runner,
        **kwargs,
    )

    if run:
        results["log_file"] = result_file
        results["files_created"] = [f"{out_prefix}.log"]
    else:
        results["command"] = result_file

    return results


def _run_ldsc_make_annot(
    bimfile: str,
    annot_file: str,
    gene_set_file: str | None = None,
    gene_coord_file: str | None = None,
    windowsize: int | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> str | None:
    """
    Internal function to create annotation files using make_annot.py

    Either gene_set_file or bed_file must be provided.
    Returns annot_file path if run=True, otherwise command string.
    """
    if runner is None:
        runner = get_ldsc_runner()

    if gene_set_file is None and bed_file is None:
        raise ValueError("Either gene_set_file or bed_file must be provided")

    _ensure_out_prefix_dir(annot_file)

    cmd = f"{runner.make_annot_command} --bimfile {bimfile} --annot-file {annot_file}"

    if gene_set_file is not None:
        cmd += f" --gene-set-file {gene_set_file}"

        if gene_coord_file is not None:
            cmd += f" --gene-coord-file {gene_coord_file}"

        if windowsize is not None:
            cmd += f" --windowsize {windowsize}"

    if bed_file is not None:
        cmd += f" --bed-file {bed_file}"

        if nomerge:
            cmd += " --nomerge"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [bimfile]
    if gene_set_file:
        file_paths.append(gene_set_file)
    if gene_coord_file:
        file_paths.append(gene_coord_file)
    if bed_file:
        file_paths.append(bed_file)

    if run:
        logger.info(f"Creating annotation file: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)
        return annot_file
    else:
        return runner._build_container_command(cmd, file_paths)


def _expand_annot_to_full_format(bimfile: str, annot_file: str) -> None:
    """
    Post-process a make_annot.py output from ANNOT-only to CHR/BP/SNP/CM/ANNOT format.

    make_annot.py writes a single-column file (header: ANNOT, values: 0/1).
    This function reads the matching bimfile, prepends the SNP coordinate columns,
    and rewrites the annotation file in place so binary and continuous annotation
    files share the same format.  Does nothing if already full format (idempotent).
    """
    annot = pd.read_csv(annot_file, sep="\t")
    if annot.shape[1] > 1:
        return
    bim = pd.read_csv(bimfile, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    full = bim[["CHR", "BP", "SNP", "CM"]].copy()
    full["ANNOT"] = annot["ANNOT"].values
    compression = "gzip" if annot_file.endswith(".gz") else None
    full.to_csv(annot_file, sep="\t", index=False, compression=compression)
    logger.info("Expanded annotation to full format: %s", annot_file)


def _normalize_chr_label(chrom: str) -> str:
    """
    Normalize a single chromosome label for cross-file comparison.
    """
    label = str(chrom).strip().upper()
    if label.startswith("CHR"):
        label = label[3:]
    return {"23": "X", "24": "Y", "25": "X", "26": "MT", "M": "MT"}.get(label, label)


def make_annot_from_bimfile(
    bimfile: str,
    annot_file: str,
    gene_set_file: str | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    run: bool = True,
    runner: "LDSCRunner | None" = None,
    scores: "pd.Series | None" = None,
    score_agg: Literal["max", "sum", "mean"] = "max",
    gene_coord_file: str | None = None,
    windowsize: int = 100_000,
    gene_coord_genome_build: str | None = None,
    bim_genome_build: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create a binary or continuous S-LDSC annotation file from a PLINK bimfile.

    Pass ``scores`` for a **continuous** annotation (each SNP gets the aggregated
    per-gene score of overlapping gene windows). Omit ``scores`` and supply
    ``gene_set_file`` or ``bed_file`` for a **binary** (0/1, or an overlap count
    under ``nomerge``) annotation, computed by calling the real ``make_annot.py``.

    Both modes write the same five-column format (CHR, BP, SNP, CM, ANNOT), so
    downstream calls are identical regardless of mode.

    Parameters
    ----------
    bimfile : str
        Path to PLINK .bim file.
    annot_file : str
        Output annotation file. Should end in ``.annot.gz``.
    gene_set_file : str, optional
        *Binary mode.* File of gene names (one per line).
    bed_file : str, optional
        *Binary mode.* UCSC BED file defining annotation regions.
    nomerge : bool, default False
        *Binary mode.* Count overlapping BED intervals instead of merging.
    run : bool, default True
        *Binary mode.* If ``False``, return the command without executing.
    runner : LDSCRunner, optional
        *Binary mode.* LDSC runner (Docker/Singularity/local).
    scores : pd.Series, optional
        *Continuous mode.* Per-gene scores indexed by gene IDs matching
        ``gene_coord_file``. Requires ``gene_coord_file``.
    score_agg : {"max", "sum", "mean"}, default "max"
        *Continuous mode.* Aggregation when multiple gene windows overlap a SNP.
    gene_coord_file : str, optional
        Gene coordinate file. Required in continuous mode; required in binary
        gene-set mode (optional/unused in binary bed-file mode).
        Accepts headed (GENE/CHR/START/END) or headless 4-column format.
    windowsize : int, default 100_000
        Flanking window in bp around each gene body (gene-set binary mode and
        continuous mode only; bed-file mode uses regions as given).
    gene_coord_genome_build : str, optional
        Genome build (e.g. ``"GRCh37"``, ``"GRCh38"``) of ``gene_coord_file`` /
        ``bed_file``. If both this and ``bim_genome_build`` are given and
        differ, raises instead of building a nonsense annotation. Neither is
        required, but pass both: a GRCh38 gene-coordinate file intersected
        against a GRCh37 1000G bim produces a non-empty, plausible-looking
        annotation with the correct gene names matched but every coordinate
        wrong, and that cannot be detected from the output alone.
    bim_genome_build : str, optional
        Genome build of ``bimfile``. See ``gene_coord_genome_build`` above.
    **kwargs
        *Binary mode.* Extra flags forwarded to ``make_annot.py``.

    Returns
    -------
    dict
        Always contains ``annot_file`` and ``files_created``.
        Continuous mode additionally returns ``n_nonzero_snps`` and
        ``n_genes_matched``.

    Examples
    --------
    Binary annotation:

    >>> make_annot_from_bimfile(
    ...     bimfile="1000G.EUR.QC.22.bim",
    ...     annot_file="CD8_Naive.22.annot.gz",
    ...     gene_set_file="CD8_Naive.GeneSet",
    ...     gene_coord_file="gene_coords.txt",
    ...     windowsize=100_000,
    ... )

    Continuous annotation (same downstream call):

    >>> make_annot_from_bimfile(
    ...     bimfile="1000G.EUR.QC.22.bim",
    ...     annot_file="CD8_Naive.22.annot.gz",
    ...     scores=specificity_df["CD8 Naive"],
    ...     gene_coord_file="gene_coords.txt",
    ...     windowsize=100_000,
    ... )
    """
    if gene_coord_genome_build is not None and bim_genome_build is not None:
        if gene_coord_genome_build != bim_genome_build:
            raise ValueError(
                f"Genome build mismatch: gene_coord_file/bed_file is {gene_coord_genome_build!r} but "
                f"bimfile is {bim_genome_build!r}. Intersecting gene/region coordinates from one build "
                "against a bim file from another produces a wrong-but-plausible-looking "
                "annotation (matched gene names, wrong positions) rather than an obvious error. Lift "
                "one of them over to match the other before calling this function."
            )
    elif (gene_coord_file is not None or bed_file is not None) and (
        gene_coord_genome_build is None or bim_genome_build is None
    ):
        logger.warning(
            "make_annot_from_bimfile: gene_coord_genome_build/bim_genome_build were not both provided, "
            "so no genome-build consistency check was performed. If gene_coord_file/bed_file and bimfile "
            "are from different genome builds, this will silently produce a wrong annotation."
        )

    if scores is not None:
        if gene_coord_file is None:
            raise ValueError("gene_coord_file is required for continuous annotations.")
        gene_coords = _load_gene_coord_file(gene_coord_file)
        annot_df = _compute_continuous_annot_for_bimfile(
            bimfile=bimfile,
            scores=scores,
            gene_coords=gene_coords,
            windowsize=windowsize,
            score_agg=score_agg,
        )
        os.makedirs(os.path.dirname(os.path.abspath(annot_file)), exist_ok=True)
        compression = "gzip" if annot_file.endswith(".gz") else None
        annot_df[["CHR", "BP", "SNP", "CM", "ANNOT"]].to_csv(annot_file, sep="\t", index=False, compression=compression)
        n_nonzero = int((annot_df["ANNOT"] != 0).sum())
        chrom = str(annot_df["CHR"].iloc[0])
        chrom_genes = gene_coords[gene_coords["chr"].astype(str) == chrom]
        n_matched = int(chrom_genes["gene"].isin(scores.index.astype(str)).sum())
        logger.info(
            "Wrote continuous annotation: %s (%d non-zero SNPs, %d genes matched)", annot_file, n_nonzero, n_matched
        )
        return {
            "annot_file": annot_file,
            "files_created": [annot_file],
            "n_nonzero_snps": n_nonzero,
            "n_genes_matched": n_matched,
        }

    if gene_set_file is None and bed_file is None:
        raise ValueError("Either scores, gene_set_file, or bed_file must be provided")

    if gene_set_file is not None and gene_coord_file is None:
        raise ValueError("gene_coord_file is required for binary gene-set annotations.")

    if runner is None:
        runner = get_ldsc_runner()
    results = {"annot_file": annot_file, "files_created": []}
    result_file = _run_ldsc_make_annot(
        bimfile=bimfile,
        annot_file=annot_file,
        gene_set_file=gene_set_file,
        gene_coord_file=gene_coord_file,
        windowsize=windowsize,
        bed_file=bed_file,
        nomerge=nomerge,
        run=run,
        runner=runner,
        **kwargs,
    )
    if run:
        _expand_annot_to_full_format(bimfile, annot_file)
        results["annot_file"] = result_file
        results["files_created"].append(annot_file)
    else:
        results["command"] = result_file
    return results


def make_annot_from_donor_data(
    dd: DonorData,
    annot_file: str,
    gene_set_file: str | None = None,
    bed_file: str | None = None,
    nomerge: bool = False,
    run: bool = True,
    runner: "LDSCRunner | None" = None,
    scores: "pd.Series | None" = None,
    score_agg: Literal["max", "sum", "mean"] = "max",
    gene_coord_file: str | None = None,
    windowsize: int = 100_000,
    out_prefix: str = "ldsc_annot",
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create a binary or continuous S-LDSC annotation file from a DonorData object.

    Exports genotype data to PLINK format, then delegates to
    :func:`make_annot_from_bimfile`. Pass ``scores`` for continuous mode or
    ``gene_set_file`` / ``bed_file`` for binary mode.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information.
    annot_file : str
        Output annotation file. Should end in ``.annot.gz``.
    gene_set_file : str, optional
        *Binary mode.* File of gene names, one per line.
    bed_file : str, optional
        *Binary mode.* UCSC BED file defining annotation regions.
    nomerge : bool, default False
        *Binary mode.* Count overlapping BED intervals instead of merging.
    run : bool, default True
        *Binary mode.* If ``False``, return the command without executing.
    runner : LDSCRunner, optional
        *Binary mode.* LDSC runner.
    scores : pd.Series, optional
        *Continuous mode.* Per-gene scores indexed by gene IDs matching
        ``gene_coord_file``.
    score_agg : {"max", "sum", "mean"}, default "max"
        *Continuous mode.* Aggregation rule for overlapping gene windows.
    gene_coord_file : str, optional
        Gene coordinate file. Required in continuous mode.
    windowsize : int, default 100_000
        Flanking window in bp around each gene body.
    out_prefix : str, default "ldsc_annot"
        Prefix for temporary PLINK files created during export.
    cleanup_files : bool, default True
        Remove temporary .bed/.bim/.fam files after writing.
    plink_export_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`~cellink.io.to_plink`.
    **kwargs
        *Binary mode.* Extra flags forwarded to ``make_annot.py``.

    Returns
    -------
    dict
        Same as :func:`make_annot_from_bimfile`.

    Examples
    --------
    Binary annotation:

    >>> make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     gene_set_file="CD8_Naive.GeneSet",
    ...     gene_coord_file="gene_coords.txt",
    ... )

    Continuous annotation:

    >>> make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     scores=specificity_df["CD8 Naive"],
    ...     gene_coord_file="gene_coords.txt",
    ... )
    """
    if plink_export_kwargs is None:
        plink_export_kwargs = {}
    logger.info("Exporting genotype data to PLINK format for annotation creation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)
    bimfile = f"{out_prefix}.bim"

    results = make_annot_from_bimfile(
        bimfile=bimfile,
        annot_file=annot_file,
        gene_set_file=gene_set_file,
        bed_file=bed_file,
        nomerge=nomerge,
        run=run,
        runner=runner,
        scores=scores,
        score_agg=score_agg,
        gene_coord_file=gene_coord_file,
        windowsize=windowsize,
        **kwargs,
    )

    if cleanup_files and (scores is not None or run):
        for ext in [".bim", ".fam", ".bed"]:
            fname = out_prefix + ext
            if os.path.isfile(fname):
                os.remove(fname)
                logger.info("Cleaned up: %s", fname)
    return results


def _colocate_annot_file(annot_file: str, out_prefix: str) -> str | None:
    """
    Ensure the .annot[.gz/.bz2] file is reachable at out_prefix, not just at annot_file.
    """
    marker = ".annot"
    idx = annot_file.rfind(marker)
    if idx == -1:
        return None
    suffix = annot_file[idx:]
    target = out_prefix + suffix
    if os.path.abspath(target) == os.path.abspath(annot_file):
        return None
    if os.path.isfile(target):
        return None

    target_dir = os.path.dirname(target)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    shutil.copyfile(annot_file, target)
    logger.info(f"Copied annotation file to {target} so --overlap-annot can find it alongside the LD scores")
    return target


def compute_ld_scores_with_annotations_from_bimfile(
    bfile_prefix: str,
    annot_file: str,
    out_prefix: str,
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    print_snps: str | None = None,
    thin_annot: bool = False,
    maf_min: float = 0.01,
    yes_really: bool = True,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Compute LD scores with cell-type-specific annotations from PLINK bfile

    This is the first step in cell-type-specific LDSC analysis. It computes
    LD scores for SNPs while incorporating cell-type-specific gene annotations.
    This function should be run for each chromosome and each cell type.

    Works with any PLINK bfile, including standard reference panels like 1000 Genomes.

    Parameters
    ----------
    bfile_prefix : str
        Path to PLINK binary files (without .bed/.bim/.fam extension).
        Typically from 1000 Genomes reference panel, e.g.,
        "1000G_EUR_Phase3_plink/1000G.EUR.QC.22"
    annot_file : str
        Path to the annotation file created by make_annot_from_donor_data()
        or make_annot_from_bimfile(). Should end in .annot.gz
        Example: "CD8_Naive.22.annot.gz"
    out_prefix : str
        Prefix for output files. Will create:
        - {out_prefix}.l2.ldscore.gz (LD scores)
        - {out_prefix}.l2.M (number of SNPs)
        - {out_prefix}.l2.M_5_50 (number of common SNPs)
        - {out_prefix}.log (log file)
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans. Only one of ld_wind_cm, ld_wind_kb,
        or ld_wind_snp can be specified.
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    print_snps : str, optional
        Path to file with SNP IDs (one per row) to restrict LD score computation.
        Commonly used with HapMap3 SNPs (e.g., "hapmap3_snps/hm.22.snp").
        The sum r^2 will still include all SNPs, but only listed SNPs will
        have LD scores computed.
    thin_annot : bool, default False
        Assume annotation files only have annotations (no SNP, CM, CHR, BP columns).
        Should typically be False for annotations created by make_annot_from_bimfile /
        make_annot_from_donor_data, which write the full CHR/BP/SNP/CM/ANNOT format.
    maf_min : float, default 0.01
        Minimum minor allele frequency threshold
    yes_really : bool, default True
        Required flag for computing whole-chromosome LD scores
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage for chromosome 22:
    >>> result = compute_ld_scores_with_annotations_from_bimfile(
    ...     bfile_prefix="1000G_EUR_Phase3_plink/1000G.EUR.QC.22",
    ...     annot_file="CD8_Naive.22.annot.gz",
    ...     out_prefix="CD8_Naive.22",
    ...     print_snps="hapmap3_snps/hm.22.snp",
    ... )

    For all chromosomes (in a loop):
    >>> for chrom in range(1, 23):
    ...     result = compute_ld_scores_with_annotations_from_bimfile(
    ...         bfile_prefix=f"1000G_EUR/1000G.EUR.QC.{chrom}",
    ...         annot_file=f"CD8_Naive.{chrom}.annot.gz",
    ...         out_prefix=f"CD8_Naive.{chrom}",
    ...         print_snps=f"hapmap3_snps/hm.{chrom}.snp",
    ...     )

    Just generate command without running:
    >>> result = compute_ld_scores_with_annotations_from_bimfile(
    ...     bfile_prefix="1000G.EUR.QC.22", annot_file="CD8_Naive.22.annot.gz", out_prefix="CD8_Naive.22", run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - This function is specifically for cell-type-specific analysis workflow
    - Should be run separately for each chromosome (1-22)
    - The annotation file should be created first using make_annot_from_donor_data()
      or make_annot_from_bimfile()
    - print_snps is typically used to restrict to HapMap3 SNPs for better
      matching with standard reference LD scores
    - After computing LD scores for all chromosomes, use
      estimate_celltype_specific_heritability() for the actual analysis

    See Also
    --------
    compute_ld_scores_with_annotations_from_donor_data : Compute from DonorData
    make_annot_from_donor_data : Create annotations from DonorData
    estimate_celltype_specific_heritability : Run cell-type-specific analysis
    """
    if runner is None:
        runner = get_ldsc_runner()

    cmd = f"{runner.ldsc_command} --l2 --bfile {bfile_prefix} --annot {annot_file} --out {out_prefix}"

    flags = [ld_wind_kb, ld_wind_snp, ld_wind_cm]
    non_null_flags = sum(f is not None for f in flags)

    if non_null_flags > 1:
        raise ValueError("Only one of ld_wind_kb, ld_wind_snp, or ld_wind_cm may be specified.")

    if ld_wind_kb is not None:
        cmd += f" --ld-wind-kb {ld_wind_kb}"
    elif ld_wind_snp is not None:
        cmd += f" --ld-wind-snp {ld_wind_snp}"
    else:
        cmd += f" --ld-wind-cm {ld_wind_cm}"

    if thin_annot:
        cmd += " --thin-annot"

    if print_snps is not None:
        cmd += f" --print-snps {print_snps}"

    if maf_min != 0.01:
        cmd += f" --maf {maf_min}"

    if yes_really:
        cmd += " --yes-really"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [f"{bfile_prefix}.bed", f"{bfile_prefix}.bim", f"{bfile_prefix}.fam", annot_file]
    if print_snps:
        file_paths.append(print_snps)

    if run:
        logger.info(f"Computing LD scores with annotations: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)

        colocated_annot = _colocate_annot_file(annot_file, out_prefix)

        files_created = [
            f"{out_prefix}.l2.ldscore.gz",
            f"{out_prefix}.l2.M",
            f"{out_prefix}.l2.M_5_50",
            f"{out_prefix}.log",
        ]
        if colocated_annot is not None:
            files_created.append(colocated_annot)

        return {
            "ld_scores_file": f"{out_prefix}.l2.ldscore.gz",
            "files_created": files_created,
        }
    else:
        return {"command": runner._build_container_command(cmd, file_paths)}


def compute_ld_scores_with_annotations_from_donor_data(
    dd: DonorData,
    annot_file: str,
    out_prefix: str = "ldscores_annot",
    ld_wind_cm: float = 1.0,
    ld_wind_kb: int | None = None,
    ld_wind_snp: int | None = None,
    print_snps: str | None = None,
    thin_annot: bool = False,
    maf_min: float = 0.01,
    yes_really: bool = True,
    cleanup_files: bool = True,
    plink_export_kwargs: dict | None = None,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Compute LD scores with cell-type-specific annotations from DonorData object

    This convenience function exports genotype data from DonorData to PLINK format,
    then computes LD scores with cell-type-specific annotations. This is useful when
    you want to compute LD scores from your own genotype data rather than using a
    reference panel like 1000 Genomes.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype information
    annot_file : str
        Path to the annotation file created by make_annot_from_donor_data()
        or make_annot_from_bimfile(). Should end in .annot.gz
        Example: "CD8_Naive.annot.gz"
    out_prefix : str, default "ldscores_annot"
        Prefix for output files (also used for temporary PLINK files).
        Will create:
        - {out_prefix}.l2.ldscore.gz (LD scores)
        - {out_prefix}.l2.M (number of SNPs)
        - {out_prefix}.l2.M_5_50 (number of common SNPs)
        - {out_prefix}.log (log file)
    ld_wind_cm : float, default 1.0
        LD window size in centiMorgans. Only one of ld_wind_cm, ld_wind_kb,
        or ld_wind_snp can be specified.
    ld_wind_kb : int, optional
        LD window size in kilobases (alternative to ld_wind_cm)
    ld_wind_snp : int, optional
        LD window size in number of SNPs (alternative to ld_wind_cm)
    print_snps : str, optional
        Path to file with SNP IDs (one per row) to restrict LD score computation.
        Commonly used with HapMap3 SNPs (e.g., "hapmap3_snps/hm.22.snp").
    thin_annot : bool, default False
        Assume annotation files only have annotations (no SNP, CM, CHR, BP columns).
        Should typically be False for annotations created by make_annot_from_bimfile /
        make_annot_from_donor_data, which write the full CHR/BP/SNP/CM/ANNOT format.
    maf_min : float, default 0.01
        Minimum minor allele frequency threshold
    yes_really : bool, default True
        Required flag for computing whole-chromosome LD scores
    cleanup_files : bool, default True
        Whether to remove temporary PLINK files after computing LD scores.
        If True, removes {out_prefix}.bed, .bim, and .fam files.
    plink_export_kwargs : dict, optional
        Additional keyword arguments to pass to to_plink()
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'ld_scores_file': Path to LD scores file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     out_prefix="CD8_Naive_ldscores",
    ...     print_snps="hapmap3_snps.txt",
    ... )

    Complete workflow for cell-type analysis:
    >>> # 1. Create annotation from DonorData
    >>> annot_result = make_annot_from_donor_data(
    ...     dd=my_donor_data,
    ...     annot_file="CD8_Naive.annot.gz",
    ...     gene_set_file="CD8_Naive.GeneSet",
    ...     gene_coord_file="gene_coords.txt",
    ...     windowsize=100000,
    ... )

    >>> # 2. Compute LD scores with annotations
    >>> ldsc_result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="CD8_Naive.annot.gz", out_prefix="CD8_Naive_ldscores"
    ... )

    Keep temporary PLINK files:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="immune_genes.annot.gz", out_prefix="my_ldscores", cleanup_files=False
    ... )

    Just generate command:
    >>> result = compute_ld_scores_with_annotations_from_donor_data(
    ...     dd=my_donor_data, annot_file="CD8_Naive.annot.gz", out_prefix="CD8_Naive_ldscores", run=False
    ... )
    >>> print(result["command"])

    Notes
    -----
    - This function exports dd.G to PLINK format, computes LD scores with annotations,
      then optionally cleans up the temporary PLINK files
    - The annotation file must match the SNPs in the DonorData object
    - Typically used when you have your own genotype data and want to compute
      custom LD scores rather than using pre-computed reference LD scores
    - For standard cell-type-specific heritability analysis, it's more common to use
      compute_ld_scores_with_annotations_from_bimfile() with 1000 Genomes data
    - Temporary PLINK files are created in the current directory and cleaned up by
      default, but you can set cleanup_files=False to keep them

    See Also
    --------
    compute_ld_scores_with_annotations_from_bimfile : Compute from existing PLINK files
    make_annot_from_donor_data : Create annotations from DonorData
    estimate_celltype_specific_heritability : Run cell-type-specific analysis
    """
    if runner is None:
        runner = get_ldsc_runner()

    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    logger.info("Exporting genotype data to PLINK format for LD score computation")
    to_plink(dd.G, out_prefix, **plink_export_kwargs)

    results = compute_ld_scores_with_annotations_from_bimfile(
        bfile_prefix=out_prefix,
        annot_file=annot_file,
        out_prefix=out_prefix,
        ld_wind_cm=ld_wind_cm,
        ld_wind_kb=ld_wind_kb,
        ld_wind_snp=ld_wind_snp,
        print_snps=print_snps,
        thin_annot=thin_annot,
        maf_min=maf_min,
        yes_really=yes_really,
        run=run,
        runner=runner,
        **kwargs,
    )

    if cleanup_files and run:
        extensions = [".bim", ".fam", ".bed"]
        for ext in extensions:
            filename = out_prefix + ext
            if os.path.isfile(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")

    return results


def estimate_celltype_specific_heritability(
    sumstats_file: str,
    ref_ld_chr: str,
    w_ld_chr: str,
    ref_ld_chr_cts: str,
    out_prefix: str,
    print_all_cts: bool = False,
    run: bool = True,
    runner: LDSCRunner | None = None,
    **kwargs,
) -> dict[str, Any]:
    r"""
    Estimate cell-type-specific heritability using LD Score regression.

    This is the second step in cell-type-specific LDSC analysis. It tests whether
    SNP heritability is enriched in specific cell types by regressing GWAS summary
    statistics against cell-type-specific LD scores.

    This function requires that LD scores with cell-type annotations have already
    been computed using compute_ld_scores_with_annotations() for all chromosomes.

    Parameters
    ----------
    sumstats_file : str
        Path to munged summary statistics file (.sumstats.gz) from munge_sumstats()
    ref_ld_chr : str
        Prefix for baseline reference LD scores (with @, e.g., "baseline_v1.2/baseline.").
        These are the standard LD scores used for controlling confounders.
    w_ld_chr : str
        Prefix for regression weights (with @, e.g., "weights_hm3_no_hla/weights.").
        These are standard weights files from the LDSC resources.
    ref_ld_chr_cts : str
        Path to control file listing cell-type-specific LD score prefixes.
        This file should have two tab-separated columns per line:
        - Cell type name
        - Prefix for that cell type's LD scores (with @ for chromosome)

        Example file content:
        ```
        CD8_Naive    cts_ldscores/CD8_Naive.
        CD4_Memory   cts_ldscores/CD4_Memory.
        B_cells      cts_ldscores/B_cells.
        ```

        LDSC will look for files like:
        cts_ldscores/CD8_Naive.1.l2.ldscore.gz through
        cts_ldscores/CD8_Naive.22.l2.ldscore.gz
    out_prefix : str
        Prefix for output files. Will create:
        - {out_prefix}.cell_type_results.txt (main results)
        - {out_prefix}.log (log file)
    print_all_cts : bool, default False
        Print results for all cell types (not just significant ones)
    run : bool, default True
        Whether to execute the command or just return it
    runner : LDSCRunner, optional
        Runner instance to use. If None, uses the global runner.
    **kwargs
        Additional command line arguments to pass to ldsc.py

    Returns
    -------
    dict
        Results dictionary containing:
        - 'results_file': Path to cell type results file (if run=True)
        - 'log_file': Path to log file (if run=True)
        - 'files_created': List of created files (if run=True)
        - 'command': Command string (if run=False)

    Examples
    --------
    Basic usage after computing LD scores:
    >>> # First create control file
    >>> with open("celltype_ldscores.txt", "w") as f:
    ...     f.write("CD8_Naive\\tcts_ldscores/CD8_Naive.\\n")
    ...     f.write("CD4_Memory\\tcts_ldscores/CD4_Memory.\\n")
    ...     f.write("B_cells\\tcts_ldscores/B_cells.\\n")

    >>> # Run cell-type-specific analysis
    >>> result = estimate_celltype_specific_heritability(
    ...     sumstats_file="height_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     ref_ld_chr_cts="celltype_ldscores.txt",
    ...     out_prefix="height_celltype_results",
    ... )

    Complete workflow example:
    >>> # 1. Prepare annotations for each cell type and chromosome
    >>> for cell_type in ["CD8_Naive", "CD4_Memory"]:
    ...     for chrom in range(1, 23):
    ...         make_annot_from_donor_data(
    ...             dd=dd_chr,
    ...             annot_file=f"annots/{cell_type}.{chrom}.annot.gz",
    ...             gene_set_file=f"genesets/{cell_type}.GeneSet",
    ...             gene_coord_file="gene_coords.txt",
    ...         )

    >>> # 2. Compute LD scores for each cell type and chromosome
    >>> for cell_type in ["CD8_Naive", "CD4_Memory"]:
    ...     for chrom in range(1, 23):
    ...         compute_ld_scores_with_annotations(
    ...             bfile_prefix=f"1000G/1000G.EUR.QC.{chrom}",
    ...             annot_file=f"annots/{cell_type}.{chrom}.annot.gz",
    ...             out_prefix=f"cts_ldscores/{cell_type}.{chrom}",
    ...             print_snps=f"hapmap3/hm.{chrom}.snp",
    ...         )

    >>> # 3. Create control file
    >>> with open("celltype_ldscores.txt", "w") as f:
    ...     f.write("CD8_Naive\\tcts_ldscores/CD8_Naive.\\n")
    ...     f.write("CD4_Memory\\tcts_ldscores/CD4_Memory.\\n")

    >>> # 4. Run cell-type-specific analysis
    >>> result = estimate_celltype_specific_heritability(
    ...     sumstats_file="disease_munged.sumstats.gz",
    ...     ref_ld_chr="baseline_v1.2/baseline.",
    ...     w_ld_chr="weights_hm3_no_hla/weights.",
    ...     ref_ld_chr_cts="celltype_ldscores.txt",
    ...     out_prefix="disease_celltype",
    ... )

    Notes
    -----
    - This function performs the final cell-type-specific heritability analysis
    - Requires baseline LD scores and weights (can be downloaded from LDSC resources)
    - The ref_ld_chr_cts file format is critical: tab-separated, cell type name
      then prefix with @ or chromosome numbers appended
    - Tests whether heritability is enriched in genes specific to each cell type
    - Results show coefficient estimates and p-values for each cell type
    - Significant positive coefficients indicate heritability enrichment in that cell type

    See Also
    --------
    compute_ld_scores_with_annotations : Compute LD scores with annotations
    make_annot_from_donor_data : Create cell-type-specific annotations
    munge_sumstats : Prepare GWAS summary statistics
    """
    if runner is None:
        runner = get_ldsc_runner()

    if not sumstats_file:
        raise ValueError("sumstats_file is required")
    if not ref_ld_chr:
        raise ValueError("ref_ld_chr is required")
    if not w_ld_chr:
        raise ValueError("w_ld_chr is required")
    if not ref_ld_chr_cts:
        raise ValueError("ref_ld_chr_cts is required")

    cmd = (
        f"{runner.ldsc_command} --h2-cts {sumstats_file} "
        f"--ref-ld-chr {ref_ld_chr} "
        f"--w-ld-chr {w_ld_chr} "
        f"--ref-ld-chr-cts {ref_ld_chr_cts} "
        f"--out {out_prefix}"
    )

    if print_all_cts:
        cmd += " --print-all-cts"

    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{flag}"
        elif value is not None:
            cmd += f" --{flag} {value}"

    file_paths = [sumstats_file, ref_ld_chr, w_ld_chr, ref_ld_chr_cts]

    if run:
        logger.info(f"Running cell-type-specific heritability analysis: {cmd}")
        runner.run_command(cmd, file_paths=file_paths, check=True)

        return {
            "results_file": f"{out_prefix}.cell_type_results.txt",
            "log_file": f"{out_prefix}.log",
            "files_created": [f"{out_prefix}.cell_type_results.txt", f"{out_prefix}.log"],
        }
    else:
        return {"command": runner._build_container_command(cmd, file_paths)}


def _load_gene_coord_file(gene_coord_file: str) -> "pd.DataFrame":
    """Load gene coordinate file in headed (GENE/CHR/START/END) or headless 4-column format."""
    sample = pd.read_csv(gene_coord_file, sep="\t", nrows=1)
    upper_cols = [c.strip().upper() for c in sample.columns]
    if "GENE" in upper_cols and "CHR" in upper_cols:
        df = pd.read_csv(gene_coord_file, sep="\t")
        df.columns = [c.strip().upper() for c in df.columns]
        df = df.rename(columns={"GENE": "gene", "CHR": "chr", "START": "start", "END": "end"})
    else:
        df = pd.read_csv(gene_coord_file, sep="\t", header=None, names=["gene", "chr", "start", "end"])
    # Normalize to bare numeric/X/Y chromosome names (PLINK .bim convention),
    # regardless of whether the source file used a "chr" prefix.
    df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True, case=False)
    return df[["gene", "chr", "start", "end"]]


def _compute_continuous_annot_for_bimfile(
    bimfile: str,
    scores: "pd.Series",
    gene_coords: "pd.DataFrame",
    windowsize: int = 100_000,
    score_agg: Literal["max", "sum", "mean"] = "max",
) -> "pd.DataFrame":
    """
    Compute continuous SNP annotations from per-gene scores for one bimfile.

    Each SNP gets the aggregated score of all genes whose ±windowsize bp window
    overlaps the SNP position.  SNPs with no overlapping gene get 0.

    Returns DataFrame with columns: CHR, BP, SNP, CM, ANNOT.
    """
    bim = pd.read_csv(bimfile, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    chrom = _normalize_chr_label(bim["CHR"].iloc[0])
    scores_idx = scores.copy()
    scores_idx.index = scores_idx.index.astype(str)
    chr_genes = gene_coords[gene_coords["chr"].astype(str).map(_normalize_chr_label) == chrom].copy()
    chr_genes = chr_genes.merge(scores_idx.rename("score").to_frame(), left_on="gene", right_index=True, how="inner")
    chr_genes["win_start"] = chr_genes["start"] - windowsize
    chr_genes["win_end"] = chr_genes["end"] + windowsize
    bp = bim["BP"].values
    score_vals = np.zeros(len(bim), dtype=np.float64)
    if score_agg == "max":
        for _, g in chr_genes.iterrows():
            mask = (bp >= g["win_start"]) & (bp <= g["win_end"])
            if mask.any():
                score_vals[mask] = np.maximum(score_vals[mask], g["score"])
    elif score_agg == "sum":
        for _, g in chr_genes.iterrows():
            mask = (bp >= g["win_start"]) & (bp <= g["win_end"])
            if mask.any():
                score_vals[mask] += g["score"]
    elif score_agg == "mean":
        count_vals = np.zeros(len(bim), dtype=np.float64)
        for _, g in chr_genes.iterrows():
            mask = (bp >= g["win_start"]) & (bp <= g["win_end"])
            if mask.any():
                score_vals[mask] += g["score"]
                count_vals[mask] += 1
        nz = count_vals > 0
        score_vals[nz] /= count_vals[nz]
    else:
        raise ValueError(f"score_agg must be 'max', 'sum', or 'mean', got {score_agg!r}")
    result = bim[["CHR", "BP", "SNP", "CM"]].copy()
    result["ANNOT"] = score_vals
    return result
