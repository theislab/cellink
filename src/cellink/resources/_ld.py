import shutil
import tarfile
from pathlib import Path

import pandas as pd

from cellink.resources._utils import _download_file, _load_config, get_data_home


def _extract_or_refresh(tgz_path: Path, extract_path: Path, refresh: bool = False) -> None:
    """
    Extract a tar.gz archive into a directory, optionally refreshing its contents.

    If `refresh` is True and the extraction directory exists, all existing contents
    will be deleted before extraction. If the directory is empty or `refresh` is True,
    the tar.gz archive is extracted. Handles nested directories by flattening if necessary.

    Parameters
    ----------
    tgz_path : pathlib.Path
        Path to the `.tar.gz` archive to extract.
    extract_path : pathlib.Path
        Directory where the archive should be extracted.
    refresh : bool, default=False
        If True, deletes existing files/directories in `extract_path` before extraction.

    Returns
    -------
    None
    """
    if refresh and extract_path.exists():
        for item in extract_path.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

    existing_contents = [p for p in extract_path.iterdir() if p != tgz_path]

    if not existing_contents:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

        contents = [p for p in extract_path.iterdir() if p != tgz_path]

        if len(contents) == 1 and contents[0].is_dir():
            nested_dir = contents[0]

            for item in nested_dir.iterdir():
                shutil.move(str(item), str(extract_path))
            nested_dir.rmdir()


def get_1000genomes_ld_scores(
    config_path: str | Path = "./cellink/resources/config/1000genomes.yaml",
    population: str = "EUR",
    data_home: str | Path | None = None,
    return_path: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str] | tuple[Path, str]:
    """
    Download, extract, and load precomputed 1000 Genomes linkage disequilibrium (LD) scores.

    This function downloads population-specific LD scores from the 1000 Genomes project,
    extracts them to a local directory, and concatenates chromosome-wise annotation and
    LD score files into pandas DataFrames.

    Parameters
    ----------
    config_path : str or pathlib.Path, default='./cellink/resources/config/1000genomes.yaml'
        Path to YAML configuration file specifying URLs and file names for LD scores.
    population : str, default='EUR'
        Population code for LD scores. Must be one of {'EUR', 'EAS'}.
    data_home : str or pathlib.Path, optional
        Root directory where data will be stored. Defaults to user-specific cache directory.
    return_path : bool, default=False
        If True, returns the path to the extracted files and file prefix instead of DataFrames.
    refresh : bool, default=False
        If True, re-downloads and re-extracts files even if they already exist locally.

    Returns
    -------
    tuple
        If `return_path=False`, returns `(annot, ldscores, prefix)`:
        - annot : pd.DataFrame
            Concatenated annotation files for all chromosomes.
        - ldscores : pd.DataFrame
            Concatenated LD score files for all chromosomes.
        - prefix : str
            File name prefix used in the extracted data.

        If `return_path=True`, returns `(DATA, prefix)`:
        - DATA : pathlib.Path
            Path to the directory containing extracted files.
        - prefix : str
            File name prefix used in the extracted data.

    Raises
    ------
    ValueError
        If `population` is not one of the populations listed in the configuration.
    """
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_ld_scores_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["ld_scores"]:
        raise ValueError("population must be one of {'EUR', 'EAS'}")

    prefix = config["ld_scores"]["prefix"]
    tgz_path = DATA / config["ld_scores"][population]["filename"]

    _download_file(config["ld_scores"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    if return_path:
        return DATA, prefix

    annots, ldscores = [], []

    for chrom in range(1, 23):
        annot_file = DATA / f"{prefix}{chrom}.annot.gz"
        if annot_file.exists():
            df_annot = pd.read_csv(annot_file, sep="\t")
            df_annot["chrom"] = chrom
            annots.append(df_annot)

        ld_file = DATA / f"{prefix}{chrom}.l2.ldscore.gz"
        if ld_file.exists():
            df_ld = pd.read_csv(ld_file, sep="\t")
            df_ld["chrom"] = chrom
            ldscores.append(df_ld)

    annot = pd.concat(annots, ignore_index=True)
    ldscores = pd.concat(ldscores, ignore_index=True)

    return annot, ldscores, prefix


def get_1000genomes_ld_weights(
    config_path: str | Path = "./cellink/resources/config/1000genomes.yaml",
    population: str = "EUR",
    data_home: str | Path | None = None,
    return_path: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[Path, str]:
    """
    Download, extract, and load precomputed 1000 Genomes LD weights.

    This function downloads population-specific LD weights from the 1000 Genomes project,
    extracts them to a local directory, and concatenates chromosome-wise weight files
    into a single pandas DataFrame.

    Parameters
    ----------
    config_path : str or pathlib.Path, default='./cellink/resources/config/1000genomes.yaml'
        Path to YAML configuration file specifying URLs and file names for LD weights.
    population : str, default='EUR'
        Population code for LD weights. Must be one of {'EUR', 'EAS'}.
    data_home : str or pathlib.Path, optional
        Root directory where data will be stored. Defaults to user-specific cache directory.
    return_path : bool, default=False
        If True, returns the path to the extracted files and file prefix instead of a DataFrame.
    refresh : bool, default=False
        If True, re-downloads and re-extracts files even if they already exist locally.

    Returns
    -------
    tuple
        If `return_path=False`, returns `(None, weights)`:
        - None : placeholder for compatibility with LD scores interface.
        - weights : pd.DataFrame
            Concatenated LD weight files for all chromosomes.

        If `return_path=True`, returns `(DATA, prefix)`:
        - DATA : pathlib.Path
            Path to the directory containing extracted files.
        - prefix : str
            File name prefix used in the extracted data.

    Raises
    ------
    ValueError
        If `population` is not one of the populations listed in the configuration.
    """
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_ld_weights_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["ld_weights"]:
        raise ValueError("population must be one of {'EUR', 'EAS'}")

    prefix = config["ld_weights"]["prefix"]
    tgz_path = DATA / config["ld_weights"][population]["filename"]

    _download_file(config["ld_weights"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    if return_path:
        return DATA, prefix

    weights = []

    for chrom in range(1, 23):
        weight_file = DATA / f"{prefix}{chrom}.l2.ldscore.gz"
        if weight_file.exists():
            df_weight = pd.read_csv(weight_file, sep="\t")
            df_weight["chrom"] = chrom
            weights.append(df_weight)

    weights = pd.concat(weights, ignore_index=True)

    return annot, weights


def get_1000genomes_plink_files(
    config_path: str | Path = "./cellink/resources/config/1000genomes.yaml",
    population: str = "EUR",
    data_home: str | Path | None = None,
    refresh: bool = False,
) -> Path:
    """Download and extract 1000 Genomes PLINK files (BED/BIM/FAM format).

    This function downloads population-specific PLINK files from the 1000 Genomes project,
    extracts them to a local directory, and returns the path to the extracted files.

    Parameters
    ----------
    config_path : str or pathlib.Path, default='./cellink/resources/config/1000genomes.yaml'
        Path to YAML configuration file specifying URLs and file names for PLINK files.
    population : str, default='EUR'
        Population code for PLINK files. Currently only 'EUR' is supported.
    data_home : str or pathlib.Path, optional
        Root directory where data will be stored. Defaults to user-specific cache directory.
    refresh : bool, default=False
        If True, re-downloads and re-extracts files even if they already exist locally.

    Returns
    -------
    - pathlib.Path
        Path to the directory containing extracted PLINK files (.bed, .bim, .fam).
        Files are named as: {prefix}{chrom}.bed/bim/fam where chrom ranges from 1-22.
    - prefix : str
        File name prefix used in the extracted data.

    Raises
    ------
    ValueError
        If `population` is not supported in the configuration.

    Examples
    --------
    >>> plink_dir = get_1000genomes_plink_files(population="EUR")
    >>> # Access chromosome 1 files at:
    >>> # plink_dir / "1000G.EUR.QC.1.bed"
    >>> # plink_dir / "1000G.EUR.QC.1.bim"
    >>> # plink_dir / "1000G.EUR.QC.1.fam"
    """
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_plink_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["plink_files"]:
        raise ValueError(f"population must be one of {list(config['plink_files'].keys())}")

    prefix = config["plink_files"]["prefix"]
    tgz_path = DATA / config["plink_files"][population]["filename"]

    _download_file(config["plink_files"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    return DATA, prefix


def get_1000genomes_frq(
    config_path: str | Path = "./cellink/resources/config/1000genomes.yaml",
    population: str = "EUR",
    data_home: str | Path | None = None,
    return_path: bool = False,
    refresh: bool = False,
) -> tuple[Path, str]:
    """
    Download and extract 1000 Genomes allele frequency files.

    Required for ``ldsc.py --overlap-annot --frqfile-chr``.
    Downloaded from Zenodo record 10515792 (``1000G_Phase3_frq.tgz``).

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to YAML configuration file.
    population : str, default='EUR'
        Population code. Currently only ``'EUR'`` is available.
    data_home : str or pathlib.Path, optional
        Root directory where data will be stored.
    return_path : bool, default=False
        If True, returns ``(DATA, prefix)`` instead of loading DataFrames.
        For frq files, ``return_path=True`` is the typical usage since these
        files are passed as a prefix to ldsc.py rather than loaded into memory.
    refresh : bool, default=False
        If True, re-downloads and re-extracts files even if they already exist.

    Returns
    -------
    tuple
        If ``return_path=True``: ``(DATA, prefix)`` where ``DATA`` is a
        ``Path`` to the directory containing the extracted ``.frq`` files and
        ``prefix`` is the file name prefix (e.g. ``"1000G.EUR.QC."``).

        If ``return_path=False``: ``(DATA, frq_df)`` where ``frq_df`` is a
        concatenated ``pd.DataFrame`` of all per-chromosome frq files.

    Raises
    ------
    ValueError
        If ``population`` is not listed in the configuration.

    Examples
    --------
    >>> frq_dir, frq_prefix = get_1000genomes_frq(population="EUR", return_path=True)
    >>> frqfile_chr = str(frq_dir / frq_prefix)  # passed as --frqfile-chr
    """
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_frq_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["frq"]:
        raise ValueError(f"population must be one of {list(config['frq'].keys())}")

    prefix = config["frq"]["prefix"]
    tgz_path = DATA / config["frq"][population]["filename"]

    _download_file(config["frq"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    return DATA, prefix


def get_1000genomes_hapmap3(
    config_path: str | Path = "./cellink/resources/config/1000genomes.yaml",
    data_home: str | Path | None = None,
    refresh: bool = False,
) -> Path:
    """
    Download the HapMap3 SNP list (no MHC region).

    Used as ``--print-snps`` when computing per-annotation LD scores to
    restrict output to well-imputed HapMap3 SNPs, and as ``--merge-alleles``
    during sumstats munging.

    Downloaded from Zenodo record 10515792 (``hm3_no_MHC.list.txt``).

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to YAML configuration file.
    data_home : str or pathlib.Path, optional
        Root directory where data will be stored.
    refresh : bool, default=False
        If True, re-downloads the file even if it already exists.

    Returns
    -------
    pathlib.Path
        Path to the downloaded ``hm3_no_MHC.list.txt`` file.

    Examples
    --------
    >>> hapmap3_snps = get_1000genomes_hapmap3()
    >>> print_snps = str(hapmap3_snps)  # passed as --print-snps to ldsc.py
    """
    data_home = get_data_home(data_home)
    DATA = data_home / "1000genomes_hapmap3"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    dest = DATA / config["hapmap3"]["filename"]

    _download_file(config["hapmap3"]["url"], dest, checksum=None)

    return dest


def merge_1000g_plink_chromosomes(
    plink_dir: str | Path,
    prefix: str,
    output_prefix: str | Path,
    chromosomes: list[int] | None = None,
    plink_cmd: str = "plink",
) -> Path:
    """
    Merge per-chromosome 1000G PLINK bfiles into a single genome-wide bfile.

    Convenience wrapper for the standard PLINK merge workflow needed before
    passing a reference panel to
    :func:`~cellink.tl.external.write_slurm_array_job`. The resulting
    ``.bed/.bim/.fam`` files can be passed directly as ``plink_bfile``.

    Skips the merge if the output ``.bed`` file already exists.

    Parameters
    ----------
    plink_dir : str or Path
        Directory containing the per-chromosome bfiles, as returned by
        :func:`get_1000genomes_plink_files`.
    prefix : str
        Filename prefix of the per-chromosome files
        (e.g. ``"1000G.EUR.QC."`` so that chromosome 1 is
        ``1000G.EUR.QC.1.bed``).
    output_prefix : str or Path
        Prefix for the merged output (e.g. ``"data/1000G_EUR_merged"``).
        PLINK appends ``.bed``, ``.bim``, ``.fam``.
    chromosomes : list of int, optional
        Chromosomes to include. Defaults to autosomes 1-22.
    plink_cmd : str, default="plink"
        PLINK 1.9 executable name or full path.

    Returns
    -------
    Path
        ``output_prefix`` as a Path (without extension).

    Examples
    --------
    >>> plink_dir, prefix = get_1000genomes_plink_files(population="EUR")
    >>> bfile = merge_1000g_plink_chromosomes(plink_dir, prefix, "data/1000G_EUR_merged")
    >>> # Pass bfile to write_slurm_array_job(plink_bfile=bfile, ...)
    """
    import subprocess
    import tempfile

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    chromosomes = chromosomes or list(range(1, 23))

    if output_prefix.with_suffix(".bed").exists():
        return output_prefix

    plink_dir = Path(plink_dir)
    chr1_bfile = str(plink_dir / f"{prefix}{chromosomes[0]}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        merge_list_path = fh.name
        for chrom in chromosomes[1:]:
            fh.write(str(plink_dir / f"{prefix}{chrom}") + "\n")

    try:
        subprocess.run(
            f"{plink_cmd} --bfile {chr1_bfile} " f"--merge-list {merge_list_path} --make-bed --out {output_prefix}",
            shell=True,
            check=True,
        )
    finally:
        Path(merge_list_path).unlink(missing_ok=True)

    return output_prefix


if __name__ == "__main__":
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EUR")
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EAS")

    annot, weights, prefix = get_1000genomes_ld_weights(population="EUR")
    annot, weights, prefix = get_1000genomes_ld_weights(population="EAS")

    plink_files, prefix = get_1000genomes_plink_files(population="EUR")
    plink_files, prefix = get_1000genomes_plink_files(population="EAS")

    frq, prefix = get_1000genomes_frq(population="EUR")
    frq, prefix = get_1000genomes_frq(population="EAS")

    hapmap3 = get_1000genomes_hapmap3()
