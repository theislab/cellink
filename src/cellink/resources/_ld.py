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
    """
    Download and extract 1000 Genomes PLINK files (BED/BIM/FAM format).
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


if __name__ == "__main__":
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EUR")
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EAS")

    annot, weights, prefix = get_1000genomes_ld_weights(population="EUR")
    annot, weights, prefix = get_1000genomes_ld_weights(population="EAS")

    plink_files, prefix = get_1000genomes_plink_files(population="EUR")
    plink_files, prefix = get_1000genomes_plink_files(population="EAS")
