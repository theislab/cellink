import hashlib
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from os.path import expanduser, join
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import yaml
from tqdm import tqdm

DEFAULT_DATA_HOME = join("~", "cellink_data")


def get_data_home(data_home: str | None = None) -> Path:
    """
    Get or create the local data storage directory.

    This function determines the directory where data files should be stored.
    If the directory does not exist, it will be created. The location can be
    specified via the `data_home` parameter or the `CELLINK_DATA` environment variable.
    If neither is provided, a default path (`~/cellink_data`) is used.

    Parameters
    ----------
    data_home : str, optional
        Path to the desired data storage directory. If None, defaults to
        the environment variable `CELLINK_DATA` or `~/cellink_data`.

    Returns
    -------
    pathlib.Path
        Path object pointing to the data home directory.
    """
    if data_home is None:
        data_home = os.environ.get("CELLINK_DATA", DEFAULT_DATA_HOME)
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return Path(data_home)


def clear_data_home(data_home: str | None = None) -> None:
    """
    Remove all cached data from the local storage directory.

    This function deletes the directory returned by `get_data_home` and all
    its contents. Use with caution as this operation is irreversible.

    Parameters
    ----------
    data_home : str, optional
        Path to the data storage directory to clear. If None, uses the default
        data home directory.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home, ignore_errors=True)


def _sha256sum(filename: str | Path) -> str:
    """
    Compute the SHA-256 checksum of a file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the file for which the checksum will be computed.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash of the file.
    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class DownloadProgressBar(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def _download_file(url: str, dest: Path, checksum: str | None = None) -> None:
    """
    Download a file from a URL and optionally verify its SHA-256 checksum.

    If the destination file already exists and matches the provided checksum,
    the download is skipped. If the file exists but the checksum does not match,
    it will be re-downloaded.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dest : pathlib.Path
        Destination path where the file should be saved.
    checksum : str, optional
        Expected SHA-256 hash of the file. If provided, downloaded file will
        be validated against this hash.

    Raises
    ------
    ValueError
        If the downloaded file does not match the provided checksum.
    """
    if dest.exists():
        logging.info(f"{dest} already exists")
        if checksum is None:
            logging.warning("No checksum provided, skipping verification")
            return
        logging.info("Veryifying checksum")
        if checksum and _sha256sum(dest) == checksum:
            return
        logging.info(f"{dest} exists but checksum mismatch. Re-downloading.")
        dest.unlink()

    logging.info(f"Downloading {url} to {dest}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urlretrieve(url, filename=dest, reporthook=t.update_to)
    if checksum and _sha256sum(dest) != checksum:
        raise ValueError(f"Checksum mismatch for {dest}")


def _run(cmd: str, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """
    Execute a system command using subprocess.

    Parameters
    ----------
    cmd : str
        Command string to execute in the shell.
    cwd : str or pathlib.Path, optional
        Working directory in which to run the command. Defaults to the current
        working directory.

    Returns
    -------
    subprocess.CompletedProcess
        Object containing execution details, including return code, stdout, and stderr.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a non-zero return code.
    """
    logging.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    return result


def _load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration data as a dictionary.

    Raises
    ------
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    FileNotFoundError
        If the file does not exist.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def _to_dataframe(data: list[dict] | None) -> pd.DataFrame:
    """
    Convert a list of dictionaries or JSON-like objects to a pandas DataFrame.

    Parameters
    ----------
    data : list[dict] or None
        Input data to convert. If None or empty, an empty DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame representing the input data.
    """
    if not data:
        return pd.DataFrame()
    return pd.json_normalize(data)


def _cache_df(data_home: Path, filename: str, refresh: bool, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """
    Cache a pandas DataFrame to disk and optionally refresh it.

    If the cached file exists and `refresh` is False, the DataFrame is loaded
    from disk. Otherwise, the `fetcher` function is called to generate the DataFrame,
    which is then saved to disk for future use.

    Parameters
    ----------
    data_home : pathlib.Path
        Directory where cached files are stored.
    filename : str
        Name of the cache file (should include '.parquet' extension).
    refresh : bool
        If True, ignores any existing cache and regenerates the DataFrame.
    fetcher : callable
        Function that returns the DataFrame to cache if needed.

    Returns
    -------
    pd.DataFrame
        Cached or freshly generated DataFrame.
    """
    path = data_home / filename
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetcher()
    df.to_parquet(path, index=False)
    return df
