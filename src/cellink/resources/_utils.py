import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from os.path import expanduser, join
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

from cellink._config import load_config

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


def _part_path(dest: Path) -> Path:
    """Unique temporary file next to ``dest``; written there first and renamed to ``dest`` when complete."""
    return dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")


def _download_file(
    url: str, dest: Path, checksum: str | None = None, refresh: bool = False, timeout: float = 60
) -> None:
    """
    Download a file from a URL and optionally verify its SHA-256 checksum.

    The file is first written to a temporary ``{dest.name}.*.part`` file next to
    ``dest`` and renamed to ``dest`` only once it is complete (and matches
    ``checksum``), so an interrupted download never leaves a truncated ``dest``
    behind and parallel callers never see a partial file. If ``dest`` already
    exists and ``refresh`` is False, the download is skipped when no checksum is
    given or the checksum matches; on a checksum mismatch it is downloaded again.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dest : pathlib.Path
        Destination path where the file should be saved.
    checksum : str, optional
        Expected SHA-256 hash of the file. If provided, downloaded file will
        be validated against this hash.
    refresh : bool, default=False
        If True, download again even if ``dest`` already exists.
    timeout : float, default=60
        Seconds to wait for the connection and for each read before raising.

    Raises
    ------
    ValueError
        If the downloaded file does not match the provided checksum.
    OSError
        If the server closes the connection before the announced size was received,
        or the connection times out.
    """
    dest = Path(dest)
    if dest.exists() and not refresh:
        logging.info(f"{dest} already exists")
        if checksum is None:
            logging.warning("No checksum provided, skipping verification")
            return
        logging.info("Verifying checksum")
        if _sha256sum(dest) == checksum:
            return
        logging.info(f"{dest} exists but checksum mismatch. Re-downloading.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    logging.info(f"Downloading {url} to {dest}")
    try:
        with urlopen(url, timeout=timeout) as response, open(part, "wb") as fh:
            size = response.headers.get("Content-Length")
            size = int(size) if size and size.isdigit() else None
            received = 0
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name, total=size) as t:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    received += len(chunk)
                    t.update(len(chunk))
        if size is not None and received != size:
            raise OSError(f"Download of {url} stopped after {received:,} of {size:,} bytes")
        if checksum and _sha256sum(part) != checksum:
            raise ValueError(f"Checksum mismatch for {dest}")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def retry_with_backoff(fn: Callable, n_attempts: int = 8, base_delay: float = 5, max_delay: float = 30):
    """
    Call ``fn()`` and retry with exponential backoff on failure.

    Ensembl's BioMart endpoints occasionally return transient 5xx errors or
    truncated/malformed responses (sometimes even an HTML "Service unavailable"
    page served with HTTP 200) under load, so callers querying BioMart should
    retry rather than fail a long-running pipeline on a one-off blip.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable to invoke.
    n_attempts : int
        Maximum number of attempts before raising.
    base_delay : float
        Delay in seconds before the first retry; doubles each subsequent attempt.
    max_delay : float
        Cap on the per-attempt delay.
    """
    for attempt in range(1, n_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == n_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logging.warning(f"Request failed ({e}); retrying in {delay}s (attempt {attempt}/{n_attempts})")
            time.sleep(delay)


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


def _load_config(path: str | Path, config_dict: dict | None = None) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Name of a config shipped with cellink (e.g. ``"1000genomes"``), or path to a YAML file.
    config_dict : dict, optional
        Configuration as a dictionary; returned as is, without reading ``path``.

    Returns
    -------
    dict
        Parsed configuration data as a dictionary.

    Raises
    ------
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    FileNotFoundError
        If the config cannot be found.
    """
    return load_config(path, config_dict)


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


def _cache_key_digest(key: dict) -> str:
    """Short, stable hash of a dict of query parameters, used in cache file names."""
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _cache_df(
    data_home: Path,
    filename: str,
    refresh: bool,
    fetcher: Callable[[], pd.DataFrame],
    key: dict | None = None,
) -> pd.DataFrame:
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
    key : dict, optional
        Parameters the result depends on (e.g. API query parameters). If given, a hash
        of them is added to the file name, so every distinct query has its own cache file.

    Returns
    -------
    pd.DataFrame
        Cached or freshly generated DataFrame.
    """
    if key is not None:
        stem, dot, suffix = filename.rpartition(".")
        filename = f"{stem}_{_cache_key_digest(key)}{dot}{suffix}" if dot else f"{filename}_{_cache_key_digest(key)}"
    path = data_home / filename
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetcher()
    part = _part_path(path)
    try:
        df.to_parquet(part, index=False)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
    return df
