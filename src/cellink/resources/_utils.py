import hashlib
import logging
import os
import shutil
import subprocess
from os.path import expanduser, join
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import yaml

DEFAULT_DATA_HOME = join("~", "cellink_data")


def get_data_home(data_home=None):
    """Get or create the local data storage directory."""
    if data_home is None:
        data_home = os.environ.get("CELLINK_DATA", DEFAULT_DATA_HOME)
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return Path(data_home)


def clear_data_home(data_home=None):
    """Remove all data from local cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home, ignore_errors=True)


def _sha256sum(filename):
    """Return the sha256 checksum of the file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url, dest, checksum=None):
    """Download a file and verify checksum."""
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
    urlretrieve(url, dest)
    if checksum and _sha256sum(dest) != checksum:
        raise ValueError(f"Checksum mismatch for {dest}")


def _run(cmd, cwd=None):
    """Run a system command using subprocess."""
    logging.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    return result


def _load_config(path):
    """Load the YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def _to_dataframe(data):
    if not data:
        return pd.DataFrame()
    return pd.json_normalize(data)


def _cache_df(data_home, filename, refresh, fetcher):
    path = data_home / filename
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetcher()
    df.to_parquet(path, index=False)
    return df
