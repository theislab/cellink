import logging
from urllib.request import urlretrieve

import pandas as pd
import requests
from typing import Any, Optional, Union
from pathlib import Path

from cellink.resources._utils import _cache_df, _to_dataframe, get_data_home

logging.basicConfig(level=logging.INFO)

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
PGS_API_BASE = "https://www.pgscatalog.org/rest"
EQTL_API_BASE = "https://www.ebi.ac.uk/eqtl/api/v3"



def _fetch(
    url: str, 
    params: Optional[dict[str, Any]] = None, 
    paginate: bool = True, 
    max_pages: Optional[int] = None
) -> Union[list, dict]:
    """
    Fetch JSON data from a REST API, optionally handling pagination.

    Parameters
    ----------
    url : str
        The API endpoint to fetch data from.
    params : dict, optional
        Query parameters to pass to the GET request.
    paginate : bool, default=True
        If True, will follow pagination links to retrieve all results.
    max_pages : int, optional
        Maximum number of pages to retrieve. Ignored if `paginate` is False.

    Returns
    -------
    list or dict
        If the endpoint supports pagination, returns a list of results aggregated across pages.
        Otherwise, returns the raw JSON response as a dictionary.
    """
    results = []
    page = 0
    while url:
        logging.info(f"Fetching {url}")
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        if "_embedded" in data:
            for v in data["_embedded"].values():
                results.extend(v)
        elif "results" in data:
            results.extend(data["results"])
        else:
            return data

        if "_links" in data:
            url = data.get("_links", {}).get("next", {}).get("href") if paginate else None
        elif "next" in data:
            url = data["next"] if paginate else None
        else:
            url = None

        page += 1
        if max_pages and page >= max_pages:
            break

    return results


def get_gwas_catalog_studies(
    data_home: Optional[Union[str, Path]] = None,
    max_pages: Optional[int] = None,
    refresh: bool = False,
    **params: Any
) -> pd.DataFrame:
    """
    Retrieve GWAS catalog studies and cache locally as a parquet file.

    Parameters
    ----------
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    max_pages : int, optional
        Maximum number of API pages to fetch.
    refresh : bool, default=False
        If True, ignore cached data and fetch fresh data from the API.
    **params
        Additional query parameters to filter studies.

    Returns
    -------
    pd.DataFrame
        DataFrame containing GWAS study metadata.
    """
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "gwas_studies.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{GWAS_API_BASE}/studies", params=params, max_pages=max_pages)),
    )


def get_gwas_catalog_study(accession_id: str, **params: Any) -> dict:
    """
    Fetch details of a single GWAS study.

    Parameters
    ----------
    accession_id : str
        GWAS study accession ID (e.g., "GCST90018690").
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    dict
        JSON dictionary with study metadata.
    """
    return _fetch(f"studies/{accession_id}", params=params, paginate=False)


def get_gwas_catalog_study_summary_stats(
    accession_id: str,
    dest: Optional[Union[str, Path]] = None,
    return_path: bool = False,
    **params: Any
) -> Union[pd.DataFrame, Path]:
    """
    Download full summary statistics for a GWAS study.

    Parameters
    ----------
    accession_id : str
        GWAS study accession ID.
    dest : str or Path, optional
        Destination path to save the summary statistics file. Defaults to data home directory.
    return_path : bool, default=False
        If True, return the local file path instead of reading the file.
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    pd.DataFrame or Path
        DataFrame containing the summary statistics, or Path to the downloaded file if `return_path=True`.
    """
    if not dest:
        data_home = get_data_home()
        dest = data_home / f"{accession_id}_summary_stats.tsv.gz"

    url = (
        _fetch(f"{GWAS_API_BASE}/studies/{accession_id}", params=params, paginate=False)["full_summary_stats"]
        + f"/{accession_id}_buildGRCh37.tsv.gz"
    )
    logging.info(f"Downloading {url} to {dest}")
    urlretrieve(url, dest)

    if return_path:
        return dest

    data = pd.read_csv(dest, compression="gzip", delimiter="\t")
    return data


def get_gwas_catalog_genes(
    data_home: Optional[Union[str, Path]] = None,
    refresh: bool = False,
    **params: Any
) -> pd.DataFrame:
    """
    Retrieve GWAS catalog gene associations and cache locally.

    Parameters
    ----------
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    refresh : bool, default=False
        If True, ignore cached data and fetch fresh data.
    **params
        Additional query parameters to filter genes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing GWAS gene associations.
    """
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "gwas_genes.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{GWAS_API_BASE}/genes", params=params)),
    )


def get_gwas_catalog_gene(gene_name: str, **params: Any) -> dict:
    """
    Fetch details for a specific GWAS catalog gene.

    Parameters
    ----------
    gene_name : str
        Gene symbol (e.g., "SASH1").
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    dict
        JSON dictionary with gene metadata.
    """
    return _fetch(f"{GWAS_API_BASE}/genes/{gene_name}", params=params, paginate=False)


def get_pgs_catalog_scores(
    data_home: Optional[Union[str, Path]] = None,
    max_pages: Optional[int] = None,
    refresh: bool = False,
    **params: Any
) -> pd.DataFrame:
    """
    Retrieve PGS catalog scores and cache locally.

    Parameters
    ----------
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    max_pages : int, optional
        Maximum number of API pages to fetch.
    refresh : bool, default=False
        If True, ignore cached data and fetch fresh data.
    **params
        Additional query parameters to filter scores.

    Returns
    -------
    pd.DataFrame
        DataFrame containing PGS scores metadata.
    """
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "pgs_scores.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{PGS_API_BASE}/score/all", params=params, max_pages=max_pages)),
    )


def get_pgs_catalog_score(pgs_id: str, **params: Any) -> dict:
    """
    Fetch details for a single PGS score.

    Parameters
    ----------
    pgs_id : str
        PGS catalog score ID (e.g., "PGS000043").
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    dict
        JSON dictionary with score metadata.
    """
    return _fetch(f"{PGS_API_BASE}/score/{pgs_id}", params=params, paginate=False)


def get_pgs_catalog_score_file(
    pgs_id: str,
    dest: Optional[Union[str, Path]] = None,
    return_path: bool = False,
    **params: Any
) -> Union[pd.DataFrame, Path]:
    """
    Download the scoring file for a PGS catalog score.

    Parameters
    ----------
    pgs_id : str
        PGS catalog score ID.
    dest : str or Path, optional
        Destination path to save the scoring file. Defaults to data home directory.
    return_path : bool, default=False
        If True, return the local file path instead of reading the file into a DataFrame.
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    pd.DataFrame or Path
        DataFrame containing scoring data, or Path to the downloaded file if `return_path=True`.
    """
    meta = _fetch(f"{PGS_API_BASE}/score/{pgs_id}", params=params, paginate=False)
    url = meta["ftp_scoring_file"]

    if not dest:
        data_home = get_data_home()
        dest = data_home / f"{pgs_id}_scoring_file.txt.gz"

    logging.info(f"Downloading {url} to {dest}")
    urlretrieve(url, dest)

    if return_path:
        return dest

    df = pd.read_csv(dest, compression="gzip", sep="\t", comment="#")
    return df


def get_eqtl_catalog_datasets(
    data_home: Optional[Union[str, Path]] = None,
    max_pages: Optional[int] = None,
    refresh: bool = False,
    **params: Any
) -> pd.DataFrame:
    """
    Retrieve eQTL catalog datasets and cache locally.

    Parameters
    ----------
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    max_pages : int, optional
        Maximum number of API pages to fetch.
    refresh : bool, default=False
        If True, ignore cached data and fetch fresh data.
    **params
        Additional query parameters to filter datasets.

    Returns
    -------
    pd.DataFrame
        DataFrame containing eQTL catalog datasets metadata.
    """
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "eqtl_datasets.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{EQTL_API_BASE}/datasets", params=params, max_pages=max_pages)),
    )


def get_eqtl_catalog_dataset_associations(
    dataset_id: str,
    data_home: Optional[Union[str, Path]] = None,
    refresh: bool = False,
    return_path: bool = False,
    **params: Any
) -> Union[pd.DataFrame, Path]:
    """
    Retrieve associations for a specific eQTL catalog dataset and cache locally.

    Parameters
    ----------
    dataset_id : str
        eQTL catalog dataset ID (e.g., "QTD000319").
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    refresh : bool, default=False
        If True, ignore cached data and fetch fresh data.
    return_path : bool, default=False
        If True, return the local cached file path instead of reading it into a DataFrame.
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    pd.DataFrame or Path
        DataFrame of eQTL associations, or Path to the cached parquet file if `return_path=True`.
    """
    data_home = get_data_home(data_home)
    dest = data_home / f"{dataset_id}_eqtl_associations.parquet"

    if dest.exists() and not refresh:
        return dest if return_path else pd.read_parquet(dest)

    data = _fetch(f"{EQTL_API_BASE}/datasets/{dataset_id}/associations", params=params)
    df = _to_dataframe(data)
    df.to_parquet(dest)

    if return_path:
        return dest

    return df


if __name__ == "__main__":
    import json

    studies = get_gwas_catalog_studies(max_pages=1)
    print(studies)

    study = get_gwas_catalog_study("GCST90018690")
    print(json.dumps(study, indent=4))

    study_summary_stat = get_gwas_catalog_study_summary_stats("GCST90018690")
    print(study_summary_stat)

    genes = get_gwas_catalog_genes()
    print(genes)

    gene = get_gwas_catalog_gene(gene_name="SASH1")
    print(json.dumps(gene, indent=4))

    pgs = get_pgs_catalog_scores()
    print(pgs.head())

    pgs_score = get_pgs_catalog_score("PGS000043")

    pgs_score_file = get_pgs_catalog_score_file("PGS000043")

    eqtl_datasets = get_eqtl_catalog_datasets()
    print(eqtl_datasets.head())

    eqtl_dataset = get_eqtl_catalog_dataset_associations("QTD000319")
