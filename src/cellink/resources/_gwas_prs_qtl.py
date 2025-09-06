import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from urllib.request import urlretrieve
from cellink.resources._utils import get_data_home, _download_file, _run, _load_config, _to_dataframe, _cache_df

logging.basicConfig(level=logging.INFO)

"""
GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"

#TODO: MOVE?
def _fetch(endpoint, params=None, paginate=True, max_pages=None):
    url = f"{GWAS_API_BASE}/{endpoint.lstrip('/')}"
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
        else:
            return data

        url = data.get("_links", {}).get("next", {}).get("href") if paginate else None
        page += 1
        if max_pages and page >= max_pages:
            break

    return results
"""

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
PGS_API_BASE = "https://www.pgscatalog.org/rest"
EQTL_API_BASE = "https://www.ebi.ac.uk/eqtl/api/v3"

def _fetch(url, params=None, paginate=True, max_pages=None):
    """Generic fetch with pagination support for APIs following HATEOAS style."""
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

def get_gwas_catalog_studies(data_home=None, max_pages=None, refresh=False, **params):
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "gwas_studies.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{GWAS_API_BASE}/studies", params=params, max_pages=max_pages)),
    )

def get_gwas_catalog_study(accession_id, **params):
    return _fetch(f"studies/{accession_id}", params=params, paginate=False)

def get_gwas_catalog_study_summary_stats(accession_id, dest=None, **params):
    if not dest:
        data_home = get_data_home()
        dest = data_home / f"{accession_id}_summary_stats.tsv.gz"
    url = _fetch(f"{GWAS_API_BASE}/studies/{accession_id}", params=params, paginate=False)['full_summary_stats'] + f"/{accession_id}_buildGRCh37.tsv.gz"
    logging.info(f"Downloading {url} to {dest}")
    urlretrieve(url, dest)
    data = pd.read_csv(dest, compression="gzip", delimiter="\t")
    return data

def get_gwas_catalog_genes(data_home=None, refresh=False, **params):
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "gwas_genes.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{GWAS_API_BASE}/genes", params=params)),
    )

def get_gwas_catalog_gene(gene_name, **params):
    return _fetch(f"{GWAS_API_BASE}/genes/{gene_name}", params=params, paginate=False)

def get_pgs_catalog_scores(data_home=None, max_pages=None, refresh=False, **params):
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "pgs_scores.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{PGS_API_BASE}/score/all", params=params, max_pages=max_pages)),
    )

def get_pgs_catalog_score(pgs_id, **params):
    return _fetch(f"{PGS_API_BASE}/score/{pgs_id}", params=params, paginate=False)

def get_pgs_catalog_score_file(pgs_id, dest=None, **params):

    meta = _fetch(f"{PGS_API_BASE}/score/{pgs_id}", params=params, paginate=False)
    url = meta['ftp_scoring_file']

    if not dest:
        data_home = get_data_home()
        dest = data_home / f"{pgs_id}_scoring_file.txt.gz"

    logging.info(f"Downloading {url} to {dest}")
    urlretrieve(url, dest)

    df = pd.read_csv(dest, compression="gzip", sep="\t", comment="#")
    return df

def get_eqtl_catalog_datasets(data_home=None, max_pages=None, refresh=False, **params):
    data_home = get_data_home(data_home)
    return _cache_df(
        data_home,
        "eqtl_datasets.parquet",
        refresh,
        lambda: _to_dataframe(_fetch(f"{EQTL_API_BASE}/datasets", params=params, max_pages=max_pages)),
    )

def get_eqtl_catalog_dataset_associations(dataset_id, data_home=None, refresh=False, **params):
    data_home = get_data_home(data_home)
    dest = data_home / f"{dataset_id}_eqtl_associations.parquet"
    if dest.exists() and not refresh:
        return pd.read_parquet(dest)

    data = _fetch(f"{EQTL_API_BASE}/datasets/{dataset_id}/associations", params=params)
    df = _to_dataframe(data)
    df.to_parquet(dest)
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
