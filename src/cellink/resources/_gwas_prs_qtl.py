import io
import logging
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd
import requests

from cellink.resources._utils import _cache_df, _download_file, _to_dataframe, get_data_home


def _normalize_build(genome_build: str) -> str:
    """Normalize genome build aliases to 'GRCh37' or 'GRCh38'."""
    lower = genome_build.lower()
    if lower in ("grch38", "hg38", "build38"):
        return "GRCh38"
    if lower in ("grch37", "hg19", "build37"):
        return "GRCh37"
    raise ValueError(f"Invalid genome_build '{genome_build}'. Use 'GRCh38', 'GRCh37', or an alias (hg38/hg19).")


_NON_DATA_FILENAME = re.compile(r"(?i)readme|license|changelog|^md5sum")


def _find_candidate_files(html: str) -> list[str]:
    """Find likely summary-stats filenames in an FTP directory listing.

    Matches anything ending in .tsv, .txt, .zip, or .gz, since some
    pre-harmonisation-era deposits ship a plain/zipped .txt instead of
    .tsv.gz, or a bare ".gz" with no .tsv/.txt in the name at all (e.g. a
    sleep-duration GWAS whose only file is "..._sumstats.txt.zip", or a
    major-depression GWAS whose only file is literally
    "MDD2018_ex23andMe.gz"). Since .tsv.gz/.txt.gz already end in ".gz",
    matching bare ".gz" covers all of those cases in one pattern. Widening
    this far also risks matching README/LICENSE/CHANGELOG files that live
    in the same directory, hence the explicit exclusion below.
    """
    files = re.findall(r'href="([^"]*\.(?:tsv|txt|zip|gz))"', html)
    return [f for f in files if not _NON_DATA_FILENAME.search(f)]


logging.basicConfig(level=logging.INFO)

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
PGS_API_BASE = "https://www.pgscatalog.org/rest"
EQTL_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL"
EQTL_TABIX_PATHS_URL = (
    "https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/"
    "master/tabix/tabix_ftp_paths.tsv"
)
EQTL_SUMSTATS_COLUMNS = (
    "molecular_trait_id", "chromosome", "position", "ref", "alt", "variant", "ma_samples",
    "maf", "pvalue", "beta", "se", "type", "ac", "an", "r2", "molecular_trait_object_id",
    "gene_id", "median_tpm", "rsid",
)
EQTL_CREDIBLE_SET_COLUMNS = (
    "molecular_trait_id", "gene_id", "cs_id", "variant", "rsid", "cs_size", "pip",
    "pvalue", "beta", "se", "z", "cs_min_r2", "region",
)


def _fetch(
    url: str, params: dict[str, Any] | None = None, paginate: bool = True, max_pages: int | None = None
) -> list | dict:
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
    results: list = []
    page = 0
    next_url: str | None = url
    next_params = params

    while next_url:
        logging.info(f"Fetching {next_url}")
        r = requests.get(next_url, params=next_params)
        r.raise_for_status()
        data = r.json()

        page_items: list | None = None
        if "_embedded" in data:
            page_items = [item for v in data["_embedded"].values() for item in v]
        elif "results" in data:
            page_items = data["results"]

        if page_items is None:
            if results:
                logging.debug(f"No collection payload on page {page}; returning {len(results)} collected items.")
                break
            return data

        results.extend(page_items)

        if not paginate:
            break

        links = data.get("_links") or {}
        next_url = (links.get("next") or {}).get("href") or None if links else data.get("next") or None
        next_params = None

        page += 1
        if max_pages and page >= max_pages:
            break

    return results


def get_gwas_catalog_studies(
    data_home: str | Path | None = None, max_pages: int | None = None, refresh: bool = False, **params: Any
) -> pd.DataFrame:
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
    return _fetch(f"{GWAS_API_BASE}/studies/{accession_id}", params=params, paginate=False)


def get_gwas_catalog_study_summary_stats(
    accession_id: str,
    dest: str | Path | None = None,
    return_path: bool = False,
    genome_build: str | None = None,
    translate_to_build: str | None = None,
    **params: Any,
) -> pd.DataFrame | Path:
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
        Cannot be combined with ``translate_to_build``.
    genome_build : str, optional
        Preferred genome build for the downloaded file: 'GRCh38', 'GRCh37', or None for
        automatic priority selection (prefers EBI harmonised GRCh38 files).
        Aliases: 'hg38'/'build38' for GRCh38, 'hg19'/'build37' for GRCh37.
    translate_to_build : str, optional
        After downloading, translate SNP positions to this genome build using liftover.
        Useful when EBI does not host a pre-built file for the requested build.
        Requires the ``liftover`` package (``pip install cellink[datasets]``).
        Cannot be combined with ``return_path=True``.
    **params
        Additional query parameters to pass to the API.

    Returns
    -------
    pd.DataFrame or Path
        DataFrame containing the summary statistics, or Path to the downloaded file if return_path=True.
    """
    if translate_to_build and return_path:
        raise ValueError("translate_to_build requires return_path=False, since liftover operates on an in-memory DataFrame.")
    study_meta = _fetch(f"{GWAS_API_BASE}/studies/{accession_id}", params=params, paginate=False)

    if "full_summary_stats" not in study_meta:
        raise ValueError(f"Study {accession_id} does not have full summary statistics available")

    base_url = study_meta["full_summary_stats"]
    harmonised_url = f"{base_url}/harmonised"

    normalized_build = _normalize_build(genome_build) if genome_build else None

    def build_priority(filename):
        """Assign priority score to filename based on genome build."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".h.tsv.gz"):
            return 2
        elif "build38" in filename_lower or "hg38" in filename_lower or "grch38" in filename_lower:
            return 2
        elif "build37" in filename_lower or "hg19" in filename_lower or "grch37" in filename_lower:
            return 1
        else:
            return 0

    def get_build_from_filename(filename):
        """Extract genome build information from filename."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".h.tsv.gz"):
            return "GRCh38"
        elif "build38" in filename_lower or "hg38" in filename_lower or "grch38" in filename_lower:
            return "GRCh38"
        elif "build37" in filename_lower or "hg19" in filename_lower or "grch37" in filename_lower:
            return "GRCh37"
        else:
            return "unknown"

    def select_file(files, requested_build=None):
        """Select appropriate file based on user preference or priority."""
        if not files:
            return None, None

        # If user specified a build, try to find it
        if requested_build:
            matches = [f for f in files if get_build_from_filename(f) == requested_build]
            if matches:
                harmonised = [f for f in matches if f.lower().endswith(".h.tsv.gz")]
                chosen = harmonised[0] if harmonised else matches[0]
                logging.info(f"Selected file matching requested build {requested_build}: {chosen}")
                return chosen, requested_build

            logging.warning(f"Requested build {requested_build} not found. Falling back to priority selection.")

        # Fall back to priority logic
        files.sort(key=build_priority, reverse=True)
        selected_file = files[0]
        detected_build = get_build_from_filename(selected_file)

        logging.info(f"Selected file with build {detected_build} (priority selection): {selected_file}")

        return selected_file, detected_build

    url = None
    filename = None
    detected_build = None

    exclude_harmonised = normalized_build and normalized_build != "GRCh38"
    if normalized_build:
        logging.info(f"User requested {normalized_build}, searching for a build-specific file")

        try:
            r = requests.get(harmonised_url)
            r.raise_for_status()
            files = re.findall(r'href="([^"]*\.tsv\.gz)"', r.text)
            files = [f for f in files if not f.endswith(".h.tsv.gz-meta.yaml")]
            if exclude_harmonised:
                files = [f for f in files if not f.endswith(".h.tsv.gz")]

            if files:
                filename, detected_build = select_file(files, requested_build=normalized_build)
                if filename:
                    url = f"{harmonised_url}/{filename}"
                    if filename.endswith(".h.tsv.gz"):
                        detected_build = "GRCh38"
                    logging.info(f"Using build-specific summary statistics from harmonised/ (build: {detected_build})")

        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not parse harmonised directory listing ({e})")

        if not url:
            try:
                r = requests.get(base_url)
                r.raise_for_status()
                files = _find_candidate_files(r.text)

                # Exclude harmonised files if they appear in the listing (see above)
                if exclude_harmonised:
                    files = [f for f in files if not f.endswith(".h.tsv.gz")]

                if files:
                    filename, detected_build = select_file(files, requested_build=normalized_build)
                    if filename:
                        url = f"{base_url}/{filename}"
                        logging.info(f"Using build-specific summary statistics (build: {detected_build})")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not parse base directory listing ({e})")

        # If still no file found, try standard naming conventions
        if not url:
            logging.info("Trying standard naming conventions for requested build")

            if normalized_build == "GRCh38":
                possible_files = [
                    f"{accession_id}_buildGRCh38.tsv.gz",
                    f"{accession_id}.tsv.gz",
                ]
            else:  # GRCh37
                possible_files = [
                    f"{accession_id}_buildGRCh37.tsv.gz",
                    f"{accession_id}.tsv.gz",
                ]

            for test_filename in possible_files:
                test_url = f"{base_url}/{test_filename}"
                try:
                    test_r = requests.head(test_url)
                    if test_r.status_code == 200:
                        url = test_url
                        filename = test_filename
                        detected_build = get_build_from_filename(test_filename)
                        logging.info(
                            f"Found file via standard naming convention (build: {detected_build}): {test_filename}"
                        )
                        break
                except requests.exceptions.RequestException:
                    continue

    else:
        # No specific build requested - prefer harmonised files
        try:
            r = requests.get(harmonised_url)
            r.raise_for_status()

            all_files = re.findall(r'href="([^"]*\.tsv\.gz)"', r.text)
            h_files = [f for f in all_files if f.endswith(".h.tsv.gz") and not f.endswith(".h.tsv.gz-meta.yaml")]

            if h_files:
                filename, detected_build = select_file(h_files)
                if filename:
                    url = f"{harmonised_url}/{filename}"
                    logging.info(f"Using harmonised summary statistics (build: {detected_build})")

        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not access harmonised directory ({e})")

        # If no harmonised files found, try base directory
        if not url:
            logging.info("No harmonised files found, trying base directory")

            try:
                r = requests.get(base_url)
                r.raise_for_status()
                files = _find_candidate_files(r.text)

                if files:
                    filename, detected_build = select_file(files)
                    if filename:
                        url = f"{base_url}/{filename}"
                        logging.info(f"Using non-harmonised summary statistics (build: {detected_build})")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not parse base directory listing ({e})")

        # If still no file found, try standard naming conventions
        if not url:
            logging.info("Trying standard naming conventions")

            possible_files = [
                f"{accession_id}_buildGRCh38.tsv.gz",
                f"{accession_id}_buildGRCh37.tsv.gz",
                f"{accession_id}.tsv.gz",
            ]

            for test_filename in possible_files:
                test_url = f"{base_url}/{test_filename}"
                try:
                    test_r = requests.head(test_url)
                    if test_r.status_code == 200:
                        url = test_url
                        filename = test_filename
                        detected_build = get_build_from_filename(test_filename)
                        logging.info(
                            f"Found file via standard naming convention (build: {detected_build}): {test_filename}"
                        )
                        break
                except requests.exceptions.RequestException:
                    continue

    if not url:
        raise ValueError(f"Could not find summary statistics file for {accession_id}")

    if url.endswith(".gz"):
        suffix, compression = ".gz", "gzip"
    elif url.endswith(".zip"):
        suffix, compression = ".zip", "zip"
    else:
        suffix, compression = "", None
    if not dest:
        data_home = get_data_home()
        dest = data_home / f"{accession_id}_summary_stats.tsv{suffix}"

    logging.info(f"Downloading {url} to {dest}")

    try:
        urlretrieve(url, dest)
    except (OSError, URLError) as e:
        raise RuntimeError(f"Failed to download summary statistics from {url}: {e}") from e

    if return_path:
        return dest

    data = pd.read_csv(dest, compression=compression, sep=r"\s+")
    if translate_to_build:
        data = liftover_gwas_summary_stats(
            data, source_build=detected_build or "GRCh38", target_build=translate_to_build
        )
    return data


def liftover_gwas_summary_stats(
    df: pd.DataFrame,
    source_build: str,
    target_build: str,
    chrom_col: str = "chromosome",
    pos_col: str = "base_pair_location",
    drop_failed: bool = True,
) -> pd.DataFrame:
    """
    Translate SNP positions in a GWAS summary-statistics DataFrame between genome builds.

    Parameters
    ----------
    df : pd.DataFrame
        GWAS summary statistics DataFrame containing at least ``chrom_col`` and ``pos_col``.
    source_build : str
        Genome build of the input positions. Accepts 'GRCh38'/'hg38' or 'GRCh37'/'hg19'.
    target_build : str
        Genome build to translate positions to.
    chrom_col : str, default='chromosome'
        Name of the chromosome column.
    pos_col : str, default='base_pair_location'
        Name of the position column.
    drop_failed : bool, default=True
        If True, rows whose positions could not be lifted over are dropped.
        If False, failed positions are set to ``pd.NA``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``pos_col`` replaced by translated positions.

    Raises
    ------
    ImportError
        If the ``liftover`` package is not installed.
    """
    _UCSC = {"GRCh37": "hg19", "GRCh38": "hg38"}
    src = _UCSC[_normalize_build(source_build)]
    tgt = _UCSC[_normalize_build(target_build)]

    if src == tgt:
        logging.info(f"source_build and target_build are both {source_build}; returning df unchanged.")
        return df.copy()

    try:
        from liftover import get_lifter
    except ImportError as e:
        raise ImportError(
            "The 'liftover' package is required for build translation. "
            "Install it with: pip install cellink[datasets]"
        ) from e

    converter = get_lifter(src, tgt, one_based=True)

    chroms = df[chrom_col].astype(str).tolist()
    positions = df[pos_col].tolist()
    results = [converter[c][p] for c, p in zip(chroms, positions, strict=False)]
    new_positions = [r[0][1] if r else None for r in results]

    out = df.copy()
    out[pos_col] = new_positions
    if drop_failed:
        n_failed = sum(p is None for p in new_positions)
        if n_failed:
            logging.warning(
                f"liftover: dropped {n_failed:,} rows with no mapping from {source_build} to {target_build}"
            )
        out = out.dropna(subset=[pos_col])
    else:
        out[pos_col] = pd.array(new_positions, dtype=pd.Int64Dtype())

    return out.reset_index(drop=True)


def get_gwas_catalog_genes(data_home: str | Path | None = None, refresh: bool = False, **params: Any) -> pd.DataFrame:
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
    data_home: str | Path | None = None, max_pages: int | None = None, refresh: bool = False, **params: Any
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
    pgs_id: str, dest: str | Path | None = None, return_path: bool = False, **params: Any
) -> pd.DataFrame | Path:
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


def _eqtl_https(url: str) -> str:
    """Rewrite an eQTL Catalogue ``ftp://`` path to its ``https://`` equivalent.

    The catalogue's metadata table publishes ``ftp://ftp.ebi.ac.uk/...`` URLs, but the
    same tree is served over HTTPS with byte-range support, which is what htslib needs
    for remote tabix queries (and what works from behind proxies that block FTP).
    """
    if url.startswith("ftp://"):
        return "https://" + url[len("ftp://") :]
    return url


def _eqtl_tabix_query(url: str, regions: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """Run a remote tabix range query against a bgzipped eQTL Catalogue file.

    A single dataset's ``.all.tsv.gz`` is ~1.4 GB, so region-restricted access is the
    only practical way to read it; the catalogue ships a ``.tbi`` alongside every
    sumstats file for exactly this purpose.

    Parameters
    ----------
    url : str
        HTTPS URL of the bgzipped, tabix-indexed file.
    regions : sequence of str
        Regions in tabix syntax, e.g. ``["6:89900000-90300000"]``. Note the catalogue
        indexes chromosomes **without** a ``chr`` prefix.
    columns : sequence of str
        Column names to apply; the catalogue's files carry a header line that is *not*
        marked as a comment, so it is absent from the tabix index and cannot be
        recovered with ``tabix -H``.

    Returns
    -------
    pd.DataFrame
    """
    if shutil.which("tabix") is None:
        raise RuntimeError(
            "The `tabix` executable is required for region-restricted eQTL Catalogue queries "
            "but was not found on $PATH. Install htslib (e.g. `conda install -c bioconda htslib`), "
            "or call this function without `region=` to download the whole dataset instead."
        )

    frames: list[pd.DataFrame] = []
    for region in regions:
        cmd = ["tabix", url, region]
        logging.info(f"tabix {url} {region}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"tabix failed for region {region!r}: {proc.stderr.strip()[:500]}")
        if not proc.stdout.strip():
            logging.warning(f"No eQTL Catalogue records returned for region {region!r}.")
            continue
        frames.append(pd.read_csv(io.StringIO(proc.stdout), sep="\t", header=None, names=list(columns)))

    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def get_eqtl_catalog_datasets(
    data_home: str | Path | None = None,
    max_pages: int | None = None,
    refresh: bool = False,
    **params: Any,
) -> pd.DataFrame:
    """
    Retrieve the eQTL Catalogue dataset index and cache locally.

    Returns one row per dataset (a study x sample-group x quantification-method
    combination), including the FTP paths of its summary statistics, SuSiE credible
    sets and log-Bayes-factor files.

    Parameters
    ----------
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    max_pages : int, optional
        Ignored. Retained only so that code written against the retired REST API keeps
        working; the index is now a single file with no pagination.
    refresh : bool, default=False
        If True, ignore cached data and re-download the index.
    **params
        Row filters applied to the returned table, as ``column=value`` (or
        ``column=[v1, v2]`` for a membership test). Useful columns are
        ``study_label``, ``tissue_label``, ``condition_label``, ``quant_method``,
        ``dataset_id`` and ``study_id``. String comparisons are case-insensitive.

    Returns
    -------
    pd.DataFrame
        Columns: ``study_id``, ``dataset_id``, ``study_label``, ``sample_group``,
        ``tissue_id``, ``tissue_label``, ``condition_label``, ``sample_size``,
        ``quant_method``, ``ftp_path``, ``ftp_cs_path``, ``ftp_lbf_path``.

    Examples
    --------
    >>> import cellink as cl
    >>> tregs = cl.resources.get_eqtl_catalog_datasets(quant_method="ge", tissue_label="Treg memory")
    >>> tregs[["dataset_id", "study_label", "sample_size"]]
    """
    if max_pages is not None:
        logging.warning(
            "`max_pages` is ignored: the eQTL Catalogue REST API was retired and the dataset "
            "index is now a single unpaginated file."
        )

    data_home = get_data_home(data_home)

    def _fetch_index() -> pd.DataFrame:
        logging.info(f"Fetching eQTL Catalogue dataset index from {EQTL_TABIX_PATHS_URL}")
        return pd.read_csv(EQTL_TABIX_PATHS_URL, sep="\t")

    df = _cache_df(data_home, "eqtl_datasets.parquet", refresh, _fetch_index)

    for key, value in params.items():
        if key not in df.columns:
            raise KeyError(f"'{key}' is not a column of the eQTL Catalogue index. Available: {list(df.columns)}")
        col = df[key]
        if isinstance(value, (list, tuple, set)):
            wanted = {str(v).casefold() for v in value}
            df = df[col.astype(str).str.casefold().isin(wanted)]
        else:
            df = df[col.astype(str).str.casefold() == str(value).casefold()]

    return df.reset_index(drop=True)


def _resolve_eqtl_dataset_path(
    dataset_id: str,
    path_column: str,
    data_home: str | Path | None,
    refresh: bool,
) -> str:
    """Look up one dataset's file URL in the catalogue index."""
    index = get_eqtl_catalog_datasets(data_home=data_home, refresh=refresh)
    hit = index[index["dataset_id"] == dataset_id]
    if hit.empty:
        raise KeyError(
            f"dataset_id '{dataset_id}' is not in the eQTL Catalogue index. "
            "List available datasets with `cellink.resources.get_eqtl_catalog_datasets()`."
        )
    url = hit.iloc[0][path_column]
    if not isinstance(url, str) or not url or url.upper() == "NA":
        raise ValueError(
            f"dataset '{dataset_id}' has no '{path_column}' entry "
            "(not every dataset has SuSiE fine-mapping results)."
        )
    return _eqtl_https(url)


def get_eqtl_catalog_dataset_associations(
    dataset_id: str,
    data_home: str | Path | None = None,
    refresh: bool = False,
    return_path: bool = False,
    region: str | Sequence[str] | None = None,
    **params: Any,
) -> pd.DataFrame | Path:
    """
    Retrieve cis-QTL summary statistics for one eQTL Catalogue dataset.

    Parameters
    ----------
    dataset_id : str
        eQTL Catalogue dataset ID (e.g., ``"QTD000625"``, OneK1K Treg memory).
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    refresh : bool, default=False
        If True, ignore cached data and re-fetch.
    return_path : bool, default=False
        If True, return the local cached file path instead of a DataFrame. Only
        meaningful for a whole-dataset download (``region=None``).
    region : str or sequence of str, optional
        One or more regions in tabix syntax (``"6:89900000-90300000"``), fetched with a
        remote range query instead of downloading the file. **Chromosomes are named
        without a ``chr`` prefix** and coordinates are GRCh38. Strongly recommended:
        a single dataset's full summary statistics are ~1.4 GB.
    **params
        Post-hoc row filters applied to the result as ``column=value``, e.g.
        ``gene_id="ENSG00000112182"`` (BACH2) or ``rsid="rs72928038"``.

    Returns
    -------
    pd.DataFrame or Path
        Summary statistics, or the cached file path when ``return_path=True``.

    Notes
    -----
    The eQTL Catalogue REST API this function previously used
    (``https://www.ebi.ac.uk/eqtl/api/v3``) was retired and now returns HTTP 410 for
    every endpoint and version. Access is via the FTP/tabix distribution described at
    https://www.ebi.ac.uk/eqtl/Data_access/.

    Examples
    --------
    >>> import cellink as cl
    >>> # cis-eQTLs at the BACH2 locus in OneK1K memory Tregs
    >>> df = cl.resources.get_eqtl_catalog_dataset_associations(
    ...     "QTD000625", region="6:89900000-90300000"
    ... )
    >>> df.nsmallest(5, "pvalue")[["rsid", "gene_id", "pvalue", "beta"]]
    """
    data_home = get_data_home(data_home)
    url = _resolve_eqtl_dataset_path(dataset_id, "ftp_path", data_home, refresh)

    if region is not None:
        regions = [region] if isinstance(region, str) else list(region)
        df = _eqtl_tabix_query(url, regions, EQTL_SUMSTATS_COLUMNS)
        if return_path:
            dest = data_home / f"{dataset_id}_eqtl_associations_{'_'.join(regions).replace(':', '-')}.parquet"
            df.to_parquet(dest, index=False)
            return dest
    else:
        dest = data_home / f"{dataset_id}.all.tsv.gz"
        if not dest.exists() or refresh:
            logging.warning(
                f"Downloading the complete summary statistics for {dataset_id} (~1 GB or more). "
                "Pass `region=` for a remote tabix range query instead."
            )
            _download_file(url, dest)
        if return_path:
            return dest
        df = pd.read_csv(dest, sep="\t")

    for key, value in params.items():
        if key not in df.columns:
            raise KeyError(f"'{key}' is not a column of the summary statistics. Available: {list(df.columns)}")
        if isinstance(value, (list, tuple, set)):
            df = df[df[key].isin(list(value))]
        else:
            df = df[df[key] == value]

    return df.reset_index(drop=True)


def get_eqtl_catalog_credible_sets(
    dataset_id: str,
    data_home: str | Path | None = None,
    refresh: bool = False,
    return_path: bool = False,
    **params: Any,
) -> pd.DataFrame | Path:
    """
    Retrieve SuSiE fine-mapped credible sets for one eQTL Catalogue dataset.

    These are the per-variant posterior inclusion probabilities used to define causal
    variants (e.g. the ``PIP >= 0.9`` threshold adopted for variant-effect benchmarks in
    the Borzoi and scooby papers), and are the natural input to a colocalization
    analysis against a GWAS.

    Parameters
    ----------
    dataset_id : str
        eQTL Catalogue dataset ID (e.g., ``"QTD000625"``).
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    refresh : bool, default=False
        If True, ignore cached data and re-download.
    return_path : bool, default=False
        If True, return the local cached file path instead of a DataFrame.
    **params
        Row filters applied to the result as ``column=value``, e.g. ``gene_id=...``.

    Returns
    -------
    pd.DataFrame or Path
        Columns: ``molecular_trait_id``, ``gene_id``, ``cs_id``, ``variant``, ``rsid``,
        ``cs_size``, ``pip``, ``pvalue``, ``beta``, ``se``, ``z``, ``cs_min_r2``,
        ``region``.

    Examples
    --------
    >>> import cellink as cl
    >>> cs = cl.resources.get_eqtl_catalog_credible_sets("QTD000625")
    >>> cs[cs.pip >= 0.9].head()
    """
    data_home = get_data_home(data_home)
    url = _resolve_eqtl_dataset_path(dataset_id, "ftp_cs_path", data_home, refresh)
    dest = data_home / f"{dataset_id}.credible_sets.tsv.gz"
    if not dest.exists() or refresh:
        _download_file(url, dest)
    if return_path:
        return dest

    df = pd.read_csv(dest, sep="\t")
    for key, value in params.items():
        if key not in df.columns:
            raise KeyError(f"'{key}' is not a column of the credible sets. Available: {list(df.columns)}")
        if isinstance(value, (list, tuple, set)):
            df = df[df[key].isin(list(value))]
        else:
            df = df[df[key] == value]
    return df.reset_index(drop=True)


def get_eqtl_catalog_lbf(
    dataset_id: str,
    data_home: str | Path | None = None,
    refresh: bool = False,
    return_path: bool = False,
) -> pd.DataFrame | Path:
    """
    Retrieve per-variant SuSiE log Bayes factors for one eQTL Catalogue dataset.

    This is the input :func:`cellink.tl.coloc_susie` expects for the QTL side of a
    SuSiE-based colocalization (one column of log Bayes factors per SuSiE component
    ``L1..L10``), as opposed to the single-causal-variant approximation used by
    :func:`cellink.tl.coloc_abf`.

    Parameters
    ----------
    dataset_id : str
        eQTL Catalogue dataset ID (e.g., ``"QTD000625"``).
    data_home : str or Path, optional
        Directory to store cached files. Defaults to user data directory.
    refresh : bool, default=False
        If True, ignore cached data and re-download.
    return_path : bool, default=False
        If True, return the local cached file path instead of a DataFrame. Recommended:
        these files are ~100 MB compressed and cover every fine-mapped region in the
        dataset, so reading a whole one into memory is usually not what you want.

    Returns
    -------
    pd.DataFrame or Path
    """
    data_home = get_data_home(data_home)
    url = _resolve_eqtl_dataset_path(dataset_id, "ftp_lbf_path", data_home, refresh)
    dest = data_home / f"{dataset_id}.lbf_variable.txt.gz"
    if not dest.exists() or refresh:
        _download_file(url, dest)
    if return_path:
        return dest
    return pd.read_csv(dest, sep="\t")


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

    eqtl_datasets = get_eqtl_catalog_datasets(quant_method="ge")
    print(eqtl_datasets.head())

    # Region-restricted query: cis-eQTLs at the BACH2 locus in OneK1K memory Tregs.
    eqtl_dataset = get_eqtl_catalog_dataset_associations("QTD000625", region="6:89900000-90300000")
    print(eqtl_dataset.nsmallest(5, "pvalue"))

    credible_sets = get_eqtl_catalog_credible_sets("QTD000625")
    print(credible_sets[credible_sets.pip >= 0.9].head())
