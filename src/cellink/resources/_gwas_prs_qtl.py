import logging
import re
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd
import requests

from cellink.resources._utils import _cache_df, _to_dataframe, get_data_home


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

    Matches anything ending in .tsv, .txt, .zip, or .gz -- some
    pre-harmonisation-era deposits ship a plain/zipped .txt instead of
    .tsv.gz, or a bare ".gz" with no .tsv/.txt in the name at all (confirmed
    directly: a sleep-duration GWAS whose only file is
    "..._sumstats.txt.zip", and a major-depression GWAS whose only file is
    literally "MDD2018_ex23andMe.gz"). Since .tsv.gz/.txt.gz already end in
    ".gz", matching bare ".gz" covers all of those cases in one pattern.
    Widening this far risks also matching README/LICENSE/CHANGELOG files
    that live in the same directory (which the original .tsv.gz-only regex
    never collided with) -- hence the explicit exclusion below.
    """
    files = re.findall(r'href="([^"]*\.(?:tsv|txt|zip|gz))"', html)
    return [f for f in files if not _NON_DATA_FILENAME.search(f)]


logging.basicConfig(level=logging.INFO)

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
PGS_API_BASE = "https://www.pgscatalog.org/rest"
EQTL_API_BASE = "https://www.ebi.ac.uk/eqtl/api/v3"


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
    return _fetch(f"studies/{accession_id}", params=params, paginate=False)

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
        raise ValueError("translate_to_build requires return_path=False — liftover operates on an in-memory DataFrame.")
    study_meta = _fetch(f"{GWAS_API_BASE}/studies/{accession_id}", params=params, paginate=False)

    if "full_summary_stats" not in study_meta:
        raise ValueError(f"Study {accession_id} does not have full summary statistics available")

    base_url = study_meta["full_summary_stats"]
    harmonised_url = f"{base_url}/harmonised"

    normalized_build = _normalize_build(genome_build) if genome_build else None

    def build_priority(filename):
        """Assign priority score to filename based on genome build."""
        filename_lower = filename.lower()
        # See get_build_from_filename below for why ".h.tsv.gz" is GRCh38 even
        # without a "grch38"/"hg38" substring -- same fix, same reasoning,
        # needed here too since this is what the tie-break fallback uses.
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
        # The EBI harmonisation pipeline always standardises to GRCh38, but the
        # file itself is just named "<accession>.h.tsv.gz" -- no "grch38"/"hg38"
        # substring. Without this check, a GRCh38 request can't recognize the
        # harmonised file as a match and falls through to an arbitrary
        # listing-order tie-break against other same-priority ("unknown" build)
        # files -- confirmed directly on GCST005538 (Sarcoidosis), where this
        # picked a same-directory "-Build36.f.tsv.gz" duplicate (misleadingly
        # named -- its actual coordinates are GRCh37, identical to the base-dir
        # file, not GRCh36) that has zero rsIDs, over the real harmonised file
        # sitting right next to it with rsIDs for ~100% of rows -- causing
        # MAGMA/S-LDSC's downstream position-based BIM matching to only hit
        # 160/128,705 SNPs.
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
                # Multiple files can legitimately share a build (e.g. a
                # same-build "-Build38.f.tsv.gz" frozen copy sitting right next
                # to the true ".h.tsv.gz" harmonised file) -- prefer the real
                # harmonised one, since EBI's harmonisation pipeline is the
                # authoritative rsID-resolution step and other same-build
                # copies are often pre-harmonisation snapshots with no rsIDs
                # at all. Confirmed directly: GCST90078640 (AnorexiaNervosa)
                # has exactly this pair, and picking the "-Build38.f" file
                # (which sorts first alphabetically, so a plain "first match"
                # selected it every time) fed S-LDSC a file with 0 rsIDs,
                # dropping ~99.6% of SNPs during position-based BIM annotation
                # (89,215 raw rows -> 357 SNPs surviving) versus the
                # harmonised file's real hm_rsid for ~100% of its 82,088 rows.
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

    # If user specified a genome build, prefer a build-specific file over the
    # truly-harmonised (.h.tsv.gz) one. EBI sometimes stores these
    # build-specific files in the harmonised/ directory alongside the
    # harmonised one (e.g. "*-Build37.f.tsv.gz"), and sometimes in the base
    # study directory — so both locations are checked.
    #
    # The harmonised (.h.tsv.gz) file is *always* GRCh38 (that's what EBI's
    # harmonisation pipeline standardises to) -- so it should only be
    # excluded when a different build (GRCh37) was requested. Unconditionally
    # excluding it here (regardless of requested build) meant any accession
    # with ONLY a harmonised file and no separate build-specific copy -- the
    # norm for studies deposited after EBI's ~2020 harmonisation rollout --
    # reported "no file found" for GRCh38 requests even though a perfectly
    # good GRCh38 file was sitting right there. Confirmed directly: several
    # accessions with a real, correctly-formed harmonised/*.h.tsv.gz file
    # (verified via direct FTP listing) failed here with genome_build="GRCh38".
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

        except Exception as e:
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

            except Exception as e:
                logging.warning(f"Could not parse base directory listing ({e})")

        # If still no file found, try standard naming conventions
        if not url:
            logging.info("Trying standard naming conventions for requested build")
            
            if normalized_build == 'GRCh38':
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
                        logging.info(f"Found file via standard naming convention (build: {detected_build}): {test_filename}")
                        break
                except:
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

        except Exception as e:
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

            except Exception as e:
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
                        logging.info(f"Found file via standard naming convention (build: {detected_build}): {test_filename}")
                        break
                except:
                    continue

    if not url:
        raise ValueError(f"Could not find summary statistics file for {accession_id}")

    # dest's extension must match the actual source file (url can now resolve
    # to a plain .tsv, .tsv.gz, or .zip, see the widened regexes above) --
    # saving an uncompressed file under a ".gz" name would make the
    # pd.read_csv(..., compression=...) call below fail to decode it.
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
    except Exception as e:
        raise RuntimeError(f"Failed to download summary statistics from {url}: {e}")

    if return_path:
        return dest

    # pandas' compression="zip" requires the archive contain exactly one
    # member (true for the one real-world case this handles -- a
    # single-file sleep-duration GWAS summary-stats zip); pandas raises its
    # own clear error if that assumption doesn't hold for some other file.
    #
    # sep=r"\s+" (not a hardcoded tab) -- confirmed directly: some
    # non-harmonised deposits (e.g. a myopia/refractive-error GWAS) are
    # space-delimited, not tab-delimited. A strict "\t" silently "succeeds"
    # on these (no error, no warning) but collapses the entire line into one
    # column, which then fails much later and less legibly (e.g. missing
    # CHR/BP/P columns) than if the read itself had failed loudly.
    data = pd.read_csv(dest, compression=compression, sep=r"\s+")
    if translate_to_build:
        data = liftover_gwas_summary_stats(data, source_build=detected_build or "GRCh38", target_build=translate_to_build)
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
    except ImportError:
        raise ImportError(
            "The 'liftover' package is required for build translation. "
            "Install it with: pip install cellink[datasets]"
        )

    converter = get_lifter(src, tgt, one_based=True)

    chroms = df[chrom_col].astype(str).tolist()
    positions = df[pos_col].tolist()
    results = [converter[c][p] for c, p in zip(chroms, positions)]
    new_positions = [r[0][1] if r else None for r in results]

    out = df.copy()
    out[pos_col] = new_positions
    if drop_failed:
        n_failed = sum(p is None for p in new_positions)
        if n_failed:
            logging.warning(f"liftover: dropped {n_failed:,} rows with no mapping from {source_build} to {target_build}")
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


def get_eqtl_catalog_datasets(
    data_home: str | Path | None = None, max_pages: int | None = None, refresh: bool = False, **params: Any
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
    data_home: str | Path | None = None,
    refresh: bool = False,
    return_path: bool = False,
    **params: Any,
) -> pd.DataFrame | Path:
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
