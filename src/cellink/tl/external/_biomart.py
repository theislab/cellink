import contextlib
import io
import logging
from xml.sax.saxutils import quoteattr

import pandas as pd
import requests

from cellink.resources._utils import retry_with_backoff

logger = logging.getLogger(__name__)

BIOMART_HOSTS = {
    "GRCh37": "https://grch37.ensembl.org",
    "GRCh38": "https://www.ensembl.org",
}


class BiomartQueryError(ValueError):
    """BioMart rejected the query (unknown attribute or filter); retrying cannot help."""


def _build_query_xml(attributes: list[str], filters: dict | None, dataset: str) -> str:
    filter_xml = ""
    for name, value in (filters or {}).items():
        if isinstance(value, list | tuple | set):
            value = ",".join(map(str, value))
        filter_xml += f"<Filter name={quoteattr(name)} value={quoteattr(str(value))}/>"
    attribute_xml = "".join(f"<Attribute name={quoteattr(a)}/>" for a in attributes)
    return (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'completionStamp="1" datasetConfigVersion="0.6">'
        f'<Dataset name={quoteattr(dataset)} interface="default">{filter_xml}{attribute_xml}</Dataset></Query>'
    )


def _fetch_martservice(url: str, xml: str, timeout: float) -> str:
    """GET one BioMart query and return the TSV body, raising on anything but a complete reply."""
    with requests.Session() as session:
        no_cache = getattr(session, "cache_disabled", contextlib.nullcontext)
        with no_cache():
            response = session.get(url, params={"query": xml}, timeout=timeout)
    response.raise_for_status()
    text = response.text
    if "Query ERROR" in text[:500]:
        return text
    body, _, stamp = text.rstrip().rpartition("\n")
    if stamp.strip() != "[success]":
        raise RuntimeError(f"Incomplete BioMart reply from {response.url}: {text[:200]!r}")
    logger.debug(f"BioMart reply from {response.url}")
    return body


def query_biomart(
    attributes: list[str],
    filters: dict | None = None,
    genome_build: str = "GRCh38",
    *,
    dataset: str = "hsapiens_gene_ensembl",
    host: str | None = None,
    dtype: dict | None = None,
    timeout: float = 600,
) -> pd.DataFrame:
    """
    Query Ensembl BioMart over HTTPS without any response cache.

    Replies that do not end with BioMart's ``[success]`` completion stamp (error pages served
    with HTTP 200, cut-off downloads) raise and are fetched again by
    :func:`~cellink.resources._utils.retry_with_backoff`.

    Parameters
    ----------
    attributes : list of str
        BioMart attribute names, e.g. ``["ensembl_gene_id", "external_gene_name"]``.
    filters : dict, optional
        BioMart filters as ``{name: value}``; list values are joined with commas.
    genome_build : {"GRCh37", "GRCh38"}, default "GRCh38"
        Selects the Ensembl host (``grch37.ensembl.org`` or ``www.ensembl.org``, which redirects
        to the archive of the current release). Ignored when ``host`` is given.
    dataset : str, default "hsapiens_gene_ensembl"
        BioMart dataset name.
    host : str, optional
        Ensembl host to query instead, e.g. an archive such as
        ``"https://jun2026.archive.ensembl.org"`` to pin an Ensembl release.
    dtype : dict, optional
        Column dtypes passed to :func:`pandas.read_csv`.
    timeout : float, default 600
        Seconds to wait for the reply.

    Returns
    -------
    pd.DataFrame
        One column per attribute, named by the attribute names, with unique rows.

    Raises
    ------
    BiomartQueryError
        If BioMart rejects the query (for example an unknown attribute).
    """
    if host is None:
        if genome_build not in BIOMART_HOSTS:
            raise ValueError(f"Invalid genome_build: {genome_build}. Must be 'GRCh37' or 'GRCh38'.")
        host = BIOMART_HOSTS[genome_build]
    url = f"{host.rstrip('/')}/biomart/martservice"
    xml = _build_query_xml(attributes, filters, dataset)

    logger.info(f"Querying Ensembl BioMart at {host} ({dataset}: {', '.join(attributes)})")
    body = retry_with_backoff(lambda: _fetch_martservice(url, xml, timeout))
    if "Query ERROR" in body[:500]:
        raise BiomartQueryError(body.strip()[:500])
    if not body.strip():
        return pd.DataFrame({a: pd.Series(dtype=object) for a in attributes})
    return pd.read_csv(io.StringIO(body), sep="\t", header=None, names=attributes, dtype=dtype, low_memory=False)
