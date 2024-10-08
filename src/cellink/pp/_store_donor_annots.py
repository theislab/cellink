import logging

import pandas as pd

from typing import TYPE_CHECKING, Tuple
from anndata import AnnData
from pandas import Index

logger = logging.getLogger(__name__)


def match_donors(
    gdata: AnnData,
    scdata: AnnData,
    sc_on: str = "donor",
) -> Tuple[AnnData, AnnData]:
    """
    Match donors between genetic and single-cell data.

    This function aligns the donors in genetic and single-cell data,
    keeping only the donors' data that is present in both datasets.

    Parameters
    ----------
    gdata : AnnData
        Genetic data in AnnData format.
    scdata : AnnData
        Single-cell data in AnnData format.
    sc_on : str, optional
        Column name in `scdata.obs` that contains donor identifiers,
        by default "donor".

    Returns
    -------
    Tuple[AnnData, AnnData]
        A tuple containing the matched genetic and single-cell data.

    Notes
    -----
    - Donor's data where a donor is not present in both datasets is dropped.
    - Warnings are logged about the number of donors kept and dropped.

    Example
    -------
    >>> import anndata
    >>> gdata = anndata.AnnData(X=np.random.rand(100, 1000))
    >>> scdata = anndata.AnnData(X=np.random.rand(1000, 2000))
    >>> gdata.obs_names = [f"sample_{i}" for i in range(100)]
    >>> scdata.obs["donor"] = [f"sample_{i}" for i in range(80)] + [f"other_{i}" for i in range(920)]
    >>> matched_gdata, matched_scdata = match_donors(gdata, scdata)
    """
    # Sort single-cell data by the specified column
    scdata = scdata[scdata.obs[sc_on].sort_values().index]

    # Get unique sample identifiers from both datasets
    sc_index: Index = pd.Index(scdata.obs[sc_on].unique())
    g_index: Index = gdata.obs.index
    
    # Find common donors and all unique donors
    keep_donors: Index = sc_index.intersection(g_index)
    all_donors: Index = sc_index.union(g_index)

    # Log warnings about sample matching
    logger.warning("Keeping %s/%s donors", len(keep_donors), len(all_donors))
    logger.warning(
        "Dropping %s/%s donors from genetic data",
        len(g_index) - len(keep_donors),
        len(g_index),
    )
    logger.warning(
        "Dropping %s/%s donors from single-cell data",
        len(sc_index) - len(keep_donors),
        len(sc_index),
    )

    # Filter both datasets to keep only matched donors
    gdata = gdata[keep_donors]
    scdata = scdata[scdata.obs[sc_on].isin(keep_donors)]

    return gdata, scdata
