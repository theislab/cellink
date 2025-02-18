import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from anndata import AnnData
import muon as mu
import scanpy as sc
from anndata.utils import asarray
from typing import Literal

import cellink as cl

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)

def read_donordata_objects(path_C: str | Path, path_G: str, *, backed_C: Literal["r", "r+"] | bool | None = None, backed_G: Literal["r", "r+"] | bool | None = None, **kwargs) -> AnnData:
    """    Read donor data from specified file paths for both the gene expression data (G) and cell-type data (C),
    and return a DonorData object containing these two datasets.

    Parameters
    ----------
    path_C : str | Path
        Path to the cell-type data file, which can either be in `.h5ad` or `.h5mu` format.
    path_G : str
        Path to the gene expression data file, expected to be in `.h5ad` format.
    backed_C : Literal["r", "r+"] | bool | None, optional
        If the file at `path_C` is to be read in backed mode (if it's too large to load into memory). Can be `'r'` 
        (read-only) or `'r+'` (read-write), or a boolean indicating whether to back the file.
    backed_G : Literal["r", "r+"] | bool | None, optional
        Same as `backed_C`, but for the gene expression data file at `path_G`.
    **kwargs : dict, optional
        Additional keyword arguments passed directly to the `sc.read_h5ad` or `mu.read` functions when reading data.

    Returns
    -------
    DonorData
        A DonorData object containing:
        - `G`: Gene expression data as an `AnnData` object or a `mu.Dataset` object.
        - `C`: Cell-type data as an `AnnData` object.

    Example
    -------
    dd = read_donordata_objects('cell_data.h5mu', 'gene_data.h5ad')
    """

    if path_C.endswith(".h5mu"):
        gdata = mu.read(path_G, backed=backed_G, **kwargs)
    else:
        gdata = sc.read_h5ad(path_G, backed=backed_G, **kwargs)
    if path_C.endswith(".h5mu"):
        adata = mu.read(path_C, backed=backed_C, **kwargs)
    else:
        adata = sc.read_h5ad(path_C, backed=backed_C, **kwargs)
    
    dd = cl.DonorData(G=gdata, C=adata, **kwargs).copy()

    return dd
