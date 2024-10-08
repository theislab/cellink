import warnings
from pathlib import Path

import dask.array
import dask.dataframe as dd
import pandas as pd
import sgkit as sg
from anndata import AnnData
import dask
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", message="The return type of `Dataset.dims` will be changed", category=FutureWarning)


def _get_snp_index(
    var: pd.DataFrame | dd.DataFrame, chrom: str = "chrom", pos: str = "position", a0: str = "a0", a1: str = "a1"
):
    index_info = var[[chrom, pos, a0, a1]].astype(str)
    index = index_info[chrom] + "_" + index_info[pos] + "_" + index_info[a0] + "_" + index_info[a1]
    return index


def read_sgkit_zarr(path: str | Path) -> AnnData:
    """Read SgKit Zarr Format"""
    _gdata = sg.load_dataset(path)
    del _gdata.attrs["vcf_header"]
    X = _gdata.call_genotype.data.sum(-1).T
    obs = pd.DataFrame(index=_gdata.sample_id.data.compute())
    var = _gdata.drop_dims(set(_gdata.dims.keys()).difference({"variants"}))
    var = var.to_dataframe()
    var.columns = var.columns.str.replace("variant_", "")
    contig_mapping = dict(enumerate(_gdata.contig_id.data.compute()))
    var["chrom"] = var.contig.map(contig_mapping)
    alleles = _gdata.variant_allele.data
    var["a0"] = alleles[:, 0]
    var["a1"] = alleles[:, 1]
    first_cols = ["chrom", "position", "a0", "a1"]
    var = var[first_cols + [c for c in var.columns if c not in first_cols]]
    index = _get_snp_index(var)
    var.index = index

    varm = {}
    if "filter_id" in _gdata and "variant_filter" in _gdata:
        columns = _gdata.filter_id.data.compute()
        data = _gdata["variant_filter"].data.compute()
        filters = pd.DataFrame(data, columns=columns, index=var.index)
        varm["filter"] = filters

    gdata = AnnData(X=X, obs=obs, var=var, varm=varm)

    return gdata
