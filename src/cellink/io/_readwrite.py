import warnings
from pathlib import Path
from typing import Literal

from mudata import MuData
from .._core import DonorData
from anndata.io import read_elem
import h5py
from anndata.io import read_elem
from anndata._types import StorageType
import zarr
from mudata._core.mudata import ModDict, MuData
from mudata._core.io import _read_h5mu_mod
from anndata.compat import _read_attr
from mudata._core.io import _write_h5mu
from anndata._io.zarr import read_dataframe
from anndata.io import write_elem

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


def _read_mudata(group: StorageType, backed: bool = True) -> MuData:
    """
    Adapted from mudata._core.io.read_h5mu (https://github.com/scverse/mudata/blob/main/src/mudata/_core/io.py#L419).
    """

    #with h5py.File("test.h5mu", "r") as f:

    d = {}
    for k in group.keys():
        if k in ["obs", "var"]:
            d[k] = read_dataframe(group[k])
        if k == "mod":
            mods = ModDict()
            gmods = group[k]
            for m in gmods.keys():
                ad = _read_h5mu_mod(gmods[m], None, True) 
                mods[m] = ad

            mod_order = None
            if "mod-order" in gmods.attrs:
                mod_order = _read_attr(gmods.attrs, "mod-order")
            if mod_order is not None and all([m in gmods for m in mod_order]):
                mods = {k: mods[k] for k in mod_order}

            d[k] = mods
        else:
            d[k] = read_elem(group[k])

    if "axis" in group.attrs:
        d["axis"] = group.attrs["axis"]

    mu = MuData._init_from_dict_(**d)
    return mu

def _read_dd(f: h5py.File) -> DonorData:
        
    if f["G"].attrs.get("encoding-type") == "MuData":
        G = _read_mudata(group=f["G"])
    elif f["G"].attrs.get("encoding-type") == "anndata":
        G = read_elem(f["G"])
    else:
        raise ValueError("Unknown encoding type for G")

    if f["C"].attrs.get("encoding-type") == "MuData":
        C = _read_mudata(group=f["C"])
    elif f["C"].attrs.get("encoding-type") == "anndata":
        C = read_elem(f["C"])
    else:
        raise ValueError("Unknown encoding type for C")

    donor_id = f.attrs.get("donor_id", "donor")
    var_dims_to_sync = list(f.attrs.get("var_dims_to_sync", []))

    uns = {}
    uns_group = f.get("uns")
    if uns_group:
        for key in uns_group:
            uns[key] = uns_group[key][()]

    dd = DonorData(G=G, C=C, donor_id=donor_id, var_dims_to_sync=var_dims_to_sync, uns=uns) 

    return dd

def read_h5_dd(path: str) -> DonorData:
    with h5py.File(path, "r") as f:
        return _read_dd(f)

def read_zarr_dd(path: str) -> DonorData:
    with zarr.open(path, mode="r") as f:
        return _read_dd(f)
        
def read_dd(path: str, fmt: str = None) -> DonorData:
    """Read donor data from specified file paths for both the gene expression data (G) and cell-type data (C),
    and return a DonorData object containing these two datasets.

    Parameters
    ----------
    path : str | Path
        Path to the DonorData data file.

    Returns
    -------
    DonorData
        A DonorData object containing:
        - `G`: Gene expression data as an `AnnData` object or a `mu.Dataset` object.
        - `C`: Cell-type data as an `AnnData` object.

    Example
    -------
    dd = read_donordata_object('path/to/donor_data.dd.h5')
    """
    if fmt is None:
        if path.endswith(".h5") or path.endswith(".dd.h5"):
            fmt = "h5"
        elif path.endswith(".zarr") or path.endswith(".dd.zarr"):
            fmt = "zarr"
        else:
            raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")

    if fmt == "h5":
        return read_h5_dd(path)
    elif fmt == "zarr":
        return read_zarr_dd(path)
    else:
        raise ValueError("Unknown format: use 'h5' or 'zarr'.")
        