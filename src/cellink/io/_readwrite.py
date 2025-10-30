import warnings

import h5py
import zarr
from anndata._io.specs.registry import read_elem
from anndata._io.zarr import read_dataframe
from anndata._types import StorageType
from anndata.compat import _read_attr
from anndata.io import read_elem
from mudata import MuData
from mudata._core.io import _read_h5mu_mod
from mudata._core.mudata import ModDict, MuData

from .._core import DonorData

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


def _read_mudata(group: StorageType, backed: bool = True) -> MuData:
    """
    Internal function to read a MuData object from a storage group (HDF5 or Zarr).

    This function reconstructs a MuData object from its serialized representation
    inside an HDF5 or Zarr group. It handles subcomponents such as observations,
    variables, modalities (`mod`), and additional elements stored in the group.

    Parameters
    ----------
    group : StorageType
        Storage group (HDF5 or Zarr) containing the serialized MuData structure.
    backed : bool, default=True
        If True, enables reading in backed mode for efficiency. Currently only
        affects reading of AnnData modalities.

    Returns
    -------
    MuData
        A MuData object reconstructed from the storage group.

    Notes
    -----
    - Adapted from `mudata._core.io.read_h5mu`.
    - Preserves modality ordering if the `mod-order` attribute is present in the group.
    """

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
    """
    Internal function to read a DonorData object from an HDF5 file handle.

    Reads donor-level genotype data (`G`) and cell-level expression data (`C`),
    reconstructs them as either AnnData or MuData objects, and collects additional
    attributes such as donor ID, synchronized variable dimensions, and unstructured
    metadata.

    Parameters
    ----------
    f : h5py.File
        An open HDF5 file handle containing the DonorData object.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`G`), cell expression (`C`), donor metadata,
        and unstructured annotations.

    Raises
    ------
    ValueError
        If the encoding type of `G` or `C` is not recognized.
    """

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
    """
    Read a DonorData object from an HDF5 file on disk.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing serialized DonorData.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`G`), cell expression (`C`), and metadata.
    """
        
    with h5py.File(path, "r") as f:
        return _read_dd(f)


def read_zarr_dd(path: str) -> DonorData:
    """
    Read a DonorData object from a Zarr store on disk.

    Parameters
    ----------
    path : str
        Path to the Zarr store containing serialized DonorData.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`G`), cell expression (`C`), and metadata.
    """
        
    with zarr.open(path, mode="r") as f:
        return _read_dd(f)


def read_dd(path: str, fmt: str = None) -> DonorData:
    """
    Read a DonorData object from disk in either HDF5 or Zarr format.

    This function automatically detects the file format from the file extension
    unless explicitly specified, and reconstructs a DonorData object containing
    donor-level genotype data (`G`), cell-level expression data (`C`), and
    associated metadata.

    Parameters
    ----------
    path : str
        Path to the DonorData file (`.h5`, `.dd.h5`, `.zarr`, or `.dd.zarr`).
    fmt : {'h5', 'zarr'}, optional
        File format to use. If None (default), inferred from the file extension.

    Returns
    -------
    DonorData
        A DonorData object with:
        - `G`: Genotype data as AnnData or MuData.
        - `C`: Cell-level expression data as AnnData or MuData.
        - `donor_id`: Donor identifier string.
        - `var_dims_to_sync`: List of variable dimensions synchronized across `G` and `C`.
        - `uns`: Unstructured metadata dictionary.

    Raises
    ------
    ValueError
        If the file extension cannot be mapped to a known format or if `fmt`
        is not one of {'h5', 'zarr'}.

    Examples
    --------
    >>> dd = read_dd("donor_data.dd.h5")
    >>> dd = read_dd("donor_data.dd.zarr")
    >>> dd = read_dd("custom_data.h5", fmt="h5")
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
