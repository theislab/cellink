import warnings
from typing import Literal

import h5py
import zarr
from anndata._io.zarr import read_dataframe
from anndata._types import StorageType
from anndata.compat import _read_attr
from anndata.experimental import read_dispatched
from anndata.io import read_elem
from mudata._core.io import _read_h5mu_mod
from mudata._core.mudata import ModDict, MuData

from cellink._core import DonorData
from cellink._core.schema import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION
from cellink.io._pgen import lazy_anndata_zarr_callback

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
                ad = _read_h5mu_mod(gmods[m], None, False)
                mods[m] = ad

            mod_order = None
            if "mod-order" in gmods.attrs:
                mod_order = _read_attr(gmods.attrs, "mod-order")
            if mod_order is not None and all(m in gmods for m in mod_order):
                mods = {k: mods[k] for k in mod_order}

            d[k] = mods
        else:
            d[k] = read_elem(group[k])

    if "axis" in group.attrs:
        d["axis"] = group.attrs["axis"]

    if hasattr(MuData, "_init_from_dict_"):
        mu = MuData._init_from_dict_(**d)          # mudata < 0.4
    else:
        d["data"] = d.pop("mod", ModDict())        # mudata >= 0.4
        mu = MuData(**d)
    return mu


def _read_dd(
    f: h5py.File, lazy: bool = False, backend: Literal["dask", "zarr"] = "dask", n_threads: int | None = None
) -> DonorData:
    """
    Internal function to read a DonorData object from an HDF5 or Zarr group.

    Reads donor-level genotype data (`G`) and cell-level expression data (`C`),
    reconstructs them as either AnnData or MuData objects, and collects additional
    attributes such as donor ID, synchronized variable dimensions, and unstructured
    metadata.

    Parameters
    ----------
    f : h5py.File or zarr.Group
        An open HDF5 file handle or Zarr group containing the DonorData object.
    lazy : bool, default=False
        If True, read `G`/`C` via :func:`lazy_anndata_zarr_callback`
        so a dense `X` stays Dask-backed instead of being materialized, which
        lets callers select a subset (e.g. one cell type) before ever loading
        the full array into memory.
    backend : {"dask", "zarr"}
        Backing of a dense `X`/`layers` entry of a Zarr store when ``lazy``; see
        :func:`cellink.io.read_pgen_zarr`. HDF5 datasets become Dask arrays.
    n_threads : int, optional
        Threads for ``backend="zarr"`` genotype reads.

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

    def _callback(func, name, elem, iospec):
        return lazy_anndata_zarr_callback(func, name, elem, iospec, backend=backend, n_threads=n_threads)

    def _read_anndata(group):
        return read_dispatched(group, callback=_callback) if lazy else read_elem(group)

    top_encoding_type = f.attrs.get("encoding-type")
    if top_encoding_type is not None and top_encoding_type != DONORDATA_ENCODING_TYPE:
        raise ValueError(
            f"Expected encoding-type={DONORDATA_ENCODING_TYPE!r} at the top level of a DonorData "
            f"file, found {top_encoding_type!r}; this file was not written by DonorData.write_h5_dd/"
            "write_zarr_dd."
        )
    top_encoding_version = f.attrs.get("encoding-version")
    if top_encoding_version is not None and top_encoding_version != DONORDATA_ENCODING_VERSION:
        warnings.warn(
            f"DonorData file has encoding-version={top_encoding_version!r}, this cellink reads "
            f"encoding-version={DONORDATA_ENCODING_VERSION!r}; reading anyway, but downstream "
            "behavior for older/newer schema versions is not guaranteed.",
            stacklevel=2,
        )

    if f["G"].attrs.get("encoding-type") == "MuData":
        G = _read_mudata(group=f["G"])
    elif f["G"].attrs.get("encoding-type") == "anndata":
        G = _read_anndata(f["G"])
    else:
        raise ValueError("Unknown encoding type for G")

    if f["C"].attrs.get("encoding-type") == "MuData":
        C = _read_mudata(group=f["C"])
    elif f["C"].attrs.get("encoding-type") == "anndata":
        C = _read_anndata(f["C"])
    else:
        raise ValueError("Unknown encoding type for C")

    donor_id = f.attrs.get("donor_id", "donor")
    var_dims_to_sync = list(f.attrs.get("var_dims_to_sync", []))

    uns = {}
    uns_group = f.get("uns")
    if uns_group:
        for key in uns_group:
            node = uns_group[key]
            try:
                uns[key] = read_elem(node)
            except Exception:
                uns[key] = node[()] if hasattr(node, "shape") else node

    dd = DonorData(G=G, C=C, donor_id=donor_id, var_dims_to_sync=var_dims_to_sync, uns=uns)

    return dd


def read_h5_dd(path: str, lazy: bool = False) -> DonorData:
    """
    Read a DonorData object from an HDF5 file on disk.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing serialized DonorData.
    lazy : bool, default=False
        If True, a dense `X` and dense layers of `G` and `C` are Dask arrays that read
        their blocks from the file when they are computed, as with :func:`read_zarr_dd`,
        so a subset can be selected before any genotypes are loaded. Blocks follow the
        HDF5 chunks, or span at most 4096 rows and columns for unchunked datasets. The
        file must stay in place while these arrays are used. Sparse matrices, MuData
        modalities and all annotations are read into memory.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`G`), cell expression (`C`), and metadata.
    """
    with h5py.File(path, "r") as f:
        return _read_dd(f, lazy=lazy)


def read_zarr_dd(path: str, backend: Literal["dask", "zarr"] = "dask", n_threads: int | None = None) -> DonorData:
    """
    Read a DonorData object from a Zarr store on disk.

    `G` and `C` are read lazily: a dense `X` stays Dask-backed rather than
    being materialized on load, so a caller can select a subset (e.g. one
    cell type via ``dd.sel(C_obs=...)``) and only that subset is ever read
    off disk once it's actually computed/materialized.

    Parameters
    ----------
    path : str
        Path to the Zarr store containing serialized DonorData.
    backend : {"dask", "zarr"}
        Backing of a dense `X`/`layers` entry. ``"zarr"`` gives `G.X` the parallel
        genotype reader (:func:`cellink.io.read_pgen_zarr`), fastest for subsetting donors.
    n_threads : int, optional
        Threads for ``backend="zarr"`` genotype reads. Default: available cores, at most 16.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`G`), cell expression (`C`), and metadata.
        Missing genotype calls of a 2-bit `G.X` (see :meth:`cellink.DonorData.write_zarr_dd`)
        are ``nan`` and `G.X` is float32; with ``backend="zarr"`` `G.X` holds the
        stored int8 codes, with -1 for missing calls.
    """
    f = zarr.open(path, mode="r")
    dd = _read_dd(f, lazy=True, backend=backend, n_threads=n_threads)
    if backend != "zarr" and isinstance(f["G"], zarr.Group) and "X" in f["G"]:
        dd.G.X = _two_bit_missing_to_nan(f["G"]["X"], dd.G.X)
    return dd


def _two_bit_missing_to_nan(stored, X):
    """Return a 2-bit genotype matrix with its missing calls (-1) as nan.

    ``X`` is unchanged unless ``stored`` is 2-bit encoded and may hold missing calls: its
    ``n_missing`` attribute is above 0, or absent.
    """
    from cellink.io._codecs import Pack2Bit
    from cellink.io._pgen import _missing_to_nan

    if not isinstance(stored, zarr.Array) or not any(isinstance(c, Pack2Bit) for c in stored.filters):
        return X
    if stored.attrs.get("n_missing", 1) == 0:
        return X
    return _missing_to_nan(X)


def read_dd(path: str, fmt: str = None, lazy: bool = False) -> DonorData:
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
    lazy : bool, default=False
        Passed to :func:`read_h5_dd` for HDF5 files. Zarr stores are always read lazily
        (:func:`read_zarr_dd`).

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
        return read_h5_dd(path, lazy=lazy)
    elif fmt == "zarr":
        return read_zarr_dd(path)
    else:
        raise ValueError("Unknown format: use 'h5' or 'zarr'.")
