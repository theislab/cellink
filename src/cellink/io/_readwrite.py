import warnings
from collections.abc import Iterator
from typing import Literal

import anndata as ad
import dask.array as da
import donordata
import numpy as np
import zarr
from donordata import DonorData
from mudata import MuData

from cellink.io._codecs import Pack2Bit, to_genotype_codes
from cellink.io._pgen import _lazy_matrix, _missing_to_nan

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)

_LEVEL_GROUPS = {"donor": "D", "sample": "S", "cell": "C"}


def _anndatas(dd: DonorData, f: zarr.Group | None = None) -> Iterator[tuple[ad.AnnData, zarr.Group | None]]:
    """Yield every AnnData of ``dd``, and the modalities of MuData levels, with its group in the store ``f``."""
    for level, container in dd.levels.items():
        group = None
        if f is not None:
            key = _LEVEL_GROUPS[level]
            group = f[key] if key in f or level != "donor" else f["G"]
        if isinstance(container, MuData):
            for name, mod in container.mod.items():
                yield mod, None if group is None else group["mod"][name]
        else:
            yield container, group


def _set_matrix(adata: ad.AnnData, name: str, value) -> None:
    if name == "X":
        adata.X = value
    else:
        adata.layers[name.removeprefix("layers/")] = value


def _matrices(adata: ad.AnnData) -> list[tuple[str, object]]:
    layers = [(f"layers/{key}", value) for key, value in adata.layers.items() if key is not None]
    return [("X", adata.X), *layers]


def _restore_matrices(
    dd: DonorData,
    f: zarr.Group | None = None,
    backend: Literal["dask", "zarr"] = "dask",
    n_threads: int | None = None,
) -> None:
    """Load sparse `X` and layers into memory and, from the Zarr store ``f``, back dense ones with cellink's reader.

    Missing calls of a 2-bit `D.X` become ``nan``, unless ``backend="zarr"``.
    """
    donor_level = dd.levels["donor"]
    for adata, group in _anndatas(dd, f):
        for name, value in _matrices(adata):
            if isinstance(value, ad.abc.CSRDataset | ad.abc.CSCDataset):
                _set_matrix(adata, name, value.to_memory())
            elif group is not None and isinstance(value, da.Array) and isinstance(group.get(name), zarr.Array):
                if adata.is_view:
                    raise ValueError(
                        "The levels of this store do not match, so DonorData dropped or reordered units while "
                        "reading it. Read it with donordata.read_zarr_dd and write it again."
                    )
                stored = group[name]
                matrix = _lazy_matrix(stored, backend, n_threads)
                if adata is donor_level and name == "X" and backend != "zarr":
                    matrix = _two_bit_missing_to_nan(stored, matrix)
                _set_matrix(adata, name, matrix)


def read_h5_dd(path: str, lazy: bool = False) -> DonorData:
    """
    Read a DonorData object from an HDF5 file on disk.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing serialized DonorData.
    lazy : bool, default=False
        If True, a dense `X` and dense layers of every level are Dask arrays that read
        their blocks from the file when they are computed, as with :func:`read_zarr_dd`,
        so a subset can be selected before any genotypes are loaded. Blocks follow the
        HDF5 chunks, or span at most 4096 rows and columns for unchunked datasets. The
        file must stay in place while these arrays are used. Sparse matrices and all
        annotations are read into memory.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`D`), cell expression (`C`), and metadata.
    """
    dd = donordata.read_h5_dd(path, lazy=lazy)
    if lazy:
        _restore_matrices(dd)
    return dd


def read_zarr_dd(path: str, backend: Literal["dask", "zarr"] = "dask", n_threads: int | None = None) -> DonorData:
    """
    Read a DonorData object from a Zarr store on disk.

    A dense `X` stays lazily backed rather than being materialized on load, so a
    caller can select a subset (e.g. one cell type via ``dd.sel(C_obs=...)``) and
    only that subset is ever read off disk once it's actually computed/materialized.
    Sparse matrices and all annotations are read into memory.

    Parameters
    ----------
    path : str
        Path to the Zarr store containing serialized DonorData.
    backend : {"dask", "zarr"}
        Backing of a dense `X`/`layers` entry. ``"zarr"`` gives `D.X` the parallel
        genotype reader (:func:`cellink.io.read_pgen_zarr`), fastest for subsetting donors.
    n_threads : int, optional
        Threads for ``backend="zarr"`` genotype reads. Default: available cores, at most 16.

    Returns
    -------
    DonorData
        A DonorData object with genotype (`D`), cell expression (`C`), and metadata.
        Missing genotype calls of a 2-bit `D.X` (see :func:`cellink.io.write_zarr_dd`)
        are ``nan`` and `D.X` is float32; with ``backend="zarr"`` `D.X` holds the
        stored int8 codes, with -1 for missing calls.
    """
    dd = donordata.read_zarr_dd(path)
    _restore_matrices(dd, zarr.open(str(path), mode="r"), backend=backend, n_threads=n_threads)
    return dd


def _two_bit_missing_to_nan(stored, X):
    """Return a 2-bit genotype matrix with its missing calls (-1) as nan.

    ``X`` is unchanged unless ``stored`` is 2-bit encoded and may hold missing calls: its
    ``n_missing`` attribute is above 0, or absent.
    """
    if not isinstance(stored, zarr.Array) or not any(isinstance(c, Pack2Bit) for c in stored.filters):
        return X
    if stored.attrs.get("n_missing", 1) == 0:
        return X
    return _missing_to_nan(X)


def read_dd(path: str, fmt: str = None, lazy: bool = False) -> DonorData:
    """
    Read a DonorData object from disk in either HDF5 or Zarr format.

    This function automatically detects the file format from the file extension
    unless explicitly specified.

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
    fmt = _format(str(path), fmt)
    if fmt == "h5":
        return read_h5_dd(path, lazy=lazy)
    return read_zarr_dd(path)


def _format(path: str, fmt: str | None) -> str:
    if fmt is None:
        if path.endswith(".h5"):
            return "h5"
        if path.endswith(".zarr"):
            return "zarr"
        raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")
    if fmt not in ("h5", "zarr"):
        raise ValueError("Unknown format: use 'h5' or 'zarr'.")
    return fmt


def write_h5_dd(dd: DonorData, path: str) -> None:
    """Write a DonorData object to an HDF5 file.

    Parameters
    ----------
    dd : DonorData
        The object to write.
    path : str
        Destination, conventionally ending in ``.dd.h5``.
    """
    dd.write_h5_dd(path)


def write_zarr_dd(
    dd: DonorData,
    path: str,
    x_chunks: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
    genotype_encoding: Literal["2bit"] | None = "2bit",
) -> None:
    """Write a DonorData object to a Zarr store, with the genotypes in `D.X` stored as 2-bit calls.

    Parameters
    ----------
    dd : DonorData
        The object to write.
    path : str
        Destination, conventionally ending in ``.dd.zarr``.
    x_chunks : tuple[int, int] | dict[str, tuple[int, int]], optional
        Chunk shape for a dense `X` and dense layers, e.g. `(4096, 4096)`. Pass a
        dict (`{"D": ..., "C": ...}`) to set the levels independently. If not
        given: a dask-backed `X` keeps its own existing chunks (via
        `dask.array.to_zarr`); a plain array gets chunks capped at 4096
        per axis. Only applies to a dense `X`; sparse matrices are unaffected.
    genotype_encoding : {"2bit", None}
        Storage of a dense `D.X`. ``None`` writes it unencoded, with its
        values and dtype unchanged. ``"2bit"`` (default) stores it as 2-bit
        packed genotype calls, four per byte; a variant chunk size not
        divisible by 4 is rounded up to the next multiple of 4, with a
        warning. Only `D.X` is affected: `obs`, `var`, `obsm`,
        `layers`, `uns`, `S` and `C` are written unchanged. With ``"2bit"``:

        - Integer values 0, 1 and 2 are stored as they are. Negative
          integers count as missing.
        - Floating-point values, e.g. imputed dosages, are rounded to the
          nearest integer, halves to even (0.5 becomes 0, 1.5 becomes 2).
          NaN counts as missing.
        - Missing values stay missing: stored as the 2-bit code 3.
          :func:`cellink.io.read_zarr_dd` returns them as ``nan`` (and `D.X`
          as float32); with ``backend="zarr"`` it returns the stored codes,
          with -1 for missing.
        - A warning reports how many non-integer values were rounded and
          how many values are missing.
        - Every value must round to 0, 1 or 2: integers 0-2, or floats in
          [-0.5, 2.5). If any value lies outside, the whole `D.X` is written
          unencoded, with its original values and dtype, and a warning.
        - Continuous data inside [-0.5, 2.5) cannot be told apart from
          dosages and is rounded. Pass ``None`` to keep such data unchanged.

    Example
    -------
    >>> write_zarr_dd(dd, "path/to/donor_data.dd.zarr")
    """
    if genotype_encoding not in ("2bit", None):
        raise ValueError(f"genotype_encoding must be '2bit' or None, got {genotype_encoding!r}")
    dd.write_zarr_dd(path, x_chunks=x_chunks)
    if genotype_encoding is None or dd.D_type != "anndata":
        return
    if isinstance(zarr.open_group(str(path), mode="r")["D"].get("X"), zarr.Array):
        chunks = x_chunks.get("D") if isinstance(x_chunks, dict) else x_chunks
        _write_genotypes(f"{path}/D/X", dd.D.X, chunks)


def write_dd(dd: DonorData, path: str, fmt: Literal["h5", "zarr"] | None = None) -> None:
    """Write a DonorData object to disk, choosing the format from the extension unless `fmt` is given.

    Zarr stores are written with :func:`write_zarr_dd`, so `D.X` is stored as 2-bit genotype calls.

    Parameters
    ----------
    dd : DonorData
        The object to write.
    path : str
        Destination ending in ``.h5`` or ``.zarr``. Without an extension, ``.dd.h5`` or
        ``.dd.zarr`` is appended according to `fmt`.
    fmt : {'h5', 'zarr'}, optional
        File format to use. If None (default), inferred from the file extension.
    """
    path = str(path)
    if _format(path, fmt) == "h5":
        write_h5_dd(dd, path if path.endswith(".h5") else path + ".dd.h5")
    else:
        write_zarr_dd(dd, path if path.endswith(".zarr") else path + ".dd.zarr")


def _write_genotypes(path: str, X, chunks: tuple[int, int] | None) -> None:
    """Replace the genotype matrix at ``path`` by 2-bit packed calls.

    If the values are not genotypes (outside 0-2), the matrix is written unencoded with a warning.
    """
    packed_chunks = chunks
    if chunks is not None and chunks[1] % 4:
        rounded = -(-chunks[1] // 4) * 4
        warnings.warn(
            f"2-bit genotype encoding packs 4 variants per byte; the variant chunk size {chunks[1]} is "
            f"rounded up to {rounded}.",
            UserWarning,
            stacklevel=3,
        )
        packed_chunks = (chunks[0], rounded)
    try:
        n_missing = _write_genotypes_2bit(path, X, packed_chunks)
    except ValueError:
        warnings.warn(
            "D.X has values outside the 0-2 genotype range; it is written without 2-bit encoding.",
            UserWarning,
            stacklevel=3,
        )
        _write_dense(path, X, chunks)
        return
    arr = zarr.open_array(path, mode="r+")
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"
    arr.attrs["n_missing"] = n_missing  # read_zarr_dd turns them into nan


def _write_genotypes_2bit(path: str, X, chunks: tuple[int, int] | None) -> int:
    """Write a genotype matrix as 2-bit packed int8 calls, one column block at a time.

    Values are converted with :func:`cellink.io._codecs.to_genotype_codes`:
    floats are rounded to the nearest integer (halves to even), NaN and negative
    integers are missing and kept as missing (2-bit code 3, decoded as -1), and
    a warning reports the counts of rounded and missing values. Raises ValueError if any value does not round to
    0-2 (integers above 2, floats outside [-0.5, 2.5)); the caller then writes the
    matrix unencoded. Returns the number of missing values.
    """
    from zarr.codecs import BloscCodec

    n, m = X.shape
    chunks = chunks if chunks is not None else (min(n, 4096), min(-(-m // 4) * 4, 4096))
    z = zarr.create_array(
        store=path,
        shape=(n, m),
        chunks=chunks,
        dtype="int8",
        filters=[Pack2Bit()],
        compressors=BloscCodec(cname="lz4hc", clevel=5, shuffle="noshuffle", typesize=1),
        fill_value=0,
        overwrite=True,
    )
    n_frac = n_missing = 0
    for j in range(0, m, chunks[1]):
        block = X[:, j : j + chunks[1]]
        block = block.compute() if isinstance(block, da.Array) else np.asarray(block)
        codes, frac, miss = to_genotype_codes(block)
        z[:, j : j + chunks[1]] = codes
        n_frac, n_missing = n_frac + frac, n_missing + miss
    if n_frac:
        warnings.warn(
            f"D.X contains {n_frac:,} imputed (non-integer) values; the 2-bit genotype store keeps them rounded "
            "to the nearest integer. Pass genotype_encoding=None to keep the original values.",
            UserWarning,
            stacklevel=4,
        )
    if n_missing:
        warnings.warn(
            f"D.X contains {n_missing:,} missing values; the 2-bit genotype store keeps them missing, "
            "read back as nan (-1 with read_zarr_dd(backend='zarr')).",
            UserWarning,
            stacklevel=4,
        )
    return n_missing


def _write_dense(path: str, X, chunks: tuple[int, int] | None) -> None:
    """Write a dense matrix unencoded to the Zarr array at ``path``."""
    if isinstance(X, da.Array):
        X = X.rechunk(chunks) if chunks is not None else X
        da.to_zarr(X, path, overwrite=True)
    else:
        array_chunks = chunks if chunks is not None else tuple(min(dim, 4096) for dim in X.shape)
        z = zarr.open_array(path, mode="w", shape=X.shape, chunks=array_chunks, dtype=X.dtype)
        z[:] = np.asarray(X)
    arr = zarr.open_array(path, mode="r+")
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"
