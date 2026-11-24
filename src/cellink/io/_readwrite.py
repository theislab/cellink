import copy
import warnings
from collections.abc import Iterator
from typing import Literal

import anndata as ad
import dask.array as da
import hidata
import numpy as np
import scipy.sparse as sp
import zarr
from hidata import HiData
from mudata import MuData

from cellink.io._codecs import Pack2Bit, to_genotype_codes
from cellink.io._pgen import _lazy_matrix, _missing_to_nan

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)

_LEVEL_GROUPS = {"group": "G", "subgroup": "S", "cell": "C"}


def _anndatas(hd: HiData, f: zarr.Group | None = None) -> Iterator[tuple[ad.AnnData, zarr.Group | None]]:
    """Yield every AnnData of ``hd``, and the modalities of MuData levels, with its group in the store ``f``."""
    for level, container in hd.levels.items():
        group = None
        if f is not None:
            key = _LEVEL_GROUPS[level]
            group = f[key] if key in f or level != "group" else f["D"]
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
    hd: HiData,
    f: zarr.Group | None = None,
    backend: Literal["dask", "zarr"] = "dask",
    n_threads: int | None = None,
) -> None:
    """Load sparse `X` and layers into memory and, from the Zarr store ``f``, back dense ones with cellink's reader.

    Missing calls of a 2-bit `G.X` become ``nan``, unless ``backend="zarr"``.
    """
    group_level = hd.levels["group"]
    for adata, group in _anndatas(hd, f):
        for name, value in _matrices(adata):
            if isinstance(value, ad.abc.CSRDataset | ad.abc.CSCDataset):
                _set_matrix(adata, name, value.to_memory())
            elif group is not None and isinstance(value, da.Array) and isinstance(group.get(name), zarr.Array):
                if adata.is_view:
                    raise ValueError(
                        "The levels of this store do not match, so HiData dropped or reordered units while "
                        "reading it. Read it with hidata.read_zarr_hd and write it again."
                    )
                stored = group[name]
                matrix = _lazy_matrix(stored, backend, n_threads)
                if adata is group_level and name == "X" and backend != "zarr":
                    matrix = _two_bit_missing_to_nan(stored, matrix)
                _set_matrix(adata, name, matrix)


def read_h5_hd(path: str, lazy: bool = False) -> HiData:
    """
    Read a HiData object from an HDF5 file on disk.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing serialized HiData, or written with donordata or cellink 0.1.0.
    lazy : bool, default=False
        If True, a dense `X` and dense layers of every level are Dask arrays that read
        their blocks from the file when they are computed, as with :func:`read_zarr_hd`,
        so a subset can be selected before any genotypes are loaded. Blocks follow the
        HDF5 chunks, or span at most 4096 rows and columns for unchunked datasets. The
        file must stay in place while these arrays are used. Sparse matrices and all
        annotations are read into memory.

    Returns
    -------
    HiData
        A HiData object with genotype (`G`), cell expression (`C`), and metadata.
    """
    hd = hidata.read_h5_hd(path, lazy=lazy)
    if lazy:
        _restore_matrices(hd)
    return hd


def read_zarr_hd(path: str, backend: Literal["dask", "zarr"] = "dask", n_threads: int | None = None) -> HiData:
    """
    Read a HiData object from a Zarr store on disk.

    A dense `X` stays lazily backed rather than being materialized on load, so a
    caller can select a subset (e.g. one cell type via ``hd.sel(C_obs=...)``) and
    only that subset is ever read off disk once it's actually computed/materialized.
    Sparse matrices and all annotations are read into memory.

    Parameters
    ----------
    path : str
        Path to the Zarr store containing serialized HiData, or written with donordata or cellink 0.1.0.
    backend : {"dask", "zarr"}
        Backing of a dense `X`/`layers` entry. ``"zarr"`` gives `G.X` the parallel
        genotype reader (:func:`cellink.io.read_pgen_zarr`), fastest for subsetting donors.
    n_threads : int, optional
        Threads for ``backend="zarr"`` genotype reads. Default: available cores, at most 16.

    Returns
    -------
    HiData
        A HiData object with genotype (`G`), cell expression (`C`), and metadata.
        Missing genotype calls of a 2-bit `G.X` (see :func:`cellink.io.write_zarr_hd`)
        are ``nan`` and `G.X` is float32; with ``backend="zarr"`` `G.X` holds the
        stored int8 codes, with -1 for missing calls.
    """
    hd = hidata.read_zarr_hd(path)
    _restore_matrices(hd, zarr.open(str(path), mode="r"), backend=backend, n_threads=n_threads)
    return hd


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


def read_hd(path: str, fmt: str = None, lazy: bool = False) -> HiData:
    """
    Read a HiData object from disk in either HDF5 or Zarr format.

    This function automatically detects the file format from the file extension
    unless explicitly specified.

    Parameters
    ----------
    path : str
        Path to the HiData file (`.h5`, `.hd.h5`, `.zarr`, or `.hd.zarr`).
    fmt : {'h5', 'zarr'}, optional
        File format to use. If None (default), inferred from the file extension.
    lazy : bool, default=False
        Passed to :func:`read_h5_hd` for HDF5 files. Zarr stores are always read lazily
        (:func:`read_zarr_hd`).

    Returns
    -------
    HiData

    Raises
    ------
    ValueError
        If the file extension cannot be mapped to a known format or if `fmt`
        is not one of {'h5', 'zarr'}.

    Examples
    --------
    >>> hd = read_hd("cohort.hd.h5")
    >>> hd = read_hd("cohort.hd.zarr")
    >>> hd = read_hd("custom_data.h5", fmt="h5")
    """
    fmt = _format(str(path), fmt)
    if fmt == "h5":
        return read_h5_hd(path, lazy=lazy)
    return read_zarr_hd(path)


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


def write_h5_hd(hd: HiData, path: str) -> None:
    """Write a HiData object to an HDF5 file.

    Parameters
    ----------
    hd : HiData
        The object to write.
    path : str
        Destination, conventionally ending in ``.hd.h5``.
    """
    hd.write_h5_hd(path)


def write_zarr_hd(
    hd: HiData,
    path: str,
    x_chunks: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
    genotype_encoding: Literal["2bit"] | None = "2bit",
) -> None:
    """Write a HiData object to a Zarr store, with the genotypes in `G.X` stored as 2-bit calls.

    Parameters
    ----------
    hd : HiData
        The object to write.
    path : str
        Destination, conventionally ending in ``.hd.zarr``.
    x_chunks : tuple[int, int] | dict[str, tuple[int, int]], optional
        Chunk shape for a dense `X` and dense layers, e.g. `(4096, 4096)`. Pass a
        dict (`{"G": ..., "C": ...}`) to set the levels independently. If not
        given: a dask-backed `X` keeps its own existing chunks (via
        `dask.array.to_zarr`); a plain array gets chunks capped at 4096
        per axis. Only applies to a dense `X`; sparse matrices are unaffected.
    genotype_encoding : {"2bit", None}
        Storage of a dense `G.X`. ``None`` writes it unencoded, with its
        values and dtype unchanged. ``"2bit"`` (default) stores it as 2-bit
        packed genotype calls, four per byte; a variant chunk size not
        divisible by 4 is rounded up to the next multiple of 4, with a
        warning. Only `G.X` is affected: `obs`, `var`, `obsm`,
        `layers`, `uns`, `S` and `C` are written unchanged. With ``"2bit"``:

        - Integer values 0, 1 and 2 are stored as they are. Negative
          integers count as missing.
        - Floating-point values, e.g. imputed dosages, are rounded to the
          nearest integer, halves to even (0.5 becomes 0, 1.5 becomes 2).
          NaN counts as missing.
        - Missing values stay missing: stored as the 2-bit code 3.
          :func:`cellink.io.read_zarr_hd` returns them as ``nan`` (and `G.X`
          as float32); with ``backend="zarr"`` it returns the stored codes,
          with -1 for missing.
        - A warning reports how many non-integer values were rounded and
          how many values are missing.
        - Every value must round to 0, 1 or 2: integers 0-2, or floats in
          [-0.5, 2.5). If any value lies outside, the whole `G.X` is written
          unencoded, with its original values and dtype, and a warning.
        - Continuous data inside [-0.5, 2.5) cannot be told apart from
          dosages and is rounded. Pass ``None`` to keep such data unchanged.

    Example
    -------
    >>> write_zarr_hd(hd, "path/to/cohort.hd.zarr")
    """
    if genotype_encoding not in ("2bit", None):
        raise ValueError(f"genotype_encoding must be '2bit' or None, got {genotype_encoding!r}")
    if genotype_encoding is None or hd.G_type != "anndata" or not _is_dense(hd.G.X):
        hd.write_zarr_hd(path, x_chunks=x_chunks)
        return
    # the genotypes are written once, as 2-bit calls, one column block at a time
    hd._check_not_reading_from(path)
    X = hd.G.X
    _without_genotypes(hd).write_zarr_hd(path, x_chunks=x_chunks)
    chunks = x_chunks.get("G") if isinstance(x_chunks, dict) else x_chunks
    _write_genotypes(f"{path}/G/X", X, chunks)


def _is_dense(X) -> bool:
    return hasattr(X, "ndim") and not sp.issparse(X) and not isinstance(X, ad.abc.CSRDataset | ad.abc.CSCDataset)


def _without_genotypes(hd: HiData) -> HiData:
    """A shallow copy of ``hd`` whose group level has no `X`, for writing everything but the genotypes."""
    G = hd.G
    shell = ad.AnnData(
        obs=G.obs,
        var=G.var,
        uns=dict(G.uns),
        obsm=dict(G.obsm),
        varm=dict(G.varm),
        obsp=dict(G.obsp),
        varp=dict(G.varp),
        layers={k: v for k, v in G.layers.items() if k is not None},
    )
    if G.raw is not None:
        shell.raw = G.raw.to_adata()
    new = copy.copy(hd)
    new.G = shell
    return new


def write_hd(hd: HiData, path: str, fmt: Literal["h5", "zarr"] | None = None) -> None:
    """Write a HiData object to disk, choosing the format from the extension unless `fmt` is given.

    Zarr stores are written with :func:`write_zarr_hd`, so `G.X` is stored as 2-bit genotype calls.

    Parameters
    ----------
    hd : HiData
        The object to write.
    path : str
        Destination ending in ``.h5`` or ``.zarr``. Without an extension, ``.hd.h5`` or
        ``.hd.zarr`` is appended according to `fmt`.
    fmt : {'h5', 'zarr'}, optional
        File format to use. If None (default), inferred from the file extension.
    """
    path = str(path)
    if _format(path, fmt) == "h5":
        write_h5_hd(hd, path if path.endswith(".h5") else path + ".hd.h5")
    else:
        write_zarr_hd(hd, path if path.endswith(".zarr") else path + ".hd.zarr")


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
            "G.X has values outside the 0-2 genotype range; it is written without 2-bit encoding.",
            UserWarning,
            stacklevel=3,
        )
        _write_dense(path, X, chunks)
        return
    arr = zarr.open_array(path, mode="r+")
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"
    arr.attrs["n_missing"] = n_missing  # read_zarr_hd turns them into nan


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
            f"G.X contains {n_frac:,} imputed (non-integer) values; the 2-bit genotype store keeps them rounded "
            "to the nearest integer. Pass genotype_encoding=None to keep the original values.",
            UserWarning,
            stacklevel=4,
        )
    if n_missing:
        warnings.warn(
            f"G.X contains {n_missing:,} missing values; the 2-bit genotype store keeps them missing, "
            "read back as nan (-1 with read_zarr_hd(backend='zarr')).",
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
