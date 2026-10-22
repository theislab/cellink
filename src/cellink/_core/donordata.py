from __future__ import annotations

import copy as _copy
import logging
from collections.abc import Callable

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import zarr
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset
from anndata.io import write_elem
from mudata import MuData
from mudata._core.io import _write_h5mu
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cellink._core.data_fields import DAnn
from cellink._core.schema import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION

logger = logging.getLogger(__name__)

HIGHLIGHT_COLOR = "bold deep_pink2"

# Aggregations computable from per-block partial sums, so an out-of-core matrix is never loaded whole.
OUT_OF_CORE_AGGREGATIONS = ("sum", "mean", "count_nonzero", "var")
_TARGET_NNZ_PER_BLOCK = 50_000_000  # sparse block size (~0.5 GB of data + indices)
_TARGET_ELEMS_PER_BLOCK = 100_000_000  # dense block size


def _is_lazy_matrix(x) -> bool:
    """True for matrices held out of core: anndata backed sparse datasets and Dask arrays."""
    return isinstance(x, CSRDataset | CSCDataset | da.Array)


def _block_bounds_csr(X: CSRDataset, target_nnz: int) -> list[tuple[int, int]]:
    """Row ranges of a backed CSR matrix holding at most ~target_nnz stored values each."""
    indptr = np.asarray(X.group["indptr"][...])
    bounds, start = [], 0
    n_rows = len(indptr) - 1
    while start < n_rows:
        stop = int(np.searchsorted(indptr, indptr[start] + target_nnz, side="right")) - 1
        stop = min(max(stop, start + 1), n_rows)
        bounds.append((start, stop))
        start = stop
    return bounds


def _aggregate_out_of_core(X, groups: pd.Series, func: str, mask: np.ndarray | None = None, dof: int = 1):
    """Aggregate the rows of an out-of-core matrix by group, reading it block by block.

    Reproduces :func:`scanpy.get.aggregate` for ``sum``, ``mean``, ``count_nonzero`` and ``var``
    (``dof=1`` and scanpy's reset of variances lost to floating-point precision), while holding
    only one block of the matrix in memory at a time: row blocks for backed CSR matrices and Dask
    arrays, column blocks for backed CSC matrices. The accumulators are dense
    (n_groups x n_vars, float64). Groups without any (unmasked) row are dropped, as in scanpy.

    Returns
    -------
    pandas.DataFrame indexed by group, one column per variable.
    """
    if func not in OUT_OF_CORE_AGGREGATIONS:
        raise ValueError(f"out-of-core aggregation supports {OUT_OF_CORE_AGGREGATIONS}, not {func!r}")
    cat = pd.Categorical(groups)
    codes = cat.codes.astype(np.int64)
    if mask is not None:
        codes = np.where(np.asarray(mask, dtype=bool), codes, -1)
    n_groups, n_obs, n_vars = len(cat.categories), X.shape[0], X.shape[1]
    keep = codes >= 0
    counts = np.bincount(codes[keep], minlength=n_groups)

    need_sum = func in ("sum", "mean", "var")
    acc = np.zeros((n_groups, n_vars), dtype=np.float64) if need_sum else None
    acc_sq = np.zeros((n_groups, n_vars), dtype=np.float64) if func == "var" else None
    acc_nnz = np.zeros((n_groups, n_vars), dtype=np.int64) if func == "count_nonzero" else None

    def _indicator(block_codes: np.ndarray) -> sp.csr_matrix:
        rows = np.flatnonzero(block_codes >= 0)
        return sp.csr_matrix((np.ones(len(rows)), (block_codes[rows], rows)), shape=(n_groups, len(block_codes)))

    def _accumulate(ind, block, cols=slice(None)):
        block = block if sp.issparse(block) else np.asarray(block)
        if need_sum:
            acc[:, cols] += np.asarray((ind @ block).todense() if sp.issparse(block) else ind @ block)
        if acc_sq is not None:
            sq = block.multiply(block) if sp.issparse(block) else np.square(block, dtype=np.float64)
            acc_sq[:, cols] += np.asarray((ind @ sq).todense() if sp.issparse(sq) else ind @ sq)
        if acc_nnz is not None:
            nz = (block != 0).astype(np.int64) if sp.issparse(block) else (block != 0).astype(np.int64)
            acc_nnz[:, cols] += np.asarray((ind @ nz).todense() if sp.issparse(nz) else ind @ nz).astype(np.int64)

    if isinstance(X, CSCDataset):
        ind = _indicator(codes)
        step = max(1, _TARGET_ELEMS_PER_BLOCK // max(n_obs, 1))
        for j0 in range(0, n_vars, step):
            j1 = min(j0 + step, n_vars)
            _accumulate(ind, X[:, j0:j1], cols=slice(j0, j1))
    elif isinstance(X, CSRDataset) and keep.mean() < 0.5:
        # Sparse selection (e.g. one cell type): read only the selected rows, in batches holding at
        # most ~_TARGET_NNZ_PER_BLOCK stored values, instead of every block of the matrix.
        rows = np.flatnonzero(keep)
        indptr = np.asarray(X.group["indptr"][...])
        cum = np.cumsum(indptr[rows + 1] - indptr[rows])
        cuts = (
            np.searchsorted(cum, np.arange(_TARGET_NNZ_PER_BLOCK, cum[-1] + 1, _TARGET_NNZ_PER_BLOCK))
            if len(rows)
            else []
        )
        for batch in np.split(rows, np.unique(np.asarray(cuts, dtype=int) + 1)):
            if len(batch):
                _accumulate(_indicator(codes[batch]), X[batch])
    else:
        if isinstance(X, CSRDataset):
            bounds = _block_bounds_csr(X, _TARGET_NNZ_PER_BLOCK)
        else:
            step = max(1, _TARGET_ELEMS_PER_BLOCK // max(n_vars, 1))
            bounds = [(s, min(s + step, n_obs)) for s in range(0, n_obs, step)]
        for s, e in bounds:
            block_codes = codes[s:e]
            if not (block_codes >= 0).any():
                continue
            block = X[s:e]
            if isinstance(block, da.Array):
                block = block.compute()
            _accumulate(_indicator(block_codes), block)

    present = counts > 0
    if func == "sum":
        out = acc
    elif func == "count_nonzero":
        out = acc_nnz
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ = acc / counts[:, None]
            if func == "mean":
                out = mean_
            else:
                sq_mean = mean_**2
                var_ = acc_sq / counts[:, None] - sq_mean
                in_dtype = getattr(X, "dtype", np.float64)
                precision = 2 << (42 if np.dtype(in_dtype) == np.float64 else 20)
                var_[precision * var_ < sq_mean] = 0
                if dof != 0:
                    var_ *= (counts / (counts - dof))[:, None]
                out = var_
    return pd.DataFrame(out[present], index=pd.Index(cat.categories[present], name=groups.name))


def _write_dense_array_zarr_chunked(f: zarr.Group, path: str, X, chunks: tuple[int, int] | None) -> None:
    """Write one dense array (`X` or a dense layer) to `f[path]` with sane chunks."""
    if isinstance(X, da.Array):
        X = X.rechunk(chunks) if chunks is not None else X
        da.to_zarr(X, path, overwrite=True)
    else:
        array_chunks = chunks if chunks is not None else tuple(min(dim, 4096) for dim in X.shape)
        z = zarr.open_array(path, mode="w", shape=X.shape, chunks=array_chunks, dtype=X.dtype)
        z[:] = np.asarray(X)
    arr = zarr.open(path, mode="r+")
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"


def _write_anndata_zarr_dense_x_chunked(
    f: zarr.Group, key: str, adata: AnnData, zarr_path: str, chunks: tuple[int, int] | None = None
) -> None:
    """Write one AnnData to a Zarr group with `X` and any dense `layers` chunked sanely.

    `anndata.io.write_elem` picks its own chunk shape for a dense array
    without regard to any chunking the input already has, including a dask
    array's own `.chunks`. For a genome-scale `X` (or a same-shaped dense
    layer, e.g. raw counts kept alongside a normalized `X`) that shape can be
    badly misaligned with how the data is laid out on disk, making both the
    write and every later read far slower than necessary. Writes everything except `X`
    and any dense layers first (via a cheap, correctly-shaped zero-nnz
    sparse placeholder for `X`, and by omitting dense layers entirely from
    this first pass), then writes each dense array separately with an
    explicit, sane `chunks`.

    Parameters
    ----------
    chunks
        Chunk shape applied to `X` and every dense layer. If `None`: a dask
        array's own chunks (which `dask.array.to_zarr` preserves
        automatically), or, for a plain array, capped at 4096 per axis.
    """
    X = adata.X
    dense_layers = {k: v for k, v in adata.layers.items() if hasattr(v, "ndim") and not sp.issparse(v)}
    sparse_layers = {k: v for k, v in adata.layers.items() if k not in dense_layers}

    placeholder = sp.csr_matrix((adata.n_obs, adata.n_vars), dtype=X.dtype)
    shell = AnnData(
        X=placeholder,
        obs=adata.obs,
        var=adata.var,
        obsm=dict(adata.obsm),
        varm=dict(adata.varm),
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
        layers=dict(sparse_layers),
        uns=dict(adata.uns),
    )
    write_elem(f, key, shell)
    del f[key]["X"]
    _write_dense_array_zarr_chunked(f, f"{zarr_path}/{key}/X", X, chunks)

    if dense_layers and "layers" not in f[key]:
        f[key].create_group("layers")
    for name, layer in dense_layers.items():
        _write_dense_array_zarr_chunked(f, f"{zarr_path}/{key}/layers/{name}", layer, chunks)


class DonorData:
    """Store and manage donor-related data with single-cell readouts.

    This class allows donor-level, especially genetic, analysis with single-cell datasets.
    It holds AnnData objects for donor-level/genetic (G) and single-cell (C) data

    Attributes
    ----------
        G (AnnData): Donor-level AnnData
        C (AnnData): Cell-level AnnData
        donor_id (str): Key for donor information in C.obs

    Returns
    -------
        DonorData object
    """

    def __init__(
        self,
        *,
        G: AnnData | MuData,
        C: AnnData | MuData,
        donor_id: str = DAnn.donor,
        var_dims_to_sync: list[str] = None,
        uns: dict | None = None,
    ):
        if uns is None:
            uns = {}
        if donor_id not in C.obs.columns:
            raise ValueError(f"'{donor_id}' not found in C.obs")
        if isinstance(G, MuData) and G.obs.index.name is None:
            for mod in G.mod.values():
                if mod.obs.index.name == donor_id:
                    G.obs.index.name = donor_id
                    break
        if donor_id not in G.obs.columns and donor_id != G.obs.index.name:
            raise ValueError(f"'{donor_id}' must be in gdata.obs or set as index")
        if donor_id != G.obs.index.name:
            G.obs = G.obs.set_index(donor_id, drop=False)

        self._var_dims_to_sync = [] if var_dims_to_sync is None else var_dims_to_sync
        self.donor_id = donor_id
        self._match_donors(G, C)
        self.uns = uns

    def _match_donors(self, G: AnnData | MuData, C: AnnData | MuData) -> None:
        G_idx = G.obs.index
        C_idx = pd.Index(C.obs[self.donor_id].unique())
        keep_donors = G_idx.intersection(C_idx)

        if len(keep_donors) == 0:
            raise ValueError(
                (
                    "No overlapping donors between dd.G and dd.C,"
                    " make sure dd.C.obs['%s'] exists and corresponds to dd.G.obs_names"
                ),
                self.donor_id,
            )
        if not keep_donors.equals(G.obs_names):
            G = G[keep_donors]

        keep_cells = C.obs[self.donor_id].isin(keep_donors)
        if not keep_cells.all():
            C = C[keep_cells]

        # Sort cells by donor order. Stable, so cells already grouped by donor (every store written by
        # write_zarr_dd/write_h5_dd) need no reordering, and an out-of-core C is not turned into a
        # reordering view that would read every cell on first access to C.X.
        sorted_cells = C.obs.iloc[
            pd.Categorical(C.obs[self.donor_id], categories=keep_donors, ordered=True).argsort(kind="stable")
        ].index
        if not sorted_cells.equals(C.obs_names):
            C = C[sorted_cells]

        self._C = C
        self._G = G

    def copy(self) -> DonorData:
        new = DonorData.__new__(DonorData)
        new._var_dims_to_sync = list(self._var_dims_to_sync)
        new.donor_id = self.donor_id
        new._G = self._G.copy()
        new._C = self._C.copy()
        new.uns = _copy.deepcopy(self.uns)
        return new

    def _write_dd(self, f: h5py.File, zarr_path: str | None = None, x_chunks=None):
        is_zarr = isinstance(f, zarr.Group)
        for key, modality in (("G", self.G), ("C", self.C)):
            if isinstance(modality, MuData):
                group = f.create_group(key)
                _write_h5mu(group, modality)
            elif (
                is_zarr
                and hasattr(modality.X, "ndim")
                and not sp.issparse(modality.X)
                and not isinstance(modality.X, CSRDataset | CSCDataset)
                and not (isinstance(modality.X, da.Array) and sp.issparse(modality.X._meta))
            ):
                key_chunks = x_chunks.get(key) if isinstance(x_chunks, dict) else x_chunks
                _write_anndata_zarr_dense_x_chunked(f, key, modality, zarr_path, chunks=key_chunks)
            else:
                write_elem(f, key, modality)
        f.attrs["encoding-type"] = DONORDATA_ENCODING_TYPE
        f.attrs["encoding-version"] = DONORDATA_ENCODING_VERSION

        f.attrs["donor_id"] = self.donor_id
        f.attrs["var_dims_to_sync"] = self._var_dims_to_sync

        for key, value in self.uns.items():
            f.create_dataset(f"uns/{key}", data=value)

    def write_h5_dd(self, path: str) -> None:
        """Write the DonorData object to the specified file path.

        Parameters
        ----------
        path : str | Path
            Path where the donor-data object should be saved.

        Example
        -------
        write_dd('path/to/donor_data.dd.h5')
        """
        with h5py.File(path, "w") as f:
            self._write_dd(f)

    def write_zarr_dd(
        self, path: str, x_chunks: tuple[int, int] | dict[str, tuple[int, int]] | None = None
    ) -> None:
        """Write the DonorData object to the specified file paths for both gene expression data (G) and cell-type data (C).

        Parameters
        ----------
        path : str | Path
            Path where the donor-data object should be saved.
        x_chunks : tuple[int, int] | dict[str, tuple[int, int]], optional
            Chunk shape for a dense `G.X`/`C.X`, e.g. `(4096, 4096)`. Pass a
            dict (`{"G": ..., "C": ...}`) to set them independently. If not
            given: a dask-backed `X` keeps its own existing chunks (via
            `dask.array.to_zarr`); a plain array gets chunks capped at 4096
            per axis. Only applies to a dense `X`; sparse/`MuData` are
            unaffected.

        Example
        -------
        write_dd('path/to/donor_data.dd.zarr')
        """
        for m in [self.G, self.C]:
            if isinstance(m, MuData):
                raise NotImplementedError("MuData not supported for zarr write")
        f = zarr.open(path, mode="w")
        self._write_dd(f, zarr_path=str(path), x_chunks=x_chunks)

    def _ensure_extension(self, path: str, ext: str) -> str:
        """Ensure the given path ends with the desired extension."""
        if not path.endswith(ext):
            path += ext
        return path

    def write_dd(self, path: str, fmt: str = None) -> None:
        """Write the DonorData object to the specified file path, format detected from the extension.

        Parameters
        ----------
        path : str | Path
            Path where the donor-data object should be saved.

        Example
        -------
        write_dd('path/to/donor_data.dd.h5')
        """
        if fmt is None:
            if path.endswith(".h5") or path.endswith(".dd.h5"):
                fmt = "h5"
            elif path.endswith(".zarr") or path.endswith(".dd.zarr"):
                fmt = "zarr"
            else:
                raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")

        if fmt == "h5":
            path = self._ensure_extension(path, ".dd.h5")
            self.write_h5_dd(path)
        elif fmt == "zarr":
            path = self._ensure_extension(path, ".dd.zarr")
            self.write_zarr_dd(path)
        else:
            raise ValueError("Unknown format: use 'h5' or 'zarr'.")

    @property
    def C(self) -> AnnData:
        return self._C

    @C.setter
    def C(self, value: AnnData | MuData) -> None:
        if not isinstance(value, AnnData | MuData):
            raise ValueError("C must be an AnnData or MuData object")
        self._C = value
        self._match_donors(self._G, self._C)

    @property
    def G(self) -> AnnData:
        return self._G

    @G.setter
    def G(self, value: AnnData | MuData) -> None:
        if not isinstance(value, AnnData | MuData):
            raise ValueError("G must be an AnnData or MuData object")
        self._G = value
        self._match_donors(self._G, self._C)

    def _sync_var_dims(self, G: AnnData | MuData, C: AnnData | MuData) -> None:
        """Sync _D.obsm if needed, drop deleted keys from _var_dims_to_sync"""
        keys_to_sync = []
        for v in self._var_dims_to_sync:
            if v in G.obsm:
                keys_to_sync.append(v)
                if not C.var.index.equals(G.obsm[v].columns):
                    if G.is_view:
                        G = G.copy()
                    G.obsm[v] = G.obsm[v].loc[:, C.var.index]
        self._var_dims_to_sync = keys_to_sync
        return G

    def sel(
        self,
        *,
        G_obs: slice = slice(None),
        G_var: slice = slice(None),
        C_obs: slice = slice(None),
        C_var: slice = slice(None),
    ):
        _G = self.G[G_obs, G_var]
        _C = self.C[C_obs, C_var]

        _G = self._sync_var_dims(_G, _C)
        return DonorData(G=_G, C=_C, donor_id=self.donor_id, var_dims_to_sync=self._var_dims_to_sync)

    # TODO: properly support ellipsis
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 4:
            raise IndexError("DonorData only supports 4 dimensions")

        key = self._expand_ellipsis(key)
        key = tuple(
            (idx,) if isinstance(idx, str) else idx for idx in key
        )  # needed because Mudata[str] looks up modalities
        key = key + (slice(None),) * (4 - len(key))
        # Only slice if key is not slice(None)
        _G = self.G[key[0]] if key[0] is not slice(None) else self.G
        _G = _G[:, key[1]] if key[1] is not slice(None) else _G
        _C = self.C[key[2]] if key[2] is not slice(None) else self.C
        _C = _C[:, key[3]] if key[3] is not slice(None) else _C

        _G = self._sync_var_dims(_G, _C)
        return DonorData(
            G=_G,
            C=_C,
            donor_id=self.donor_id,
            var_dims_to_sync=self._var_dims_to_sync,
        )

    def _expand_ellipsis(self, index_tuple):
        ellipsis_count = sum(1 for idx in index_tuple if idx is Ellipsis)
        if ellipsis_count == 0:
            return index_tuple
        if ellipsis_count > 1:
            raise IndexError("DonorData only supports a single ellipsis ('...').")

        ellipsis_index = index_tuple.index(Ellipsis)
        provided = len(index_tuple) - 1  # minus the ellipsis
        num_full_slices = 4 - provided  # dd has 4 dimensions

        new_index = index_tuple[:ellipsis_index] + (slice(None),) * num_full_slices + index_tuple[ellipsis_index + 1 :]
        return new_index

    def aggregate(
        self,
        *,
        key_added: None | str = None,
        layer: None | str = None,
        obs: None | str = None,
        obsm: None | str = None,
        filter_key: None | str = None,
        filter_value: None | str = None,
        add_to_obs: bool = False,
        func: str | Callable = "mean",
        sync_var: bool = False,
        verbose: bool = False,
    ) -> None:
        """Aggregate single-cell data to donor-level.

        If neither layer, obsm, or obs is provided, adata.X is aggregated.
        Args:
            key_added:
                The key in gdata to store the aggregated data.
            layer:
                The layer in adata to aggregate. Defaults to None.
            obs:
                The key in adata.obs to aggregate. Defaults to None.
            obsm:
                The key in adata.obsm to aggregate. Defaults to None.
            filter_key:
                The key in adata.obs to filter by. Defaults to None.
            filter_value:
                The value in adata.obs[filter_key] to filter by. Defaults to None.
            add_to_obs:
                Whether to add the aggregated data to adata.obs. Defaults to False.
            func:
                The aggregation function to use. Defaults to "mean".
            sync_var:
                Whether to set the variable dimensions of the aggregated data to sync with
                the variable dimensions of the single-cell data. Defaults to False.
            verbose:
                Whether to print verbose output. Defaults to False.
        """
        if filter_key is not None:
            assert filter_value is not None, "filter_value must be provided if filter_key is provided"

        assert (layer is None) + (obsm is None) + (
            obs is None
        ) >= 2, "Only one of layer, obsm, varm, or obs can be provided"

        slot = layer or obsm or obs or "X"
        if key_added is None:
            key_added = f"{slot}_{func}"
            if filter_value is not None:
                key_added += f"_{filter_value}"

        # Locate the matrix without materializing it. For a view (e.g. after a donor drop), accessing
        # .X would read the whole view, so the view's parent is used with a row mask instead; anndata
        # has no public accessor for a view's parent, hence _adata_ref.
        matrix, base, rows_mask, var_idx = None, self.C, None, None
        if obs is None and obsm is None and isinstance(self.C, AnnData):
            ref = getattr(self.C, "_adata_ref", None) if self.C.is_view else None
            if ref is not None and _is_lazy_matrix(ref.layers[layer] if layer is not None else ref.X):
                base = ref
                rows_mask = np.zeros(ref.n_obs, dtype=bool)
                rows_mask[ref.obs_names.get_indexer(self.C.obs_names)] = True
                var_idx = ref.var_names.get_indexer(self.C.var_names)
            if not self.C.is_view or base is not self.C:
                matrix = base.layers[layer] if layer is not None else base.X
        if matrix is not None and _is_lazy_matrix(matrix) and func in OUT_OF_CORE_AGGREGATIONS:
            # Out-of-core C (e.g. read_zarr_dd): stream the matrix block by block; the filter is
            # applied inside each block, so neither the full matrix nor the filtered cells are loaded.
            mask = rows_mask
            if filter_key is not None:
                fmask = (base.obs[filter_key] == filter_value).to_numpy()
                mask = fmask if mask is None else (mask & fmask)
            aggres = _aggregate_out_of_core(matrix, base.obs[self.donor_id], func, mask=mask)
            if var_idx is not None:
                aggres = aggres.iloc[:, var_idx]
            columns = self.C.var_names
            aggres.columns = columns
            if sync_var:
                self._var_dims_to_sync.append(key_added)
            target = "obsm"
            adata = None
        else:
            adata = self.C
            if filter_key is not None:
                adata = adata[adata.obs[filter_key] == filter_value]
            if matrix is not None and _is_lazy_matrix(matrix):
                logger.warning(
                    "aggregate(func=%r) has no out-of-core implementation (supported: %s); loading the "
                    "%s selected cells into memory.",
                    func,
                    ", ".join(OUT_OF_CORE_AGGREGATIONS),
                    f"{adata.n_obs:,}",
                )
                adata = adata.to_memory() if not adata.is_view else adata.copy()

        if adata is None:
            pass
        elif obs is not None:
            columns = obs if isinstance(obs, list) else [obs]
            aggres = adata.obs.groupby(self.donor_id, observed=True)[columns].agg(func)
            target = "obs" if add_to_obs else "obsm"
        else:
            aggdata = sc.get.aggregate(adata, by=self.donor_id, func=func, layer=layer, obsm=obsm)
            if slot == "obsm":
                # use columns of obsm dataframe if provided
                columns = getattr(getattr(adata, slot), "columns", None)
            else:
                columns = adata.var_names
                if sync_var:
                    self._var_dims_to_sync.append(key_added)
            # Convert the aggregated layer to a DataFrame.
            aggres = pd.DataFrame(aggdata.layers[func], index=aggdata.obs_names)
            target = "obsm"

        if verbose:
            logger.info(f"Aggregated {slot} to {key_added}")
            logger.info("Observation found for %s donors.", aggres.shape[0])
        if target == "obsm" and columns is not None and len(columns) == aggres.shape[1]:
            # The frame the fill below produces (rows matched by donor, columns by position), without its
            # object-dtype round trip: for 4,381 donors x 36,601 genes that took ~40 s and ~9x the result's
            # memory. An integer aggregate with donors missing becomes float with NaN instead of raising.
            data = aggres.set_axis(pd.Index(columns), axis=1).reindex(self.G.obs_names)
        else:
            data = pd.DataFrame(index=self.G.obs_names, columns=columns)
            data.loc[aggres.index] = aggres
            aggres.columns = data.columns
            data = data.astype(aggres.dtypes.to_dict())

        if self.G.is_view:  # because we will write to self.G
            self.G = self.G.copy()

        if target == "obs":
            for col in columns:
                if data[col].dtype == "category":
                    self.G.obs[col] = pd.Categorical(data[col], categories=data[col].cat.categories)
                self.G.obs.loc[data.index, col] = data[col]
        else:
            self.G.obsm[key_added] = data

    def prep_repr(self) -> str:
        """String representation of DonorData showing side-by-side dd.G and dd.C views."""
        # Split the representations into lines
        G_lines, G_highlight = _anndata_repr(self.G, self.G.n_obs, self.G.n_vars, self._var_dims_to_sync)
        C_lines, C_highlight = _anndata_repr(self.C, self.C.n_obs, self.C.n_vars, [self.donor_id])

        def pad_lists(l1, l2):
            max_lines = max(len(l1), len(l2))
            l1 += [""] * (max_lines - len(l1))
            l2 += [""] * (max_lines - len(l2))
            return l1, l2

        # Ensure both have the same number of lines by padding with empty lines if necessary
        G_lines, C_lines = pad_lists(G_lines, C_lines)
        G_highlight, C_highlight = pad_lists(G_highlight, C_highlight)

        def highlight_lines(lines, highlights):
            _lines = []
            for line, highlight in zip(lines, highlights, strict=True):
                text = Text()
                for idx, (el, hl) in enumerate(zip(line, highlight, strict=True)):
                    _el = f"{el}, " if idx not in [0, len(line) - 1] else el
                    text.append(_el, HIGHLIGHT_COLOR if hl else None)
                _lines.append(text)
            return _lines

        G_lines = highlight_lines(G_lines, G_highlight)
        C_lines = highlight_lines(C_lines, C_highlight)

        # Create table
        table = Table(show_header=True, header_style=HIGHLIGHT_COLOR)
        table.add_column("G (donors)", max_width=50, justify="left")
        table.add_column("C (cells)", max_width=50, justify="left")
        for gdata_line, adata_line in zip(G_lines, C_lines, strict=True):
            table.add_row(gdata_line, adata_line)

        # Create title and surrounding panel
        n_cells_per_donor = self.C.obs[self.donor_id].value_counts()
        min_n_cells, max_n_cells = n_cells_per_donor.min(), n_cells_per_donor.max()
        header_line = Text(
            f"DonorData(n_donors={self.G.shape[0]:,}, "
            f"n_cells_per_donor=[{min_n_cells:,}-{max_n_cells:,}], "
            f"donor_id='{self.donor_id}')",
            style=HIGHLIGHT_COLOR,
        )
        panel = Panel(
            table,
            title=header_line,
            title_align="left",  # left, center, right
            box=DOUBLE,  # DOUBLE, HEAVY, MINIMAL, etc.
            expand=False,
        )
        return panel

    def __repr__(self) -> str:
        table = self.prep_repr()
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()

    def __str__(self) -> str:
        n_donors, n_donor_vars, n_cells, n_cell_vars = self.shape
        return (
            f"{self.__class__.__name__}"
            f"(n_G_obs={n_donors:,}, n_G_vars={n_donor_vars:,}, "
            f"n_C_obs={n_cells:,}, n_C_vars={n_cell_vars:,})"
        )

    @property
    def shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return *self.G.shape, *self.C.shape


def _anndata_repr(adata, n_obs, n_vars, highlight_keys=None) -> str:
    if adata.isbacked:
        backed_at = f" backed at {str(adata.filename)!r}"
    else:
        backed_at = ""
    lines = [[f"AnnData object with n_obs × n_vars = {n_obs:,} × {n_vars:,} {backed_at}"]]
    if adata.is_view:
        lines[0][0] = "View of " + lines[0][0]
    highlight = [[False]]
    for attr in ["obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp"]:
        keys = getattr(adata, attr).keys()

        if len(keys) > 0:
            line = [f"    {attr}: "]
            line_highlight = [False]
            for key in keys:
                if attr in ["obsm", "obs"] and key in highlight_keys:
                    line_highlight.append(True)
                else:
                    line_highlight.append(False)
                line.append(f"'{key}'")
            lines.append(line)
            highlight.append(line_highlight)
    return lines, highlight


if __name__ == "__main__":
    from cellink._core.dummy_data import sim_adata, sim_gdata

    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    dd = DonorData(G=gdata, C=adata)
    print(dd)
