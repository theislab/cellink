"""IO helpers for streaming cell-level data via the `annbatch` package.

This module exposes two thin wrappers around `annbatch` that make it easy to
materialize the cell-level AnnData inside a `DonorData` (i.e. ``dd.C``) as a
sharded zarr collection, and to open such a collection as a configured
``annbatch.Loader``.

The donor-side AnnData (``dd.G``) is intentionally not handled here -- its
``obs`` axis is small (donors) and streaming is the wrong tool.

The optional dependency stack is installable via:

    pip install cellink[annbatch]
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# The `annbatch` extra is optional. Importing this module without it should
# raise a clear, actionable error rather than a confusing ModuleNotFoundError
# from deep inside a dependency.
try:
    import anndata
    import zarr
    from annbatch import DatasetCollection, Loader
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Cannot import `cellink.io._annbatch`: this feature requires the "
        "`annbatch` extra. Install with:\n\n    pip install cellink[annbatch]"
    ) from e

from anndata import AnnData

from cellink._core import DonorData

if TYPE_CHECKING:
    from os import PathLike


__all__ = ["write_annbatch_collection", "open_annbatch_loader"]


def _configure_zarrs_codec_pipeline() -> None:
    # Recommended by annbatch docs for performance on zarr v3 stores. Setting
    # it inside our wrappers means callers don't have to remember.
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def _materialize_to_h5ad(adata: AnnData, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    h5_path = tmp_dir / f"{uuid.uuid4().hex}.h5ad"
    adata.write_h5ad(h5_path)
    return h5_path


def _resolve_source_paths(
    source: AnnData | DonorData | str | Path | Sequence[str | Path],
    tmp_dir: Path,
    *,
    layer: str | None,
) -> tuple[list[Path], bool]:
    """Resolve ``source`` to a list of on-disk h5ad/zarr paths.

    Returns ``(paths, materialized)`` where ``materialized`` is True when we
    wrote a temp h5ad that the caller is responsible for cleaning up.
    """
    if isinstance(source, AnnData):
        if layer is not None and layer not in source.layers:
            raise KeyError(f"Layer {layer!r} not present in source AnnData")
        return [_materialize_to_h5ad(source, tmp_dir)], True
    if isinstance(source, DonorData):
        adata = source.C
        if not isinstance(adata, AnnData):
            raise TypeError(
                "DonorData.C must be an AnnData for annbatch IO; "
                f"got {type(adata).__name__}"
            )
        if layer is not None and layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not present in DonorData.C.layers")
        return [_materialize_to_h5ad(adata, tmp_dir)], True
    if isinstance(source, str | Path):
        return [Path(source)], False
    # Assume an iterable of paths.
    paths = [Path(p) for p in source]
    if not paths:
        raise ValueError("`source` is an empty sequence of paths")
    return paths, False


def write_annbatch_collection(
    source: AnnData | DonorData | str | Path | Sequence[str | Path],
    path: str | PathLike[str],
    *,
    obs_keys: Sequence[str],
    layer: str | None = None,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    **add_adatas_kwargs: Any,
) -> Path:
    """Write a sharded zarr `annbatch` collection from cell-level data.

    Parameters
    ----------
    source
        Cell-level data to stream into the collection. May be:

        - an :class:`anndata.AnnData` (will first be written to a temp h5ad),
        - a :class:`cellink.DonorData` (uses ``source.C``; same temp-write
          fallback),
        - a path or list of paths to existing h5ad/zarr files.
    path
        Directory path for the output zarr collection.
    obs_keys
        ``obs`` columns the future loader will need to expose. Validated to be
        non-empty so callers don't accidentally drop the donor id column at
        read time. Filtering itself happens at read time in
        :func:`open_annbatch_loader` via the ``load_adata`` closure -- nothing
        is dropped at write time.
    layer
        Name of a layer to expose to the loader as ``X``. ``None`` means the
        loader will read ``X`` directly. When ``source`` is in-memory data,
        the layer is validated for presence; when ``source`` is a path, no
        validation is performed.
    shuffle
        Forwarded to :meth:`annbatch.DatasetCollection.add_adatas`. Default
        ``True`` produces a globally-shuffled collection, which is usually
        what you want for IID per-cell mini-batching.
    rng
        Random number generator used for shuffling at write time. Pass a
        seeded :class:`numpy.random.Generator` for reproducible collections.
    **add_adatas_kwargs
        Forwarded verbatim to :meth:`annbatch.DatasetCollection.add_adatas`
        (e.g. ``shard_size``, ``dataset_size``, ``shuffle_chunk_size``,
        ``var_subset``, ``zarr_compressor``).

    Returns
    -------
    Path
        Path to the created collection directory.
    """
    if not obs_keys:
        raise ValueError("`obs_keys` must be a non-empty sequence")

    _configure_zarrs_codec_pipeline()

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Place temp inputs alongside the output so they share a filesystem (cheap
    # rename / hard-linkable) and any stale residue is easy to spot.
    tmp_dir = out_path.parent / f".{out_path.name}.tmp_inputs"
    paths, materialized = _resolve_source_paths(source, tmp_dir, layer=layer)

    try:
        collection = DatasetCollection(str(out_path))
        collection.add_adatas(
            adata_paths=[str(p) for p in paths],
            shuffle=shuffle,
            rng=rng,
            **add_adatas_kwargs,
        )
    finally:
        if materialized and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


def _make_load_adata(
    obs_keys: Sequence[str],
    layer: str | None,
):
    """Build a ``load_adata`` closure for ``Loader.use_collection``.

    The closure restricts ``obs`` to ``obs_keys`` and selects ``X`` or
    ``layers[layer]`` as the feature matrix. Restricting ``obs`` is important
    for performance -- the default loader pulls every column.
    """
    obs_keys = list(obs_keys)

    def _load_adata(g: Any) -> AnnData:
        if layer is None:
            x_node = g["X"]
        else:
            x_node = g["layers"][layer]
        # Sparse vs dense is auto-detected: zarr.Array == dense; anything else
        # is a group encoding a sparse matrix and needs `sparse_dataset`.
        if isinstance(x_node, zarr.Array):
            X = x_node
        else:
            X = anndata.io.sparse_dataset(x_node)
        obs = anndata.io.read_elem(g["obs"])
        missing = [k for k in obs_keys if k not in obs.columns]
        if missing:
            raise KeyError(
                f"obs_keys {missing!r} not found in on-disk obs (have: "
                f"{list(obs.columns)})"
            )
        return AnnData(X=X, obs=obs[obs_keys])

    return _load_adata


def open_annbatch_loader(
    path: str | PathLike[str],
    *,
    obs_keys: Sequence[str],
    layer: str | None = None,
    batch_size: int = 4096,
    chunk_size: int = 32,
    preload_nchunks: int = 256,
    to_torch: bool = True,
    rng: np.random.Generator | None = None,
    **loader_kwargs: Any,
) -> Loader:
    """Open an annbatch collection at ``path`` and return a configured loader.

    The returned :class:`annbatch.Loader` is bound to the collection via a
    ``load_adata`` closure that:

    - exposes ``X`` (if ``layer is None``) or ``layers[layer]`` as the feature
      matrix, and
    - subsets ``obs`` to ``obs_keys``.

    Parameters
    ----------
    path
        Path to a collection produced by :func:`write_annbatch_collection`.
    obs_keys
        ``obs`` columns to surface in each yielded batch.
    layer
        Layer name to read; ``None`` => ``X``.
    batch_size, chunk_size, preload_nchunks, to_torch
        Forwarded to :class:`annbatch.Loader`. The defaults match the values
        suggested in the annbatch docs for cell-level IID training.
    rng
        Random number generator used for the loader's shuffling. Pass a
        seeded :class:`numpy.random.Generator` for reproducible iteration
        order. Mutually exclusive with a custom ``batch_sampler`` (which
        would carry its own rng).
    **loader_kwargs
        Forwarded verbatim to :class:`annbatch.Loader` (e.g. ``shuffle``,
        ``drop_last``, ``return_index``, ``preload_to_gpu``,
        ``concat_strategy``, ``batch_sampler``).

    Returns
    -------
    annbatch.Loader
        A loader ready to iterate. Callers can wrap it in a
        :class:`torch.utils.data.DataLoader` if multi-worker IO is desired.
    """
    if not obs_keys:
        raise ValueError("`obs_keys` must be a non-empty sequence")

    _configure_zarrs_codec_pipeline()

    coll_path = Path(path)
    if not coll_path.exists():
        raise FileNotFoundError(f"annbatch collection not found at {coll_path}")

    collection = DatasetCollection(str(coll_path), mode="r")

    loader = Loader(
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        to_torch=to_torch,
        rng=rng,
        **loader_kwargs,
    )
    return loader.use_collection(collection, load_adata=_make_load_adata(obs_keys, layer))
