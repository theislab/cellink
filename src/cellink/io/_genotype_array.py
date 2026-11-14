from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from zarr.storage import LocalStore

_LUT = np.array([[(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3] for b in range(256)], dtype=np.int8)
_BLOSC = Blosc()


def _default_threads() -> int:
    try:
        return min(16, len(os.sched_getaffinity(0)))
    except AttributeError:
        return min(16, os.cpu_count() or 1)


def _as_index(idx, n: int) -> np.ndarray:
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n))
    idx = np.asarray(idx)
    if idx.dtype == bool:
        return np.flatnonzero(idx)
    idx = idx.astype(np.int64).ravel()
    return np.where(idx < 0, idx + n, idx)


def _groups(idx: np.ndarray, size: int) -> dict[int, np.ndarray]:
    order = np.argsort(idx // size, kind="stable")
    keys, starts = np.unique(idx[order] // size, return_index=True)
    return dict(zip(keys.tolist(), np.split(order, starts[1:]), strict=True))


def _slim(p: np.ndarray):
    if len(p) and p[-1] - p[0] == len(p) - 1 and (np.diff(p) == 1).all():
        return slice(int(p[0]), int(p[-1]) + 1)
    return p


@dataclass
class GenotypeArray(zarr.Array):
    """A `zarr.Array` with a parallel, selection-aware reader for genotype stores.

    For 2-D int8 genotype arrays on a local store whose codecs are blosc,
    optionally preceded by :class:`cellink.io._codecs.Pack2Bit`, indexing reads
    every touched chunk in parallel, keeps only the requested rows and unpacks
    only the requested 2-bit columns. Any other array or selection falls back to
    `zarr.Array` indexing, so values are always identical to a plain zarr read.
    """

    n_threads: int = 0

    @classmethod
    def wrap(cls, z: zarr.Array, n_threads: int | None = None) -> GenotypeArray:
        """Wrap an opened `zarr.Array` without copying data."""
        return cls(z._async_array, n_threads or _default_threads())

    def _layout(self) -> bool | None:
        if self.ndim != 2 or self.dtype != np.int8 or not isinstance(self.store, LocalStore):
            return None
        names = [c.to_dict()["name"] for c in self.metadata.codecs]
        if names == ["bytes", "blosc"]:
            return False
        if names == ["cellink_pack2bit", "bytes", "blosc"]:
            return True
        return None

    def _parse(self, sel):
        if not isinstance(sel, tuple):
            sel = (sel, slice(None))
        if len(sel) != 2 or any(s is Ellipsis or s is None for s in sel):
            return None
        r, c = sel
        if isinstance(r, np.ndarray) and isinstance(c, np.ndarray) and r.ndim == 2 and c.ndim == 2:
            if r.shape[1] != 1 or c.shape[0] != 1:
                return None
            r, c = r[:, 0], c[0]
        scalar = tuple(isinstance(s, int | np.integer) for s in (r, c))
        return _as_index(r, self.shape[0]), _as_index(c, self.shape[1]), scalar

    def _read(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        packed = self._layout()
        cr, cc = self.metadata.chunk_grid.chunk_shape
        root = Path(self.store.root) / self.path
        dt, w = (np.uint8, cc // 4) if packed else (np.int8, cc)
        out = np.empty((len(rows), len(cols)), dtype=np.int8)
        tasks, cg = [], _groups(cols, cc)
        for i, rp in _groups(rows, cr).items():
            rs, lr = _slim(rp), _slim(rows[rp] - i * cr)
            for j, cp in cg.items():
                cs, loc = _slim(cp), cols[cp] - j * cc
                dst = (rp[:, None], cp) if not isinstance(rs, slice) and not isinstance(cs, slice) else (rs, cs)
                tasks.append((root / self.metadata.encode_chunk_key((i, j)), lr, loc, _slim(loc), dst))

        def one(task):
            f, lr, loc, ls, dst = task
            try:
                buf = f.read_bytes()
            except FileNotFoundError:
                out[dst] = self.fill_value
                return
            a = np.frombuffer(_BLOSC.decode(buf), dtype=dt).reshape(cr, w)[lr]
            if not packed:
                out[dst] = a[:, ls]
            elif isinstance(ls, slice):
                # unpack covered bytes only
                b0, b1 = ls.start // 4, (ls.stop + 3) // 4
                out[dst] = _LUT[a[:, b0:b1]].reshape(len(a), -1)[:, ls.start - 4 * b0 : ls.stop - 4 * b0]
            else:
                out[dst] = _LUT[a[:, loc // 4], loc % 4]

        with ThreadPoolExecutor(self.n_threads or 1) as ex:
            list(ex.map(one, tasks))
        return out

    def __getitem__(self, sel):
        p = self._parse(sel) if self._layout() is not None else None
        if p is None:
            return super().__getitem__(sel)
        rows, cols, scalar = p
        out = self._read(rows, cols)
        if scalar[0] and scalar[1]:
            return out[0, 0]
        return out[0] if scalar[0] else out[:, 0] if scalar[1] else out

    def __array__(self, dtype=None, copy=None):
        a = self[:, :]
        return a if dtype is None else a.astype(dtype)
