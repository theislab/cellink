from __future__ import annotations

import re

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

__all__ = ["fetch_slot"]

_INDEXED_EMB_RE = re.compile(r"^(?P<base>(?=.*[A-Za-z_])[A-Za-z0-9_]+)_(?P<idx>\d+)$")

try:
    from mudata import MuData
except ImportError:  # pragma: no cover
    MuData = type("MuDataStub", (), {})  # safe stub if mudata not installed


def _tag_axis(obj, axis, source=None):
    obj.attrs["axis"] = axis  # 'obs' or 'var'
    if source is not None:  # e.g. 'C' or 'G' (or 'A','G:rna')
        obj.attrs["source"] = source
    return obj


def _as_series(A, arr, axis, name):
    arr = ad.utils.asarray(arr).ravel()
    idx = A.obs_names if axis == "obs" else A.var_names
    return _tag_axis(pd.Series(arr, index=idx, name=name), axis)


def _as_dataframe(A, arr, axis, base):
    arr = ad.utils.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    cols = [f"{base}_{i + 1}" for i in range(arr.shape[1])]
    idx = A.obs_names if axis == "obs" else A.var_names
    return _tag_axis(pd.DataFrame(arr, index=idx, columns=cols), axis)


def _iter_anndatas(dd, adatas=("G", "C"), modalities=None, delim=":"):
    """Iterate over AnnData objects in a DonorData-like object.

    Yields ``(name, AnnData)`` across:
      - plain AnnData -> ``('A', dd)``
      - plain MuData -> ``(mod, dd.mod[mod])`` in requested order
      - DonorData-like with ``.G``/``.C``; if ``.G``/``.C`` is MuData, flatten
        as ``'G:mod'`` / ``'C:mod'``.

    ``modalities`` can be a sequence or a dict mapping outer name
    (``'G'``, ``'C'``, ``'A'``) -> sequence.
    """

    def _mods(mud, outer):
        if modalities is None:
            return list(mud.mod.keys())
        if isinstance(modalities, dict):
            return list(modalities.get(outer, mud.mod.keys()))
        return list(modalities)

    if isinstance(dd, pd.DataFrame):
        # Treat the DataFrame's columns as observations, allowing formulaic to find them.
        yield ("A", ad.AnnData(obs=dd))
        return

    if isinstance(dd, ad.AnnData):
        yield ("A", dd)
        return

    if isinstance(dd, MuData):
        for m in _mods(dd, "A"):
            if m in dd.mod:
                yield (m, dd.mod[m])
        return

    # DonorData-like
    for outer in adatas:
        A = getattr(dd, outer, None)
        if A is None:
            continue
        if isinstance(A, ad.AnnData):
            yield (outer, A)
        elif isinstance(A, MuData):
            for m in _mods(A, outer):
                if m in A.mod:
                    yield (f"{outer}{delim}{m}", A.mod[m])


def fetch_slot(dd, key, *, axis=None, adatas=("G", "C"), layer=None, raise_on_multiple=True, modalities=None):
    """Fetch a named slot from a DonorData-like, AnnData, MuData, or DataFrame object.

    If ``axis`` is None, both ``obs`` and ``var`` are searched; a key found in
    neither, or in more than one place (with ``raise_on_multiple=True``),
    raises. Returns a ``pd.Series`` (axis-aligned) or ``pd.DataFrame`` (when
    ``key`` names a whole ``obsm``/``varm`` entry). The result carries
    ``result.attrs['axis']`` in ``{'obs', 'var'}``.
    """

    def axis_maps(A, ax):
        idx = A.obs_names if ax == "obs" else A.var_names
        other = A.var_names if ax == "obs" else A.obs_names
        meta = A.obs if ax == "obs" else A.var
        emb = A.obsm if ax == "obs" else A.varm
        emb_name = "obsm" if ax == "obs" else "varm"
        return idx, other, meta, emb, emb_name

    def search_axis(A, ax):
        hits = []
        idx, other, meta, emb, emb_name = axis_maps(A, ax)

        # 0) full embedding by entry name
        if key in emb:
            v = emb[key]
            if isinstance(v, pd.DataFrame):
                hits.append((f".{emb_name}['{key}']", _tag_axis(v, ax)))
            else:
                hits.append((f".{emb_name}['{key}'](array)", _as_dataframe(A, v, ax, key)))

        # 1) obs/var column
        if key in meta:
            hits.append((f".{ax}['{key}']", _tag_axis(meta[key], ax)))

        # 2) embedding DataFrame columns (skip entry==key to avoid duplicate)
        for k, v in emb.items():
            if k == key:
                continue
            if isinstance(v, pd.DataFrame) and key in v.columns:
                hits.append((f".{emb_name}['{k}']['{key}']", _tag_axis(v[key], ax)))
        # 2b) ndarray embedding with suffix index: e.g. X_pca_5 -> emb['X_pca'][:, 5]
        m_idx = _INDEXED_EMB_RE.match(key)

        if m_idx:
            base = m_idx.group("base")
            j = int(m_idx.group("idx"))
            if base in emb:
                v = emb[base]
                if isinstance(v, np.ndarray | scipy.sparse.spmatrix) and v.ndim == 2 and 0 <= j < v.shape[1]:
                    hits.append(
                        (
                            f".{emb_name}['{base}'][:, {j}]",
                            _as_series(A, v[:, j], ax, key),
                        )
                    )

        # 3) X / layers vector aligned to this axis
        if key in other:
            if layer and layer in A.layers:
                arr = A[:, key].layers[layer] if ax == "obs" else A[key, :].layers[layer]
            else:
                arr = A[:, key].X if ax == "obs" else A[key, :].X
            hits.append((f"{f'.layers[{layer}]' if layer else '.X'} @ {ax}", _as_series(A, arr, ax, key)))

        return hits

    axes = ("obs", "var") if axis is None else (axis,)
    candidates = []
    for name, A in _iter_anndatas(dd, adatas=adatas, modalities=modalities):
        for ax in axes:
            for loc, obj in search_axis(A, ax):
                candidates.append((f"{name}{loc}", obj))

    if not candidates:
        src = (
            "modalities"
            if isinstance(dd, MuData) or any(isinstance(getattr(dd, x, None), MuData) for x in adatas)
            else "adatas"
        )
        raise KeyError(
            f"'{key}' not found ({src}={modalities if src == 'modalities' else adatas}, axis={axis or 'obs|var'})."
        )

    if len(candidates) > 1 and raise_on_multiple:
        where = ", ".join(w for w, _ in candidates)
        raise ValueError(
            f"Key '{key}' is not unique. Found in: {where}. Run with raise_on_multiple=False to return the first match."
        )

    return candidates[0][1]
