"""Tests for lazy DonorData support (dask-backed G)."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink._core.data_fields import DAnn
from cellink._core.donordata import DonorData, _has_dask_X
from cellink.io import read_lazy_dd, read_zarr_dd

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_lazy_gdata(gdata: AnnData) -> AnnData:
    """Convert an in-memory AnnData to one with a dask-backed X."""
    lazy_X = da.from_array(np.asarray(gdata.X), chunks=(gdata.n_obs, 2))
    return AnnData(X=lazy_X, obs=gdata.obs.copy(), var=gdata.var.copy())


def _make_lazy_adata(adata: AnnData) -> AnnData:
    """Convert an in-memory single-cell AnnData to one with a dask-backed X.

    obs is preserved as a pandas DataFrame (so .isin / Categorical-argsort work);
    only X becomes a dask.Array. _has_dask_X(adata) returns True.
    """
    lazy_X = da.from_array(np.asarray(adata.X), chunks=(adata.n_obs, 2))
    return AnnData(X=lazy_X, obs=adata.obs.copy(), var=adata.var.copy())


# ── _has_dask_X ─────────────────────────────────────────────────────────────


def test_has_dask_X_false(gdata):
    assert _has_dask_X(gdata) is False


def test_has_dask_X_true(gdata):
    lazy = _make_lazy_gdata(gdata)
    assert _has_dask_X(lazy) is True


# ── DonorData with lazy G ────────────────────────────────────────────────────


def test_lazy_init(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    assert dd.is_lazy is True


def test_eager_init(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    assert dd.is_lazy is False


def test_lazy_shape(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    assert dd.shape == (*lazy_g.shape, *adata.shape)


def test_lazy_donor_matching(adata, gdata):
    """Lazy G with extra donors: filter is deferred (avoids fancy row-indexing on
    dask before var-slicing) and applied at to_memory()."""
    extra_obs = pd.DataFrame(index=pd.Index(["EXTRA_DONOR"], name=DAnn.donor))
    extra_X = da.zeros((1, gdata.n_vars), chunks=(1, gdata.n_vars))
    extra = AnnData(
        X=da.concatenate([da.from_array(np.asarray(gdata.X)), extra_X], axis=0),
        obs=pd.concat([gdata.obs, extra_obs]),
        var=gdata.var.copy(),
    )
    dd = DonorData(G=extra, C=adata)
    # Filter deferred — _G still holds EXTRA_DONOR
    assert "EXTRA_DONOR" in dd.G.obs_names
    assert dd._lazy_G_obs_filter is not None and "EXTRA_DONOR" not in dd._lazy_G_obs_filter
    # After to_memory(), the deferred filter is applied
    dd_mem = dd.to_memory()
    assert "EXTRA_DONOR" not in dd_mem.G.obs_names


def test_lazy_slice_variants(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    keep = ["SNP0", "SNP1"]
    dd_sub = dd[:, keep]
    assert dd_sub.shape[1] == len(keep)
    # Slicing lazy should remain lazy
    assert dd_sub.is_lazy is True


# ── to_memory ────────────────────────────────────────────────────────────────


def test_to_memory(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    dd_mem = dd.to_memory()
    assert dd_mem.is_lazy is False
    assert isinstance(dd_mem.G.X, np.ndarray)
    np.testing.assert_array_equal(dd_mem.G.X, np.asarray(gdata.X))


def test_to_memory_idempotent(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    dd_mem = dd.to_memory()
    assert dd_mem is dd  # no-op for in-memory


# ── aggregate guard ──────────────────────────────────────────────────────────


def test_aggregate_works_with_lazy_G_only(adata, gdata):
    """With per-side laziness, aggregate operates on C — lazy G alone should not block it."""
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    dd.aggregate(key_added="Gex")
    assert "Gex" in dd.G.obsm


def test_aggregate_raises_on_lazy_C(adata, gdata):
    """Aggregation reads C — lazy C must still block until to_memory()."""
    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=gdata, C=lazy_c)
    with pytest.raises(RuntimeError, match="lazy"):
        dd.aggregate(key_added="Gex")


def test_aggregate_works_after_to_memory(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    dd_mem = dd.to_memory()
    dd_mem.aggregate(key_added="Gex")
    assert "Gex" in dd_mem.G.obsm


# ── copy ─────────────────────────────────────────────────────────────────────


def test_copy_lazy(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    dd_copy = dd.copy()
    # copy should not materialise
    assert dd_copy.is_lazy is True
    assert isinstance(dd_copy.G.X, da.Array)


# ── read_lazy_dd roundtrip ───────────────────────────────────────────────────


@pytest.mark.slow
def test_read_lazy_dd_roundtrip(tmp_path, adata, gdata):
    """Write a DonorData as zarr, read it back lazily, verify shapes."""
    out = tmp_path / "test.dd.zarr"
    dd = DonorData(G=gdata, C=adata)
    dd.write_zarr_dd(str(out))

    dd_lazy = read_lazy_dd(str(out))
    assert dd_lazy.is_lazy is True
    assert dd_lazy.G.shape == dd.G.shape
    assert dd_lazy.C.shape == dd.C.shape

    # Materialise and check values
    dd_mem = dd_lazy.to_memory()
    assert dd_mem.is_lazy is False
    np.testing.assert_array_almost_equal(dd_mem.G.X, np.asarray(dd.G.X))


@pytest.mark.slow
def test_read_lazy_dd_matches_eager(tmp_path, adata, gdata):
    """Lazy and eager reads of the same zarr should produce identical data."""
    out = tmp_path / "test.dd.zarr"
    dd = DonorData(G=gdata, C=adata)
    dd.write_zarr_dd(str(out))

    dd_eager = read_zarr_dd(str(out))
    dd_lazy = read_lazy_dd(str(out))
    dd_lazy_mem = dd_lazy.to_memory()

    assert dd_lazy_mem.G.shape == dd_eager.G.shape
    assert dd_lazy_mem.C.shape == dd_eager.C.shape
    np.testing.assert_array_almost_equal(dd_lazy_mem.G.X, np.asarray(dd_eager.G.X))


# ── per-side laziness flags ──────────────────────────────────────────────────


def test_lazy_G_only(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    assert dd._lazy_G is True
    assert dd._lazy_C is False
    assert dd.is_lazy is True


def test_lazy_C_only(adata, gdata):
    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=gdata, C=lazy_c)
    assert dd._lazy_G is False
    assert dd._lazy_C is True
    assert dd.is_lazy is True


def test_lazy_both(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=lazy_g, C=lazy_c)
    assert dd._lazy_G is True
    assert dd._lazy_C is True
    assert dd.is_lazy is True


def test_eager_both(adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    assert dd._lazy_G is False
    assert dd._lazy_C is False
    assert dd.is_lazy is False


def test_lazy_flags_track_setter(adata, gdata):
    """_lazy_G / _lazy_C are computed from current G/C, not cached at init."""
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    assert dd._lazy_G is True
    # Replace G with eager — flag must flip
    dd.G = gdata
    assert dd._lazy_G is False
    assert dd.is_lazy is False


# ── to_memory: lazy C side ───────────────────────────────────────────────────


def test_to_memory_lazy_C(adata, gdata):
    # Eager reference: cells get sorted by donor order during _match_donors,
    # so we compare against an eager DonorData's C.X (same sort order).
    eager_dd = DonorData(G=gdata, C=adata.copy())
    expected = np.asarray(eager_dd.C.X)

    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=gdata, C=lazy_c)
    dd_mem = dd.to_memory()
    assert dd_mem.is_lazy is False
    assert isinstance(dd_mem.C.X, np.ndarray)
    np.testing.assert_array_equal(dd_mem.C.X, expected)


def test_to_memory_lazy_both(adata, gdata):
    eager_dd = DonorData(G=gdata, C=adata.copy())
    expected_C = np.asarray(eager_dd.C.X)

    lazy_g = _make_lazy_gdata(gdata)
    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=lazy_g, C=lazy_c)
    dd_mem = dd.to_memory()
    assert dd_mem.is_lazy is False
    assert isinstance(dd_mem.G.X, np.ndarray)
    assert isinstance(dd_mem.C.X, np.ndarray)
    np.testing.assert_array_equal(dd_mem.G.X, np.asarray(gdata.X))
    np.testing.assert_array_equal(dd_mem.C.X, expected_C)


def test_to_memory_lazy_C_preserves_donor_intersection(adata, gdata):
    """When C has cells from a donor not in G, lazy-C donors filter is deferred
    and applied at to_memory."""
    # Drop one donor from G to force a mismatch
    g_subset = gdata[gdata.obs_names[:-1]].copy()
    lazy_c = _make_lazy_adata(adata)
    dd = DonorData(G=g_subset, C=lazy_c)
    # Filter is deferred for lazy C
    assert dd._lazy_C_obs_filter is not None
    dd_mem = dd.to_memory()
    # After materialisation, no cells from the dropped donor remain
    dropped = set(adata.obs[DAnn.donor]) - set(g_subset.obs_names)
    assert dropped, "test setup should produce a non-trivial drop"
    assert not set(dd_mem.C.obs[DAnn.donor]).intersection(dropped)


# ── _eagerize_obs_var helper ─────────────────────────────────────────────────


def test_eagerize_obs_var_keeps_X_lazy(gdata):
    from cellink.io._readwrite import _eagerize_obs_var

    lazy = _make_lazy_gdata(gdata)
    eager_ov = _eagerize_obs_var(lazy)
    assert isinstance(eager_ov.X, da.Array)  # X stays dask
    assert isinstance(eager_ov.obs, pd.DataFrame)
    assert isinstance(eager_ov.var, pd.DataFrame)


def test_eagerize_obs_var_preserves_columns_and_index(gdata):
    from cellink.io._readwrite import _eagerize_obs_var

    lazy = _make_lazy_gdata(gdata)
    eager_ov = _eagerize_obs_var(lazy)
    assert list(eager_ov.obs.columns) == list(gdata.obs.columns)
    assert list(eager_ov.var.columns) == list(gdata.var.columns)
    assert list(eager_ov.obs.index) == list(gdata.obs.index)
    assert list(eager_ov.var.index) == list(gdata.var.index)


# ── read_lazy_dd: zarr + new flags ───────────────────────────────────────────


@pytest.mark.slow
def test_read_lazy_dd_zarr_eager_obs_var(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.zarr"
    DonorData(G=gdata, C=adata).write_zarr_dd(str(out))

    dd = read_lazy_dd(str(out), eager_obs_var=True)
    # G: X dask, obs/var pandas
    assert isinstance(dd.G.X, da.Array)
    assert isinstance(dd.G.obs, pd.DataFrame)
    assert isinstance(dd.G.var, pd.DataFrame)
    # C is still eager (default lazy_C=False) — obs/var also pandas
    assert isinstance(dd.C.obs, pd.DataFrame)


@pytest.mark.slow
def test_read_lazy_dd_zarr_lazy_C(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.zarr"
    DonorData(G=gdata, C=adata).write_zarr_dd(str(out))

    dd = read_lazy_dd(str(out), lazy_C=True)
    assert dd._lazy_G is True
    assert dd._lazy_C is True
    assert isinstance(dd.G.X, da.Array)
    assert isinstance(dd.C.X, da.Array)


@pytest.mark.slow
def test_read_lazy_dd_zarr_lazy_C_eager_obs_var(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.zarr"
    eager_dd = DonorData(G=gdata, C=adata)
    eager_dd.write_zarr_dd(str(out))
    expected_C = np.asarray(eager_dd.C.X)

    dd = read_lazy_dd(str(out), lazy_C=True, eager_obs_var=True)
    # X dask on both sides
    assert isinstance(dd.G.X, da.Array)
    assert isinstance(dd.C.X, da.Array)
    # obs/var pandas on both sides
    assert isinstance(dd.G.obs, pd.DataFrame)
    assert isinstance(dd.C.obs, pd.DataFrame)
    # Round-trip values still correct after to_memory
    dd_mem = dd.to_memory()
    np.testing.assert_array_almost_equal(dd_mem.G.X, np.asarray(gdata.X))
    np.testing.assert_array_almost_equal(dd_mem.C.X, expected_C)


# ── read_lazy_dd: HDF5 ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_read_lazy_dd_h5_basic(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd = read_lazy_dd(str(out))
    assert dd._lazy_G is True
    assert dd._lazy_C is False  # default
    assert isinstance(dd.G.X, da.Array)

    dd_mem = dd.to_memory()
    np.testing.assert_array_almost_equal(dd_mem.G.X, np.asarray(gdata.X))


@pytest.mark.slow
def test_read_lazy_dd_h5_eager_obs_var(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd = read_lazy_dd(str(out), eager_obs_var=True)
    assert isinstance(dd.G.X, da.Array)
    assert isinstance(dd.G.obs, pd.DataFrame)
    assert isinstance(dd.G.var, pd.DataFrame)


@pytest.mark.slow
def test_read_lazy_dd_h5_lazy_C(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.h5"
    eager_dd = DonorData(G=gdata, C=adata)
    eager_dd.write_h5_dd(str(out))
    expected_C = np.asarray(eager_dd.C.X)

    dd = read_lazy_dd(str(out), lazy_C=True, eager_obs_var=True)
    assert dd._lazy_G is True
    assert dd._lazy_C is True
    assert isinstance(dd.G.X, da.Array)
    assert isinstance(dd.C.X, da.Array)

    dd_mem = dd.to_memory()
    np.testing.assert_array_almost_equal(dd_mem.G.X, np.asarray(gdata.X))
    np.testing.assert_array_almost_equal(dd_mem.C.X, expected_C)


@pytest.mark.slow
def test_read_lazy_dd_h5_handle_pinned(tmp_path, adata, gdata):
    """The h5py file handle must be pinned on dd._h5_handle so dask reads stay valid."""
    import h5py

    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd = read_lazy_dd(str(out))
    assert hasattr(dd, "_h5_handle")
    assert isinstance(dd._h5_handle, h5py.File)
    assert dd._h5_handle.id.valid  # file is open


@pytest.mark.slow
def test_read_lazy_dd_h5_compute_after_slice(tmp_path, adata, gdata):
    """Slicing then computing on a lazy h5-backed G should return the correct values."""
    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd = read_lazy_dd(str(out))
    keep = list(dd.G.var_names[:2])
    sub = dd.G[:, keep]
    arr = np.asarray(sub.X) if not hasattr(sub.X, "compute") else sub.X.compute()
    assert arr.shape[1] == len(keep)


# ── read_lazy_dd: per-side lazy flags + non-lazy fallback ────────────────────


@pytest.mark.slow
def test_read_lazy_dd_eager_G_lazy_C_zarr(tmp_path, adata, gdata):
    """Uncommon combo: G eager, C lazy."""
    out = tmp_path / "test.dd.zarr"
    DonorData(G=gdata, C=adata).write_zarr_dd(str(out))

    dd = read_lazy_dd(str(out), lazy_G=False, lazy_C=True)
    assert dd._lazy_G is False
    assert dd._lazy_C is True
    assert isinstance(dd.G.X, np.ndarray)
    assert isinstance(dd.C.X, da.Array)


@pytest.mark.slow
def test_read_lazy_dd_both_eager_delegates_zarr(tmp_path, adata, gdata):
    """lazy_G=False, lazy_C=False → behaves as a regular non-lazy read_dd."""
    from cellink.io import read_dd

    out = tmp_path / "test.dd.zarr"
    DonorData(G=gdata, C=adata).write_zarr_dd(str(out))

    dd_eager = read_dd(str(out))
    dd = read_lazy_dd(str(out), lazy_G=False, lazy_C=False)
    assert dd.is_lazy is False
    assert isinstance(dd.G.X, np.ndarray)
    assert isinstance(dd.C.X, np.ndarray)
    assert dd.G.shape == dd_eager.G.shape
    assert dd.C.shape == dd_eager.C.shape
    np.testing.assert_array_almost_equal(dd.G.X, np.asarray(dd_eager.G.X))


@pytest.mark.slow
def test_read_lazy_dd_eager_G_lazy_C_h5(tmp_path, adata, gdata):
    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd = read_lazy_dd(str(out), lazy_G=False, lazy_C=True)
    assert dd._lazy_G is False
    assert dd._lazy_C is True
    assert isinstance(dd.G.X, np.ndarray)
    assert isinstance(dd.C.X, da.Array)


@pytest.mark.slow
def test_read_lazy_dd_both_eager_delegates_h5(tmp_path, adata, gdata):
    from cellink.io import read_dd

    out = tmp_path / "test.dd.h5"
    DonorData(G=gdata, C=adata).write_h5_dd(str(out))

    dd_eager = read_dd(str(out))
    dd = read_lazy_dd(str(out), lazy_G=False, lazy_C=False)
    assert dd.is_lazy is False
    assert isinstance(dd.G.X, np.ndarray)
    assert isinstance(dd.C.X, np.ndarray)
    np.testing.assert_array_almost_equal(dd.G.X, np.asarray(dd_eager.G.X))
    # No h5 handle pinned — both-eager path goes through read_dd which closes the file.
    assert not hasattr(dd, "_h5_handle")


# ── read_lazy_dd dispatch by extension ───────────────────────────────────────


def test_read_lazy_dd_unknown_extension(tmp_path):
    out = tmp_path / "test.unknown"
    out.touch()
    with pytest.raises(ValueError, match="Cannot infer format"):
        read_lazy_dd(str(out))
