"""Tests for lazy DonorData support (dask-backed G)."""

import numpy as np
import pandas as pd
import pytest

import dask.array as da
from anndata import AnnData

from cellink._core.data_fields import DAnn
from cellink._core.donordata import DonorData, _has_dask_X
from cellink.io import read_lazy_dd, read_zarr_dd


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_lazy_gdata(gdata: AnnData) -> AnnData:
    """Convert an in-memory AnnData to one with a dask-backed X."""
    lazy_X = da.from_array(np.asarray(gdata.X), chunks=(gdata.n_obs, 2))
    return AnnData(X=lazy_X, obs=gdata.obs.copy(), var=gdata.var.copy())


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
    """Lazy G with extra donors should be subset to those present in C."""
    extra_obs = pd.DataFrame(index=pd.Index(["EXTRA_DONOR"], name=DAnn.donor))
    extra_X = da.zeros((1, gdata.n_vars), chunks=(1, gdata.n_vars))
    extra = AnnData(
        X=da.concatenate([da.from_array(np.asarray(gdata.X)), extra_X], axis=0),
        obs=pd.concat([gdata.obs, extra_obs]),
        var=gdata.var.copy(),
    )
    dd = DonorData(G=extra, C=adata)
    # EXTRA_DONOR has no cells in C, so should be dropped
    assert "EXTRA_DONOR" not in dd.G.obs_names


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


def test_aggregate_raises_on_lazy(adata, gdata):
    lazy_g = _make_lazy_gdata(gdata)
    dd = DonorData(G=lazy_g, C=adata)
    with pytest.raises(RuntimeError, match="lazy"):
        dd.aggregate(key_added="test")


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
