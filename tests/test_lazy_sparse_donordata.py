"""Out-of-core sparse single-cell data in DonorData (read_zarr_dd, sel, aggregate, write)."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset

from cellink._core import DonorData
from cellink._core.donordata import _aggregate_out_of_core
from cellink.io import read_zarr_dd


def _make_dd(fmt="csr", n_donors=12, cells_per_donor=25, n_genes=40, seed=0):
    rng = np.random.default_rng(seed)
    donors = [f"D{i:02d}" for i in range(n_donors)]
    n_cells = n_donors * cells_per_donor
    counts = sp.random(n_cells, n_genes, density=0.15, format="csr", random_state=seed, dtype=np.float32)
    counts.data = rng.poisson(3, counts.nnz).astype(np.float32) + 1
    X = counts if fmt == "csr" else counts.tocsc()
    obs = pd.DataFrame(
        {"donor_id": np.repeat(donors, cells_per_donor), "celltype": rng.choice(["T", "B", "NK"], n_cells)},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    C = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    C.layers["counts"] = X.copy()
    G = AnnData(
        X=rng.integers(0, 3, (n_donors, 30)).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(donors, name="donor_id")),
        var=pd.DataFrame(
            {"chrom": "1", "pos": np.arange(30) * 100 + 1, "a0": "A", "a1": "G"}, index=[f"v{k}" for k in range(30)]
        ),
    )
    return DonorData(G=G, C=C, donor_id="donor_id")


@pytest.fixture(params=["csr", "csc"])
def stored(tmp_path, request):
    dd = _make_dd(fmt=request.param)
    path = tmp_path / "dd.dd.zarr"
    dd.write_zarr_dd(str(path))
    return dd, path, request.param


def test_read_zarr_dd_keeps_sparse_c_backed(stored):
    eager, path, fmt = stored
    dd = read_zarr_dd(str(path))
    assert isinstance(dd.C.X, CSRDataset if fmt == "csr" else CSCDataset)
    assert isinstance(dd.C.layers["counts"], CSRDataset if fmt == "csr" else CSCDataset)
    assert dd.C.shape == eager.C.shape


def test_selection_matches_eager_and_toarray_idiom_works(stored):
    eager, path, _ = stored
    dd = read_zarr_dd(str(path))
    mask = (dd.C.obs["celltype"] == "T").to_numpy()
    sub = dd.sel(C_obs=mask)
    X = sub.C.X.toarray() if hasattr(sub.C.X, "toarray") else np.asarray(sub.C.X)
    ref = eager.C.X[(eager.C.obs["celltype"] == "T").to_numpy()].toarray()
    np.testing.assert_array_equal(X, ref)
    assert set(sub.C.obs["donor_id"]) <= set(sub.G.obs_names)


@pytest.mark.parametrize("func", ["sum", "mean", "count_nonzero", "var"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("layer", [None, "counts"])
def test_out_of_core_aggregate_matches_scanpy(stored, func, filtered, layer):
    eager, path, _ = stored
    kw = {
        "func": func,
        "layer": layer,
        "filter_key": "celltype" if filtered else None,
        "filter_value": "T" if filtered else None,
        "key_added": "agg",
    }
    lazy = read_zarr_dd(str(path))
    lazy.aggregate(**kw)
    eager_dd = eager.copy()
    eager_dd.aggregate(**kw)
    got, want = lazy.G.obsm["agg"], eager_dd.G.obsm["agg"]
    assert list(got.columns) == list(want.columns)
    np.testing.assert_allclose(
        got.loc[want.index].to_numpy(float), want.to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True
    )


def test_out_of_core_aggregate_dask_dense_matches_scanpy():
    dd = _make_dd()
    dense = dd.C.X.toarray()
    ref = sc.get.aggregate(AnnData(X=dense, obs=dd.C.obs), by="donor_id", func="mean").layers["mean"]
    got = _aggregate_out_of_core(da.from_array(dense, chunks=(37, 40)), dd.C.obs["donor_id"], "mean")
    np.testing.assert_allclose(got.to_numpy(), np.asarray(ref), rtol=1e-6)


def test_unsupported_func_falls_back_to_selected_cells(stored, caplog):
    eager, path, _ = stored
    lazy = read_zarr_dd(str(path))
    lazy.aggregate(func="median", filter_key="celltype", filter_value="NK", key_added="med")
    eager_dd = eager.copy()
    eager_dd.aggregate(func="median", filter_key="celltype", filter_value="NK", key_added="med")
    want = eager_dd.G.obsm["med"]
    np.testing.assert_allclose(lazy.G.obsm["med"].loc[want.index].to_numpy(float), want.to_numpy(float), equal_nan=True)
    assert "no out-of-core implementation" in caplog.text


def test_write_roundtrip_from_backed_c_stays_sparse(stored, tmp_path):
    eager, path, _ = stored
    lazy = read_zarr_dd(str(path))
    out = tmp_path / "roundtrip.dd.zarr"
    lazy.write_zarr_dd(str(out))
    back = read_zarr_dd(str(out))
    assert isinstance(back.C.X, CSRDataset | CSCDataset)
    np.testing.assert_array_equal(
        back.C.X[...].toarray() if hasattr(back.C.X[...], "toarray") else back.C.X.to_memory().toarray(),
        eager.C.X.toarray(),
    )


@pytest.mark.parametrize("func", ["sum", "mean"])
def test_aggregate_on_view_over_backed_c_streams_and_matches(stored, func):
    """A donor drop after reading turns C into a view; aggregation must still stream and stay correct."""
    eager, path, _ = stored
    lazy = read_zarr_dd(str(path))
    keep = lazy.G.obs_names[2:]  # drop two donors -> their cells leave C
    lazy_sub = DonorData(G=lazy.G[keep].copy(), C=lazy.C, donor_id="donor_id")
    assert lazy_sub.C.is_view
    lazy_sub.aggregate(func=func, key_added="agg")
    eager_sub = DonorData(G=eager.G[keep].copy(), C=eager.C.copy(), donor_id="donor_id")
    eager_sub.aggregate(func=func, key_added="agg")
    want = eager_sub.G.obsm["agg"]
    np.testing.assert_allclose(
        lazy_sub.G.obsm["agg"].loc[want.index].to_numpy(float), want.to_numpy(float), rtol=1e-5, atol=1e-6
    )


def test_selected_rows_path_matches_scanpy(stored, monkeypatch):
    """Force tiny batches so the read-only-selected-rows path runs over many batches."""
    import cellink._core.donordata as ddmod

    eager, path, fmt = stored
    if fmt != "csr":
        pytest.skip("selected-rows path is CSR-only")
    monkeypatch.setattr(ddmod, "_TARGET_NNZ_PER_BLOCK", 50)
    lazy = read_zarr_dd(str(path))
    lazy.aggregate(func="sum", filter_key="celltype", filter_value="B", key_added="agg")
    eager_dd = eager.copy()
    eager_dd.aggregate(func="sum", filter_key="celltype", filter_value="B", key_added="agg")
    want = eager_dd.G.obsm["agg"]
    np.testing.assert_allclose(lazy.G.obsm["agg"].loc[want.index].to_numpy(float), want.to_numpy(float), rtol=1e-6)


def test_var_columns_reads_only_requested_genotype_annotations(stored):
    _, path, _ = stored
    dd = read_zarr_dd(str(path), var_columns=["chrom", "pos"])
    assert list(dd.G.var.columns) == ["chrom", "pos"]
    full = read_zarr_dd(str(path))
    assert list(full.G.var.columns) == ["chrom", "pos", "a0", "a1"]
    assert dd.G.var_names.equals(full.G.var_names)
    assert list(dd.C.var.columns) == list(full.C.var.columns)  # C.var is never column-filtered


@pytest.mark.parametrize("lazy", [False, True])
def test_aggregate_output_frame_matches_previous_fill(tmp_path, lazy):
    # D03 has no T cells, so its row is NaN; the frame must equal what the former object-dtype fill built
    dd = _make_dd()
    dd.C.obs.loc[dd.C.obs["donor_id"] == "D03", "celltype"] = "B"
    if lazy:
        dd.write_zarr_dd(str(tmp_path / "dd.dd.zarr"))
        dd = read_zarr_dd(str(tmp_path / "dd.dd.zarr"))
    eager_C = dd.C.to_memory() if lazy else dd.C
    agg = sc.get.aggregate(eager_C[(eager_C.obs["celltype"] == "T").to_numpy()], by="donor_id", func="sum")
    aggres = pd.DataFrame(agg.layers["sum"], index=agg.obs_names)
    want = pd.DataFrame(index=dd.G.obs_names, columns=dd.C.var_names)
    want.loc[aggres.index] = aggres
    want = want.astype(np.float64)
    dd.aggregate(func="sum", filter_key="celltype", filter_value="T", key_added="pb")
    got = dd.G.obsm["pb"]
    assert got.loc["D03"].isna().all()
    pd.testing.assert_index_equal(got.index, want.index)
    pd.testing.assert_index_equal(got.columns, want.columns)
    np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=1e-6, equal_nan=True)
