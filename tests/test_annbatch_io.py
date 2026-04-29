from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

# Skip the entire module if the optional extra isn't installed.
pytest.importorskip("annbatch", reason="needs cellink[annbatch] extra")

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.io import open_annbatch_loader, write_annbatch_collection


def _make_cells(n_obs: int = 200, n_vars: int = 50, *, with_layer: bool = False, seed: int = 0) -> AnnData:
    X = sp.random(n_obs, n_vars, density=0.2, format="csr", random_state=seed).astype(np.float32)
    obs = pd.DataFrame(
        {
            "donor_id": [f"D{i % 5}" for i in range(n_obs)],
            "cell_label": [f"L{i % 3}" for i in range(n_obs)],
        },
        index=[f"C{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)
    if with_layer:
        # Use distinctive values so we can tell layer from X by content.
        counts = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=seed + 1).astype(np.float32)
        counts.data += 100.0  # make values clearly larger than X
        adata.layers["counts"] = counts
    return adata


def _unpack_batch(batch):
    """Return (X, obs) from an annbatch LoaderOutput (a TypedDict)."""
    return batch["X"], batch["obs"]


def _iter_loader(loader):
    """Iterate the loader and collect (obs frames, X-row counts, n_cols)."""
    obs_frames: list[pd.DataFrame] = []
    x_rows: list[int] = []
    x_cols: int | None = None
    for batch in loader:
        x, obs = _unpack_batch(batch)
        n_rows = int(x.shape[0])
        x_rows.append(n_rows)
        if x_cols is None:
            x_cols = int(x.shape[1])
        else:
            assert int(x.shape[1]) == x_cols
        obs_frames.append(obs if isinstance(obs, pd.DataFrame) else pd.DataFrame(obs))
    return obs_frames, x_rows, x_cols


def test_roundtrip_anndata(tmp_path: Path) -> None:
    adata = _make_cells(n_obs=200, n_vars=50)
    out = tmp_path / "collection.zarr"
    written = write_annbatch_collection(
        adata,
        out,
        obs_keys=["donor_id", "cell_label"],
    )
    assert Path(written).exists()
    assert Path(written) == out

    loader = open_annbatch_loader(
        out,
        obs_keys=["donor_id", "cell_label"],
        batch_size=64,
        chunk_size=16,
        preload_nchunks=8,
        to_torch=False,
        shuffle=False,
    )
    obs_frames, x_rows, x_cols = _iter_loader(loader)

    assert sum(x_rows) == adata.n_obs
    assert all(r <= 64 for r in x_rows)
    assert x_cols == adata.n_vars

    obs_concat = pd.concat(obs_frames, ignore_index=True)
    assert list(obs_concat.columns) == ["donor_id", "cell_label"]
    assert len(obs_concat) == adata.n_obs


def test_roundtrip_donordata(tmp_path: Path) -> None:
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    dd = DonorData(G=gdata, C=adata)

    out = tmp_path / "collection.zarr"
    write_annbatch_collection(dd, out, obs_keys=["donor_id", "celltype"])

    loader = open_annbatch_loader(
        out,
        obs_keys=["donor_id", "celltype"],
        batch_size=32,
        chunk_size=8,
        preload_nchunks=4,
        to_torch=False,
        shuffle=False,
    )
    obs_frames, x_rows, _ = _iter_loader(loader)

    assert sum(x_rows) == dd.C.n_obs

    obs_concat = pd.concat(obs_frames, ignore_index=True)
    seen_donors = set(obs_concat["donor_id"].astype(str).unique())
    known_donors = set(dd.G.obs_names.astype(str))
    assert seen_donors.issubset(known_donors)


def test_layer_selection(tmp_path: Path) -> None:
    adata = _make_cells(n_obs=200, n_vars=50, with_layer=True)
    out = tmp_path / "collection.zarr"
    write_annbatch_collection(
        adata,
        out,
        obs_keys=["donor_id", "cell_label"],
        layer="counts",
    )

    loader = open_annbatch_loader(
        out,
        obs_keys=["donor_id", "cell_label"],
        layer="counts",
        batch_size=64,
        chunk_size=16,
        preload_nchunks=8,
        to_torch=False,
        shuffle=False,
    )
    _, _, _ = _iter_loader(loader)

    # The stored layer values had +100.0 added; X values do not. So checking
    # the maximum across all batches discriminates between X and layers["counts"].
    loader2 = open_annbatch_loader(
        out,
        obs_keys=["donor_id", "cell_label"],
        layer="counts",
        batch_size=64,
        chunk_size=16,
        preload_nchunks=8,
        to_torch=False,
        shuffle=False,
    )
    max_layer_val = 0.0
    for batch in loader2:
        x, _ = _unpack_batch(batch)
        arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        if arr.size and arr.max() > max_layer_val:
            max_layer_val = float(arr.max())
    assert max_layer_val >= 100.0, "Expected layer values to exceed 100.0; got X instead?"


def test_missing_extra_error_message() -> None:
    """If `annbatch` is missing, accessing the symbol gives a clear hint.

    The whole module is gated by ``importorskip("annbatch")`` at the top, so
    when the extra is installed we can only sanity-check that the error path
    in ``cellink.io.__init__`` references the install hint.
    """
    import cellink.io as cio

    src = Path(cio.__file__).read_text()
    assert "pip install cellink[annbatch]" in src
