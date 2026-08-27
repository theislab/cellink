from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

torchnmf = pytest.importorskip("torchnmf")

from cellink.tl.external._sclinker import compute_nmf_programs  # noqa: E402


def test_compute_nmf_programs_torchnmf_backend_orients_w_and_h_correctly():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 200, 500
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.obs["predicted.celltype.l2"] = rng.choice(["A", "B", "C"], size=n_cells)

    W, H, corr = compute_nmf_programs(adata, celltype_col="predicted.celltype.l2", device="cpu", save=False)

    assert W.shape[0] == n_cells
    assert H.shape[0] == n_genes
    assert list(W.index) == list(adata.obs_names)
    assert list(H.index) == list(adata.var_names)
