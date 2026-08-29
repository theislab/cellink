from __future__ import annotations

import numpy as np
import pytest

from cellink._core.donordata import DonorData
from cellink.tl.external._tensorqtl import run_dense_trans_scan

pytest.importorskip("tensorqtl", reason="tensorqtl not installed")


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    return DonorData(G=gdata, C=adata)


@pytest.fixture
def far_variant_id(dd):
    """Move the first variant's position far past every gene's own window,
    so it behaves like a genuine trans (unlinked) variant: nothing should
    be dropped by tensorqtl's own cis-window filter, unlike `dummy_data`'s
    default variant placement (deliberately inside some gene's window, to
    exercise cis-mode tests elsewhere)."""
    variant_id = dd.G.var_names[0]
    far_pos = int(dd.C.var["end"].max()) + 10_000_000
    dd.G.var.loc[variant_id, "pos"] = far_pos
    return variant_id


def test_run_dense_trans_scan_ranks_all_genes(dd, far_variant_id):
    """A dense scan (pval_threshold=1.0 under the hood) must return every
    gene that survives the cis-window filter, sorted by ascending p-value
    and numbered 1..N by `rank_by_pval` -- not just the sparse subset a
    normal trans scan would keep."""
    df = run_dense_trans_scan(dd, variant_id=far_variant_id, maf_threshold=0.0, encode_sex=False)

    assert df["variant_id"].eq(far_variant_id).all()
    assert list(df["rank_by_pval"]) == list(range(1, len(df) + 1))
    assert df["pval"].is_monotonic_increasing
    assert {"chrom", "start", "end"}.issubset(df.columns)
    # a genuinely unlinked (far-away) variant shouldn't lose any gene to the cis-window filter
    assert len(df) == dd.C.n_vars


def test_run_dense_trans_scan_raises_on_ambiguous_variant(dd):
    with pytest.raises(ValueError, match="expected exactly 1 match"):
        run_dense_trans_scan(dd, variant_id="not-a-real-variant", encode_sex=False)


def test_run_dense_trans_scan_logs_gene_of_interest_rank(dd, far_variant_id, caplog):
    gene_of_interest = dd.C.var_names[0]
    with caplog.at_level("INFO", logger="cellink.tl.external._tensorqtl"):
        df = run_dense_trans_scan(
            dd, variant_id=far_variant_id, gene_of_interest=gene_of_interest,
            maf_threshold=0.0, encode_sex=False,
        )
    goi_row = df[df["phenotype_id"] == gene_of_interest]
    assert len(goi_row) == 1
    assert any(gene_of_interest in message and "rank=" in message for message in caplog.messages)


def test_run_dense_trans_scan_warns_on_missing_gene_of_interest(dd, caplog):
    variant_id = dd.G.var_names[0]
    with caplog.at_level("WARNING", logger="cellink.tl.external._tensorqtl"):
        run_dense_trans_scan(
            dd, variant_id=variant_id, gene_of_interest="not-a-real-gene",
            maf_threshold=0.0, encode_sex=False,
        )
    assert any("not-a-real-gene" in message and "not found" in message for message in caplog.messages)


def test_run_dense_trans_scan_truncates_gpcs_to_n_pcs(dd, monkeypatch):
    """gPCs (if present) are auto-added as a covariate and truncated to
    n_pcs columns before being handed to run_tensorqtl."""
    rng = np.random.default_rng(0)
    dd.G.obsm["gPCs"] = rng.normal(size=(dd.G.n_obs, 10))
    variant_id = dd.G.var_names[0]

    captured = {}

    import cellink.tl.external._tensorqtl as tqtl_module

    real_run_tensorqtl = tqtl_module.run_tensorqtl

    def spy(dd_scan, *args, **kwargs):
        captured["gpcs_shape"] = dd_scan.G.obsm["gPCs"].shape
        captured["additional_covariates"] = kwargs.get("additional_covariates")
        return real_run_tensorqtl(dd_scan, *args, **kwargs)

    monkeypatch.setattr(tqtl_module, "run_tensorqtl", spy)

    run_dense_trans_scan(dd, variant_id=variant_id, n_pcs=3, maf_threshold=0.0, encode_sex=False)

    assert captured["gpcs_shape"][1] == 3
    assert captured["additional_covariates"] == ["gPCs"]
