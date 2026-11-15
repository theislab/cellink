from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.pl import expression_by_genotype, manhattan, qq
from cellink.pl._ldsc_magma import enrichment_scatter


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def dd():
    adata = sim_adata()
    return DonorData(G=sim_gdata(adata=adata), C=adata).copy()


def test_expression_by_genotype_rejects_unknown_gene(dd):
    with pytest.raises(KeyError, match="GO"):
        expression_by_genotype(dd, snp="SNP0", gene_filter=["GO"], show=False)


def test_expression_by_genotype_all_genes_by_default_and_several_genes(dd):
    assert expression_by_genotype(dd, snp="SNP0", show=False) is not None
    assert expression_by_genotype(dd, snp="SNP0", gene_filter=["G0", "G1"], mode="gene", show=False) is not None


def test_expression_by_genotype_uses_donordata_donor_key():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    adata.obs = adata.obs.rename(columns={"donor_id": "individual"})
    gdata.obs.index.name = "individual"
    dd = DonorData(G=gdata, C=adata, donor_id="individual")
    assert expression_by_genotype(dd, snp="SNP0", gene_filter=["G0"], show=False) is not None


def test_manhattan_leaves_caller_frame_and_reads_chr_prefix():
    df = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chrX", "chr2"], "pos": [10, 20, 5, 7], "pv_adj": [0.1, 1e-9, 0.5, 0.01]}
    )
    before = df.copy()
    manhattan(df, show=False)
    pd.testing.assert_frame_equal(df, before)


def test_qq_splits_given_ax_for_several_groups():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pv_adj": rng.uniform(size=300), "group": np.repeat(["A", "B", "C"], 100)})
    fig, ax = plt.subplots()
    out = qq(df, group_col="group", ax=ax, show=False)
    assert out is fig
    assert len(fig.axes) == 3
    assert sorted(a.get_title() for a in fig.axes) == ["A", "B", "C"]


def test_enrichment_scatter_collapses_duplicate_pairs():
    df = pd.DataFrame(
        {
            "cell_type": ["T", "T", "B"],
            "phenotype": ["scz", "scz", "scz"],
            "neglog10p": [1.0, 4.0, 0.5],
        }
    )
    assert enrichment_scatter(df, show=False) is not None
