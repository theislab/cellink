from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.pl import expression_by_genotype, locus, manhattan, qq, volcano
from cellink.pl._ldsc_magma import enrichment_scatter

DATA = Path("tests/data")


def test_locus():
    locus_df = pd.DataFrame(
        {
            "pos": np.linspace(1_000_000, 1_100_000, 300),
            "pval": np.random.uniform(1e-6, 1, 300),
            "snp": [f"rs{i}" for i in range(300)],
        }
    )

    gene_df = pd.DataFrame(
        {
            "gene": ["GENE1", "GENE2", "GENE3"],
            "start": [1_015_000, 1_045_000, 1_075_000],
            "end": [1_025_000, 1_060_000, 1_090_000],
            "strand": ["+", "-", "+"],
        }
    )

    fig = locus(
        locus_df=locus_df,
        gene_df=gene_df,
        label_column="snp",
        significance_threshold=1e-2,
    )

    assert fig is not None


def test_volcano():
    n = 1000
    df = pd.DataFrame(
        {
            "logFC": np.random.randn(n),
            "pval": np.random.beta(0.5, 10, size=n),
            "gene": [f"Gene_{i}" for i in range(n)],
        }
    )

    fig = volcano(df, label_col="gene")

    assert fig is not None


def test_qq():
    n_samples_per_group = 500
    groups = ["A", "B", "C"]

    data = []
    for group in groups:
        null_pvals = np.random.uniform(0, 1, int(0.8 * n_samples_per_group))
        alt_pvals = np.random.beta(a=0.5, b=10, size=int(0.2 * n_samples_per_group))
        pvals = np.concatenate([null_pvals, alt_pvals])
        data.extend(zip(pvals, [group] * n_samples_per_group, strict=False))

    test_df = pd.DataFrame(data, columns=["pv_adj", "group"])

    fig = qq(test_df)
    assert fig is not None


def test_manhattan():
    data = pd.DataFrame(
        {
            "chrom": [1, 1, 1, 2, 2, 2],
            "pos": [100, 200, 300, 400, 500, 600],
            "pv_adj": [0.05, 0.01, 0.5, 1e-6, 0.2, 0.9],
        }
    )
    fig = manhattan(data)
    assert fig is not None


def test_expression_by_genotype(tmp_path, adata, gdata):
    dd = DonorData(G=gdata, C=adata).copy()
    print(dd.C.obs)
    print(dd.C.var)

    fig = expression_by_genotype(dd, snp="SNP0", gene_filter=["G0"])
    assert fig is not None


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
