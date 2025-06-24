from pathlib import Path

import numpy as np
import pandas as pd

from cellink import DonorData
from cellink.pl import expression_by_genotype, locus, manhattan, qq, volcano

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

    fig = expression_by_genotype(dd, snp="SNP0", gene_filter=["GO"])
    assert fig is not None
