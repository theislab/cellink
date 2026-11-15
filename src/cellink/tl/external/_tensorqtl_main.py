import sys

import numpy as np
import pandas as pd


def read_phenotype_bed(phenotype_bed):
    """Load a phenotype BED file as phenotype and position DataFrames.

    Same behaviour as ``tensorqtl.core.read_phenotype_bed`` (tensorqtl 1.0.10, BSD-3-Clause).
    Chromosomes must form contiguous blocks, and positions must be sorted by ``start`` and
    ``end`` within each block.
    """
    if phenotype_bed.lower().endswith((".bed.gz", ".bed")):
        phenotype_df = pd.read_csv(phenotype_bed, sep="\t", index_col=3, dtype={"#chr": str, "#Chr": str})
    elif phenotype_bed.lower().endswith(".bed.parquet"):
        phenotype_df = pd.read_parquet(phenotype_bed)
        phenotype_df = phenotype_df.set_index(phenotype_df.columns[3])
    else:
        raise ValueError("Unsupported file type.")
    phenotype_df = phenotype_df.rename(columns={i: i.lower().replace("#chr", "chr") for i in phenotype_df.columns[:3]})

    phenotype_df["start"] += 1  # change to 1-based
    pos_df = phenotype_df[["chr", "start", "end"]]
    phenotype_df = phenotype_df.drop(["chr", "start", "end"], axis=1)

    chrom_rank = pd.Categorical(pos_df["chr"], categories=pd.unique(pos_df["chr"]), ordered=True).codes
    order = np.lexsort((pos_df["end"].to_numpy(), pos_df["start"].to_numpy(), chrom_rank))
    if not np.array_equal(order, np.arange(len(order))):
        raise AssertionError("Positions in BED file must be sorted.")

    if (pos_df["start"] == pos_df["end"]).all():
        pos_df = pos_df[["chr", "end"]].rename(columns={"end": "pos"})

    return phenotype_df, pos_df


def main():
    """Run ``tensorqtl.main()`` with the compatibility fixes applied."""
    from cellink.tl.external._tensorqtl import _ensure_tensorqtl_importable

    _ensure_tensorqtl_importable()
    import tensorqtl.tensorqtl as tensorqtl_cli

    tensorqtl_cli.read_phenotype_bed = read_phenotype_bed
    sys.argv[0] = "tensorqtl"
    tensorqtl_cli.main()


if __name__ == "__main__":
    main()
