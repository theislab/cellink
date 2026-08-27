from __future__ import annotations

import pandas as pd

from cellink.tl.external._ldsc import combine_chr_ld_scores


def test_combine_frq_only_does_not_require_ldscore_files(tmp_path):
    chr_prefix = str(tmp_path / "1000G.EUR.QC.")
    for chrom in range(1, 4):
        pd.DataFrame({"SNP": [f"rs{chrom}"], "A1": ["A"], "A2": ["G"], "MAF": [0.2]}).to_csv(
            f"{chr_prefix}{chrom}.frq", sep="\t", index=False
        )

    out_dir = tmp_path / "combined"
    combined_prefix = combine_chr_ld_scores(
        chr_prefix, str(out_dir), num_chr=3, combine_frq=True, combine_ldscore=False
    )

    assert (out_dir / "1000G.EUR.QC.frq").exists()
    assert not (out_dir / "1000G.EUR.QC.l2.ldscore.gz").exists()
    combined = pd.read_csv(out_dir / "1000G.EUR.QC.frq", sep="\t")
    assert len(combined) == 3
