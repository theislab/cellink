from __future__ import annotations

import pandas as pd

from cellink.tl.external._ldsc import filter_sumstats_by_merge_alleles


def test_filter_sumstats_by_merge_alleles_matches_real_ldsc_logic(tmp_path):
    # rs1: exact match to reference (A/G vs A/G) -> kept
    # rs2: swapped orientation (C/T vs T/C) -> kept (still a valid, non-ambiguous pair)
    # rs3: strand-ambiguous reference pair (A/T) -> dropped, even with a plausible-looking match
    # rs4: genuinely mismatched alleles (C/G in sumstats vs A/G in reference) -> dropped
    # rs5: not present in the reference at all -> dropped
    sumstats = tmp_path / "sumstats.tsv"
    pd.DataFrame(
        {
            "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "effect_allele": ["A", "C", "A", "C", "A"],
            "other_allele": ["G", "T", "T", "G", "G"],
            "p_value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(sumstats, sep="\t", index=False)

    merge_alleles = tmp_path / "merge_alleles.snplist"
    pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "T", "A", "A"],
            "A2": ["G", "C", "T", "G"],
        }
    ).to_csv(merge_alleles, sep="\t", index=False)

    out_path = tmp_path / "filtered.tsv"
    filter_sumstats_by_merge_alleles(
        str(sumstats),
        str(merge_alleles),
        str(out_path),
        snp_col="variant_id",
        a1_col="effect_allele",
        a2_col="other_allele",
    )

    result = pd.read_csv(out_path, sep="\t")
    assert sorted(result["variant_id"]) == ["rs1", "rs2"]
    assert list(result.columns) == ["variant_id", "effect_allele", "other_allele", "p_value"]
