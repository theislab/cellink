from __future__ import annotations

import pandas as pd
import pytest

from cellink.tl import compare_gene_pair_effects

VARIANT = "1:100:A:G"
GENE_A = "ENSG_A"
GENE_B = "ENSG_B"


def _write_celltype_parquet(root, cohort, celltype, rows):
    celltype_dir = root / cohort / celltype
    celltype_dir.mkdir(parents=True)
    pd.DataFrame(rows, columns=["gene", "variant_id", "beta", "se", "pval"]).to_parquet(
        celltype_dir / "tensorqtl.parquet"
    )
    return celltype_dir


def test_compare_gene_pair_effects_keeps_only_celltypes_with_both_genes(tmp_path):
    _write_celltype_parquet(
        tmp_path, "ukb_european", "NK_CD16",
        [
            (GENE_A, VARIANT, 0.3, 0.05, 1e-8),
            (GENE_B, VARIANT, -0.25, 0.06, 1e-6),
            (GENE_A, "1:200:A:G", 0.1, 0.05, 0.2), 
        ],
    )
    
    _write_celltype_parquet(tmp_path, "ukb_european", "T_CD4_naive", [(GENE_A, VARIANT, 0.1, 0.02, 0.01)])

    res = compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="ukb_european")

    assert list(res["celltype"].unique()) == ["NK_CD16"]
    assert list(res["gene"]) == [GENE_A, GENE_B]  
    assert list(res.columns) == ["gene", "variant_id", "beta", "se", "pval", "celltype"]


def test_compare_gene_pair_effects_orders_gene_b_first_row_matches_gene_b(tmp_path):
    """Row order within a celltype always follows (gene_a, gene_b), regardless
    of the row order on disk."""
    _write_celltype_parquet(
        tmp_path, "ukb_european", "T_gd",
        [
            (GENE_B, VARIANT, -0.4, 0.07, 1e-5),
            (GENE_A, VARIANT, 0.35, 0.06, 1e-7),
        ],
    )
    res = compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="ukb_european")
    assert list(res["gene"]) == [GENE_A, GENE_B]
    assert res.iloc[0]["beta"] == pytest.approx(0.35)
    assert res.iloc[1]["beta"] == pytest.approx(-0.4)


def test_compare_gene_pair_effects_discovers_celltypes_when_none_given(tmp_path):
    _write_celltype_parquet(
        tmp_path, "ukb_european", "NK_CD16",
        [(GENE_A, VARIANT, 0.3, 0.05, 1e-8), (GENE_B, VARIANT, -0.25, 0.06, 1e-6)],
    )
    _write_celltype_parquet(
        tmp_path, "ukb_european", "T_CD8_EM",
        [(GENE_A, VARIANT, 0.2, 0.04, 1e-4), (GENE_B, VARIANT, -0.15, 0.05, 1e-3)],
    )
    res = compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="ukb_european")
    assert set(res["celltype"]) == {"NK_CD16", "T_CD8_EM"}


def test_compare_gene_pair_effects_returns_empty_frame_when_nothing_matches(tmp_path):
    _write_celltype_parquet(tmp_path, "ukb_european", "NK_CD16", [(GENE_A, VARIANT, 0.3, 0.05, 1e-8)])
    res = compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="ukb_european")
    assert res.empty
    assert list(res.columns) == ["gene", "variant_id", "beta", "se", "pval", "celltype"]


def test_compare_gene_pair_effects_raises_when_cohort_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="no_such_cohort")


def test_compare_gene_pair_effects_skips_celltype_missing_parquet(tmp_path):
    _write_celltype_parquet(
        tmp_path, "ukb_european", "NK_CD16",
        [(GENE_A, VARIANT, 0.3, 0.05, 1e-8), (GENE_B, VARIANT, -0.25, 0.06, 1e-6)],
    )
    (tmp_path / "ukb_european" / "T_gd").mkdir(parents=True)  # no tensorqtl.parquet inside

    res = compare_gene_pair_effects(tmp_path, VARIANT, GENE_A, GENE_B, cohort="ukb_european")
    assert set(res["celltype"]) == {"NK_CD16"}
