from __future__ import annotations

from cellink.tl.external._ldsc import _validate_sumstats_pre_munge


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_raises_on_negative_log10_pvalue_column(tmp_path):
    path = _write(
        tmp_path,
        "regenie_style.tsv",
        "SNP\tCHR\tBP\tA1\tA2\tLOG10P\n"
        "rs1\t1\t100\tA\tG\t3.5\n"
        "rs2\t1\t200\tA\tG\t8.2\n"
        "rs3\t1\t300\tA\tG\t1.1\n",
    )
    try:
        _validate_sumstats_pre_munge(path, p_col="LOG10P", snp_col="SNP", merge_alleles=None)
        raise AssertionError("expected ValueError for -log10(p) column")
    except ValueError as e:
        assert "log10" in str(e).lower() or "p-value" in str(e).lower()


def test_accepts_genuine_pvalue_column(tmp_path):
    path = _write(
        tmp_path,
        "normal.tsv",
        "SNP\tCHR\tBP\tA1\tA2\tP\n" "rs1\t1\t100\tA\tG\t0.01\n" "rs2\t1\t200\tA\tG\t0.5\n" "rs3\t1\t300\tA\tG\t0.9\n",
    )
    _validate_sumstats_pre_munge(path, p_col="P", snp_col="SNP", merge_alleles=None)


def test_raises_on_non_rsid_snp_column_with_merge_alleles(tmp_path):
    path = _write(
        tmp_path,
        "positional_ids.tsv",
        "SNP\tCHR\tBP\tA1\tA2\tP\n"
        "1:100:A:G\t1\t100\tA\tG\t0.01\n"
        "1:200:A:G\t1\t200\tA\tG\t0.5\n"
        "1:300:A:G\t1\t300\tA\tG\t0.9\n",
    )
    try:
        _validate_sumstats_pre_munge(path, p_col="P", snp_col="SNP", merge_alleles="w_hm3.snplist")
        raise AssertionError("expected ValueError for non-rsID SNP column with merge_alleles")
    except ValueError as e:
        assert "rsid" in str(e).lower() or "id-scheme" in str(e).lower() or "id scheme" in str(e).lower()


def test_accepts_rsid_snp_column_with_merge_alleles(tmp_path):
    path = _write(
        tmp_path,
        "rsid_ids.tsv",
        "SNP\tCHR\tBP\tA1\tA2\tP\n" "rs1\t1\t100\tA\tG\t0.01\n" "rs2\t1\t200\tA\tG\t0.5\n" "rs3\t1\t300\tA\tG\t0.9\n",
    )
    _validate_sumstats_pre_munge(path, p_col="P", snp_col="SNP", merge_alleles="w_hm3.snplist")
