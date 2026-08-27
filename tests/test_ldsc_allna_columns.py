from __future__ import annotations

import pandas as pd

from cellink.tl.external._ldsc import _drop_allna_columns, _sniff_delimiter


def test_sniff_delimiter_detects_tab(tmp_path):
    f = tmp_path / "in.tsv"
    f.write_text("A\tB\tC\n1\t2\t3\n")
    assert _sniff_delimiter(str(f)) == "\t"


def test_sniff_delimiter_detects_comma(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("A,B,C\n1,2,3\n")
    assert _sniff_delimiter(str(f)) == ","


def test_sniff_delimiter_falls_back_to_whitespace(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("A B C\n1 2 3\n")
    assert _sniff_delimiter(str(f)) == r"\s+"


def test_drop_allna_columns_removes_fully_empty_column_with_real_tab_gap(tmp_path):
    src = tmp_path / "in.tsv"
    src.write_text("SNP\tA1\tA2\tP\tBETA\tvariant_id\n" "rs1\tA\tG\t0.01\t0.1\t\n" "rs2\tA\tG\t0.02\t0.2\t\n")
    out = tmp_path / "out.tsv"

    result_path = _drop_allna_columns(str(src), str(out), chunksize=10_000)

    assert result_path == str(out)
    df = pd.read_csv(result_path, sep="\t")
    assert "variant_id" not in df.columns
    assert list(df.columns) == ["SNP", "A1", "A2", "P", "BETA"]
    assert len(df) == 2


def test_drop_allna_columns_is_noop_when_nothing_all_missing(tmp_path):
    src = tmp_path / "in.tsv"
    src.write_text("SNP\tA1\tA2\tP\n" "rs1\tA\tG\t0.01\n" "rs2\tA\tG\t0.02\n")

    result_path = _drop_allna_columns(str(src), str(tmp_path / "out.tsv"), chunksize=10_000)

    assert result_path == str(src)


def test_drop_allna_columns_streams_across_chunk_boundary(tmp_path):
    rows = ["SNP\tA1\tA2\tP\tEXTRA"]
    for i in range(3):
        rows.append(f"rs{i}\tA\tG\t0.0{i}\t")
    for i in range(3, 6):
        rows.append(f"rs{i}\tA\tG\t0.0{i}\t{i}")
    src = tmp_path / "in.tsv"
    src.write_text("\n".join(rows) + "\n")

    result_path = _drop_allna_columns(str(src), str(tmp_path / "out.tsv"), chunksize=3)

    df = pd.read_csv(result_path, sep="\t")
    assert "EXTRA" in df.columns
    assert len(df) == 6
