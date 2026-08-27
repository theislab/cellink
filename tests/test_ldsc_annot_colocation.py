from __future__ import annotations

import gzip
import os

from cellink.tl.external._ldsc import (
    LDSCRunner,
    _colocate_annot_file,
    compute_ld_scores_with_annotations_from_bimfile,
)


def test_colocate_annot_file_copies_when_directories_differ(tmp_path):
    annot_dir = tmp_path / "annotations"
    annot_dir.mkdir()
    annot_file = annot_dir / "CD8_Naive.22.annot.gz"
    with gzip.open(annot_file, "wt") as f:
        f.write("CHR\tBP\tSNP\tCM\tANN\n1\t100\trs1\t0\t1\n")

    ldscore_dir = tmp_path / "ldscores"
    ldscore_dir.mkdir()
    out_prefix = str(ldscore_dir / "CD8_Naive.22")

    target = _colocate_annot_file(str(annot_file), out_prefix)

    expected = out_prefix + ".annot.gz"
    assert target == expected
    assert os.path.isfile(expected)
    with gzip.open(expected, "rt") as f:
        assert f.read() == "CHR\tBP\tSNP\tCM\tANN\n1\t100\trs1\t0\t1\n"


def test_colocate_annot_file_noop_when_already_at_out_prefix(tmp_path):
    out_prefix = str(tmp_path / "CD8_Naive.22")
    annot_file = out_prefix + ".annot.gz"
    with gzip.open(annot_file, "wt") as f:
        f.write("x\n")

    assert _colocate_annot_file(annot_file, out_prefix) is None


def test_colocate_annot_file_noop_when_target_already_exists(tmp_path):
    annot_dir = tmp_path / "annotations"
    annot_dir.mkdir()
    annot_file = annot_dir / "CD8_Naive.22.annot.gz"
    with gzip.open(annot_file, "wt") as f:
        f.write("source\n")

    ldscore_dir = tmp_path / "ldscores"
    ldscore_dir.mkdir()
    out_prefix = str(ldscore_dir / "CD8_Naive.22")
    existing_target = out_prefix + ".annot.gz"
    with gzip.open(existing_target, "wt") as f:
        f.write("preexisting\n")

    assert _colocate_annot_file(str(annot_file), out_prefix) is None
    with gzip.open(existing_target, "rt") as f:
        assert f.read() == "preexisting\n"


def test_compute_ld_scores_with_annotations_from_bimfile_colocates_annot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    annot_dir = tmp_path / "annotations"
    annot_dir.mkdir()
    annot_file = annot_dir / "CD8_Naive.22.annot.gz"
    with gzip.open(annot_file, "wt") as f:
        f.write("CHR\tBP\tSNP\tCM\tANN\n1\t100\trs1\t0\t1\n")

    ldscore_dir = tmp_path / "ldscores"
    ldscore_dir.mkdir()
    out_prefix = str(ldscore_dir / "CD8_Naive.22")

    runner = LDSCRunner(
        config_dict={
            "execution_mode": "local",
            "ldsc_command": "true",  
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }
    )

    result = compute_ld_scores_with_annotations_from_bimfile(
        bfile_prefix=str(tmp_path / "dummy"),
        annot_file=str(annot_file),
        out_prefix=out_prefix,
        runner=runner,
    )

    expected_colocated = out_prefix + ".annot.gz"
    assert os.path.isfile(expected_colocated)
    assert expected_colocated in result["files_created"]
