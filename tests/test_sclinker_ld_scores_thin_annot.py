from __future__ import annotations

import gzip

import pytest

from cellink.tl.external._sclinker_utils import compute_ld_scores_for_sclinker


class _RecordingRunner:
    ldsc_command = "ldsc.py"

    def __init__(self):
        self.commands = []

    def run_command(self, cmd, file_paths=None, check=True):
        self.commands.append(cmd)


class _AllFailRunner:
    ldsc_command = "ldsc.py"

    def run_command(self, cmd, file_paths=None, check=True):
        raise RuntimeError("Error parsing .annot file")


def _write_thin_annot(path, n_snps=5):
    with gzip.open(path, "wt") as f:
        f.write("ANNOT\n")
        for _ in range(n_snps):
            f.write("0.0\n")


def test_compute_ld_scores_for_sclinker_passes_thin_annot(tmp_path):
    # the sc-linker annotation writers write thin annotations (a single ANNOT column, no
    # CHR/BP/SNP/CM), so compute_ld_scores_with_annotations_from_bimfile gets thin_annot=True
    annot_dir = tmp_path / "annotations" / "myprog" / "ABC_Road_BLD"
    annot_dir.mkdir(parents=True)
    _write_thin_annot(annot_dir / "myprog.1.annot.gz")

    annotation_prefixes = {"myprog": {"ABC_Road_BLD": str(annot_dir / "myprog.")}}

    runner = _RecordingRunner()
    compute_ld_scores_for_sclinker(
        annotation_prefixes=annotation_prefixes,
        bim_prefix=str(tmp_path / "1000G.EUR.QC."),
        ld_scores_dir=tmp_path / "ldscores",
        chromosomes=[1],
        n_jobs=1,
        runner=runner,
    )

    assert len(runner.commands) == 1
    assert "--thin-annot" in runner.commands[0]


def test_compute_ld_scores_for_sclinker_raises_if_all_jobs_fail(tmp_path):
    annot_dir = tmp_path / "annotations" / "myprog" / "ABC_Road_BLD"
    annot_dir.mkdir(parents=True)
    _write_thin_annot(annot_dir / "myprog.1.annot.gz")
    _write_thin_annot(annot_dir / "myprog.2.annot.gz")

    annotation_prefixes = {"myprog": {"ABC_Road_BLD": str(annot_dir / "myprog.")}}

    with pytest.raises(RuntimeError, match="All 2 LD score job"):
        compute_ld_scores_for_sclinker(
            annotation_prefixes=annotation_prefixes,
            bim_prefix=str(tmp_path / "1000G.EUR.QC."),
            ld_scores_dir=tmp_path / "ldscores",
            chromosomes=[1, 2],
            n_jobs=1,
            runner=_AllFailRunner(),
        )
