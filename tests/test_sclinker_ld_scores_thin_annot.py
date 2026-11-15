from __future__ import annotations

import gzip
from pathlib import Path

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


class _FakeLdscRunner:
    """Writes what ldsc --l2 writes for a thin annotation with columns P0, P1, ..."""

    ldsc_command = "ldsc.py"

    def __init__(self):
        self.commands = []

    def run_command(self, cmd, file_paths=None, check=True):
        import shlex

        import pandas as pd

        self.commands.append(cmd)
        args = shlex.split(cmd)
        out, annot = args[args.index("--out") + 1], args[args.index("--annot") + 1]
        cols = list(pd.read_csv(annot, sep="\t").columns)
        ld = pd.DataFrame({"CHR": [1, 1], "SNP": ["rs1", "rs2"], "BP": [10, 20]})
        for k, col in enumerate(cols):
            ld[f"{col}L2"] = [1.5 + k, 2.5 + k]
        ld.to_csv(f"{out}.l2.ldscore.gz", sep="\t", index=False, compression="gzip")
        with open(f"{out}.l2.M", "w") as f:
            f.write("\t".join(str(10 * (k + 1)) for k in range(len(cols))) + "\n")
        with open(f"{out}.l2.M_5_50", "w") as f:
            f.write("\t".join(str(5 * (k + 1)) for k in range(len(cols))) + "\n")


def test_compute_ld_scores_for_sclinker_runs_the_annotations_of_a_chromosome_together(tmp_path):
    import pandas as pd

    annotation_prefixes = {}
    for prog in ("progA", "progB"):
        annotation_prefixes[prog] = {}
        for strategy in ("ABC_Road_BLD", "100kb"):
            annot_dir = tmp_path / "annotations" / prog / strategy
            annot_dir.mkdir(parents=True)
            _write_thin_annot(annot_dir / f"{prog}.1.annot.gz", n_snps=2)
            annotation_prefixes[prog][strategy] = str(annot_dir / f"{prog}.")

    runner = _FakeLdscRunner()
    ld_prefixes = compute_ld_scores_for_sclinker(
        annotation_prefixes=annotation_prefixes,
        bim_prefix=str(tmp_path / "1000G.EUR.QC."),
        ld_scores_dir=tmp_path / "ldscores",
        chromosomes=[1],
        n_jobs=2,
        runner=runner,
    )

    assert len(runner.commands) == 1  # one ldsc run for both programs and strategies
    columns = [(prog, strategy) for prog in ("progA", "progB") for strategy in ("ABC_Road_BLD", "100kb")]
    for k, (prog, strategy) in enumerate(columns):
        prefix = ld_prefixes[prog][strategy] + "1"
        ld = pd.read_csv(f"{prefix}.l2.ldscore.gz", sep="\t")
        assert list(ld.columns) == ["CHR", "SNP", "BP", "L2"]
        assert list(ld["L2"]) == [1.5 + k, 2.5 + k]
        with open(f"{prefix}.l2.M") as f:
            assert f.read().split() == [str(10 * (k + 1))]
        with open(f"{prefix}.l2.M_5_50") as f:
            assert f.read().split() == [str(5 * (k + 1))]
        # --overlap-annot reads the annotation at the LD score prefix; a real file, since a link
        # to the annotations directory is not visible inside the ldsc container
        copied = Path(f"{prefix}.annot.gz")
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_bytes() == Path(annotation_prefixes[prog][strategy] + "1.annot.gz").read_bytes()


def test_run_sclinker_heritability_runs_every_fit_in_parallel(tmp_path, monkeypatch):
    from cellink.tl.external import _ldsc
    from cellink.tl.external._sclinker_utils import run_sclinker_heritability

    seen = []
    monkeypatch.setattr(_ldsc, "estimate_heritability", lambda **kw: seen.append((kw["ref_ld_chr"], kw["out_prefix"])))
    ld_prefixes = {p: {s: str(tmp_path / "ldscores" / p / s / f"{p}.") for s in ("100kb", "ABC")} for p in ("a", "b")}
    out = run_sclinker_heritability(
        ld_prefixes=ld_prefixes,
        sumstats_files=[str(tmp_path / "t1.sumstats.gz"), str(tmp_path / "t2.sumstats.gz")],
        ref_ld_chr="baseline.",
        w_ld_chr="weights.",
        out_dir=tmp_path / "res",
        frqfile_chr="frq.",
        n_jobs=4,
        runner=object(),
    )
    assert len(seen) == 8
    assert {pref for _, pref in seen} == {
        out[p][s][t].removesuffix(".log") for p in out for s in out[p] for t in out[p][s]
    }


def test_genescores_to_annotations_writes_the_same_files_in_parallel(tmp_path, monkeypatch):
    import pandas as pd

    from cellink.tl.external import _sclinker_utils

    def bedgraphs(genescores, *args, **kwargs):
        return {
            p: pd.DataFrame({"chr": ["1", "2"], "start": [100, 250], "end": [300, 400], "score": [0.5 + k, 0.25]})
            for k, p in enumerate(genescores.columns)
        }

    monkeypatch.setattr(_sclinker_utils, "genescores_to_abc_road_bedgraph", bedgraphs)
    monkeypatch.setattr(_sclinker_utils, "genescores_to_100kb_bedgraph", bedgraphs)
    for chrom in (1, 2, 3):
        rows = [f"{chrom}\trs{chrom}_{bp}\t0\t{bp}\tA\tG" for bp in (50, 150, 260, 350, 500)]
        (tmp_path / f"ref.{chrom}.bim").write_text("\n".join(rows) + "\n")
    genescores = pd.DataFrame({"progA": [1.0], "progB": [0.5]}, index=["G1"])

    def annotate(out_dir, n_jobs):
        prefixes = _sclinker_utils.genescores_to_annotations(
            genescores,
            roadmap_links=None,
            abc_links=None,
            gene_annotation=None,
            bim_prefix=str(tmp_path / "ref."),
            out_dir=tmp_path / out_dir,
            chromosomes=[1, 2, 3],
            use_bedtools=False,
            on_build_mismatch="ignore",
            n_jobs=n_jobs,
        )
        return {
            (prog, strategy, chrom): pd.read_csv(f"{prefix}{chrom}.annot.gz", sep="\t")["ANNOT"].tolist()
            for prog, strategies in prefixes.items()
            for strategy, prefix in strategies.items()
            for chrom in (1, 2, 3)
        }

    serial = annotate("serial", n_jobs=1)
    assert len(serial) == 2 * 2 * 3
    assert serial[("progB", "100kb", 1)] == [0.0, 1.5, 1.5, 0.0, 0.0]
    assert serial[("progA", "100kb", 3)] == [0.0] * 5  # no bedgraph interval on chr3
    assert annotate("parallel", n_jobs=4) == serial


def test_annotate_python_reads_bedgraph_coordinates_like_bedtools():
    import numpy as np
    import pandas as pd

    from cellink.tl.external._sclinker_utils import _annotate_python

    bg = pd.DataFrame(
        {"chr": ["chr1", "1", "2"], "start": [100, 250, 0], "end": [300, 400, 50], "score": [0.5, 0.25, 2.0]}
    )
    # the 0-based half-open [100, 300) covers the 1-based positions 101..300; overlapping intervals add up
    bim = pd.DataFrame({"CHR": [1, 1, 1, 1, 1, 2, 3], "BP": [100, 101, 300, 301, 260, 50, 10]})
    np.testing.assert_array_equal(_annotate_python(bg, bim), [0.0, 0.5, 0.75, 0.25, 0.75, 2.0, 0.0])


def test_annotate_with_bedtools_adds_up_the_overlaps(monkeypatch):
    import subprocess

    import numpy as np
    import pandas as pd

    from cellink.tl.external import _sclinker_utils

    bim = pd.DataFrame({"CHR": [1, 1, 1, 1, 1], "BP": [100, 101, 300, 301, 260]})
    bg = pd.DataFrame({"chr": ["1", "1"], "start": [100, 250], "end": [300, 400], "score": [0.5, 0.25]})
    stdout = (
        "1\t100\t101\t1\t1\t100\t300\t0.5\n"
        "1\t299\t300\t2\t1\t100\t300\t0.5\n"
        "1\t299\t300\t2\t1\t250\t400\t0.25\n"
        "1\t300\t301\t3\t1\t250\t400\t0.25\n"
        "1\t259\t260\t4\t1\t100\t300\t0.5\n"
        "1\t259\t260\t4\t1\t250\t400\t0.25\n"
    )
    monkeypatch.setattr(
        _sclinker_utils.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")
    )
    expected = [0.0, 0.5, 0.75, 0.25, 0.75]
    np.testing.assert_array_equal(_sclinker_utils._annotate_with_bedtools(bg, bim), expected)
    np.testing.assert_array_equal(_sclinker_utils._annotate_python(bg, bim), expected)  # same coordinates
    monkeypatch.setattr(
        _sclinker_utils.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
    )
    np.testing.assert_array_equal(_sclinker_utils._annotate_with_bedtools(bg, bim), np.zeros(5))
