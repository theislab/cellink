from __future__ import annotations


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
