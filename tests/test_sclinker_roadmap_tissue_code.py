from __future__ import annotations

import gzip

from cellink.tl.external._sclinker_utils import load_roadmap_links


def test_roadmap_exact_tissue_code_match_not_swallowed_by_zero_row_fallback(tmp_path):
    roadmap_file = tmp_path / "roadmap.txt.gz"
    with gzip.open(roadmap_file, "wt") as f:
        f.write("chr,start,end,TargetGene,tissuename\n")
        f.write("1,100,200,GENE1,BLD\n")
        f.write("1,300,400,GENE2,BRN\n")
        f.write("1,500,600,GENE3,BLD\n")

    result = load_roadmap_links(str(roadmap_file), tissue="BLD")
    assert len(result) == 2
    assert set(result["TargetGene"]) == {"GENE1", "GENE3"}


def test_roadmap_falls_back_to_all_rows_for_a_genuinely_absent_tissue(tmp_path, caplog):
    roadmap_file = tmp_path / "roadmap.txt.gz"
    with gzip.open(roadmap_file, "wt") as f:
        f.write("chr,start,end,TargetGene,tissuename\n")
        f.write("1,100,200,GENE1,BLD\n")
        f.write("1,300,400,GENE2,BRN\n")

    result = load_roadmap_links(str(roadmap_file), tissue="KID")
    assert len(result) == 2  # genuinely no KID rows -> documented all-rows fallback
