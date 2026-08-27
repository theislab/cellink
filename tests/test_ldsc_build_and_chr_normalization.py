from __future__ import annotations

import pandas as pd

from cellink.tl.external._ldsc import (
    _compute_continuous_annot_for_bimfile,
    _normalize_chr_label,
    make_annot_from_bimfile,
)


def test_genome_build_mismatch_raises_when_both_declared(tmp_path):
    bim = tmp_path / "1000G.QC.21.bim"
    bim.write_text("21\trs1\t0\t100\tA\tG\n")
    gene_coord = tmp_path / "genes.txt"
    gene_coord.write_text("GENE\tCHR\tSTART\tEND\nSOD1\t21\t50\t150\n")

    try:
        make_annot_from_bimfile(
            bimfile=str(bim),
            annot_file=str(tmp_path / "out.annot.gz"),
            gene_set_file=None,
            scores=pd.Series({"SOD1": 1.0}),
            gene_coord_file=str(gene_coord),
            gene_coord_genome_build="GRCh38",
            bim_genome_build="GRCh37",
        )
        raise AssertionError("expected ValueError for build mismatch")
    except ValueError as e:
        assert "build" in str(e).lower()


def test_normalize_chr_label_maps_plink_sex_codes_and_prefixes():
    assert _normalize_chr_label("23") == "X"
    assert _normalize_chr_label("24") == "Y"
    assert _normalize_chr_label("26") == "MT"
    assert _normalize_chr_label("chr21") == "21"
    assert _normalize_chr_label("CHR21") == "21"
    assert _normalize_chr_label("X") == "X"


def test_continuous_annot_matches_plink_numeric_x_against_gene_coord_x(tmp_path):
    bim = tmp_path / "1000G.QC.23.bim"
    bim.write_text("23\trs1\t0\t100\tA\tG\n23\trs2\t0\t200\tA\tG\n")
    gene_coords = pd.DataFrame({"gene": ["XIST"], "chr": ["X"], "start": [50], "end": [150]})
    scores = pd.Series({"XIST": 2.0})

    result = _compute_continuous_annot_for_bimfile(str(bim), scores, gene_coords, windowsize=0)
    assert result.loc[result["SNP"] == "rs1", "ANNOT"].iloc[0] == 2.0
    assert result.loc[result["SNP"] == "rs2", "ANNOT"].iloc[0] == 0.0
