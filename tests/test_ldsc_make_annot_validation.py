from __future__ import annotations

import pandas as pd
import pytest

from cellink.tl.external._ldsc import make_annot_from_bimfile


def _write_bim(path, rows):
    pd.DataFrame(rows, columns=["CHR", "SNP", "CM", "BP", "A1", "A2"]).to_csv(path, sep="\t", header=False, index=False)


def test_binary_mode_requires_gene_coord_file_for_gene_set(tmp_path):
    bim = tmp_path / "test.bim"
    _write_bim(bim, [["1", "rs1", 0.0, 500, "A", "G"]])
    gene_set = tmp_path / "set.GeneSet"
    gene_set.write_text("GENE1\n")

    with pytest.raises(ValueError, match="gene_coord_file"):
        make_annot_from_bimfile(bimfile=str(bim), annot_file=str(tmp_path / "out.annot.gz"), gene_set_file=str(gene_set))


def test_make_annot_from_bimfile_requires_some_input(tmp_path):
    bim = tmp_path / "test.bim"
    _write_bim(bim, [["1", "rs1", 0.0, 500, "A", "G"]])
    with pytest.raises(ValueError, match="scores, gene_set_file, or bed_file"):
        make_annot_from_bimfile(bimfile=str(bim), annot_file=str(tmp_path / "out.annot.gz"))
