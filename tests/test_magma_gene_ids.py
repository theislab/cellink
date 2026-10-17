from __future__ import annotations

import pandas as pd
import pytest

from cellink.tl.external._magma import (
    _resolve_entrez_to_ensembl,
    convert_gene_loc_file,
    download_magma_references,
    load_magma_config,
)

GENE_LOC_COLUMNS = ["entrez_id", "chr", "start", "end", "strand", "gene_name"]

GENE_LOC = pd.DataFrame(
    [
        (60, "7", 5566779, 5570232, "-", "ACTB"),
        (71, "17", 79476997, 79479892, "-", "ACTG1"),
        (3123, "6", 32546546, 32557613, "-", "HLA-DRB1"),
        (3125, "6", 32715000, 32730000, "-", "HLA-DRB3"),
        (999, "1", 1000, 2000, "+", "NOMATCH"),
        (500, "2", 1000, 2000, "+", "GENE500"),
        (501, "2", 3000, 4000, "+", "GENE501"),
    ],
    columns=GENE_LOC_COLUMNS,
)
PAIRS = pd.DataFrame(
    [
        ("ENSG_ACTB", "ACTB", 60),
        ("ENSG_ACTG1", "ACTG1", 60),
        ("ENSG_ACTG1", "ACTG1", 71),
        ("ENSG_DRB1", "HLA-DRB1", 3123),
        ("ENSG_DRB1", "HLA-DRB1", 3125),
        ("ENSG_DRB3", "HLA-DRB3", 3125),
        ("ENSG_X1", "FOO", 999),
        ("ENSG_X2", "BAR", 999),
        ("ENSG_AMB", "AMB", 500),
        ("ENSG_AMB", "AMB", 501),
    ],
    columns=["ensembl_gene_id", "external_gene_name", "entrezgene_id"],
)


def test_resolve_entrez_to_ensembl_breaks_ties_by_symbol():
    resolved = _resolve_entrez_to_ensembl(GENE_LOC, PAIRS)
    by_gene = resolved.groupby("gene_name")["ensembl_gene_id"].apply(sorted).to_dict()

    assert by_gene["ACTB"] == ["ENSG_ACTB"]
    assert by_gene["ACTG1"] == ["ENSG_ACTG1"]
    assert by_gene["HLA-DRB1"] == ["ENSG_DRB1"]
    assert by_gene["HLA-DRB3"] == ["ENSG_DRB3"]
    assert by_gene["NOMATCH"] == ["ENSG_X1", "ENSG_X2"]
    assert "GENE500" not in by_gene and "GENE501" not in by_gene
    assert resolved["ensembl_gene_id"].is_unique


def test_convert_gene_loc_file_writes_unique_ensembl_ids(tmp_path, monkeypatch):
    gene_loc_file = tmp_path / "NCBI37.3.gene.loc"
    GENE_LOC.to_csv(gene_loc_file, sep="\t", header=False, index=False)
    monkeypatch.setattr("cellink.tl.external._magma._query_ensembl_entrez_pairs", lambda genome_build: PAIRS)

    out = convert_gene_loc_file(gene_loc_file, tmp_path / "out.gene.loc", genome_build="GRCh37")

    converted = pd.read_csv(out, sep="\t", header=None, names=["gene_id", *GENE_LOC_COLUMNS[1:]])
    assert converted["gene_id"].is_unique
    assert converted.set_index("gene_name").loc["ACTB", "gene_id"] == "ENSG_ACTB"
    assert converted.set_index("gene_name").loc["HLA-DRB1", "start"] == 32546546


def test_convert_gene_loc_file_gene_name_does_not_query_biomart(tmp_path, monkeypatch):
    gene_loc_file = tmp_path / "NCBI37.3.gene.loc"
    GENE_LOC.to_csv(gene_loc_file, sep="\t", header=False, index=False)

    def fail(*args, **kwargs):
        raise AssertionError("BioMart should not be queried for gene_name IDs")

    monkeypatch.setattr("cellink.tl.external._magma._query_ensembl_entrez_pairs", fail)

    out = convert_gene_loc_file(gene_loc_file, tmp_path / "out.gene.loc", gene_id_type="gene_name")
    converted = pd.read_csv(out, sep="\t", header=None)
    assert list(converted[0]) == list(GENE_LOC["gene_name"])


def test_default_magma_config_ships_with_the_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_magma_config()
    assert {"GRCh37", "GRCh38"} <= set(config["gene_loc"])
    assert "EUR" in config["reference_data"]


def test_load_magma_config_prefers_config_dict():
    assert load_magma_config("does_not_exist", config_dict={"gene_loc": {}}) == {"gene_loc": {}}


def test_download_magma_references_ignores_pre_fix_converted_cache(tmp_path, monkeypatch):
    magma_dir = tmp_path / "magma_references"
    magma_dir.mkdir()
    GENE_LOC.to_csv(magma_dir / "NCBI37.3.gene.loc", sep="\t", header=False, index=False)
    (magma_dir / "GRCh37_ensembl.gene.loc").write_text("ENSG_ALT_HAPLOTYPE\t6\t1\t2\t+\tC6orf10\n")
    monkeypatch.setattr("cellink.tl.external._magma._query_ensembl_entrez_pairs", lambda genome_build: PAIRS)

    gene_loc_file, ref_prefix = download_magma_references(
        genome_build="GRCh37", gene_id_type="ensembl", data_home=tmp_path
    )

    assert gene_loc_file == magma_dir / "GRCh37_ensembl_primary_assembly.gene.loc"
    assert ref_prefix is None
    assert "ENSG_ALT_HAPLOTYPE" not in gene_loc_file.read_text()


def test_download_magma_references_rejects_unknown_build(tmp_path):
    with pytest.raises(ValueError, match="Invalid genome_build"):
        download_magma_references(genome_build="GRCh36", data_home=tmp_path)
