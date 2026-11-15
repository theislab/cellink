import shutil
from pathlib import Path

import pandas as pd
import pytest

from cellink._core.data_fields import AAnn
from cellink._core.dummy_data import sim_gdata
from cellink.io import write_variants_to_vcf
from cellink.tl import (
    add_vep_annos_to_gdata,
    aggregate_annotations_for_varm,
    combine_annotations,
    run_vep,
    subset_gene,
    subset_genomic_region,
)

DATA = Path("tests/data")


@pytest.fixture
def gdata():
    return sim_gdata()


@pytest.fixture
def sample_vep_annos():
    vep_file = DATA / "test_variants_vep_annotated.txt"
    return vep_file


def test_write_variants_to_vcf(gdata):
    write_variants_to_vcf(gdata, "variants.vcf")
    vcf_file = Path("variants.vcf")
    assert vcf_file.exists()
    vcf_file.unlink()


def _ensembl_database_reachable(port=3306, timeout=5):
    """The shipped `vep` config runs VEP online against Ensembl's public MySQL server."""
    import socket

    try:
        with socket.create_connection(("ensembldb.ensembl.org", port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    shutil.which("vep") is None or not _ensembl_database_reachable(),
    reason="VEP is not installed or the Ensembl database is not reachable",
)
def test_run_vep(gdata, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_variants_to_vcf(gdata, "variants.vcf")
    run_vep(config_file="vep", input_vcf="variants.vcf", output="variants_annotated.txt")  # config shipped with cellink
    for this_file in ["variants.vcf", "variants_annotated.txt"]:
        assert Path(this_file).exists()


def test_add_vep_annos_to_gdata(gdata, sample_vep_annos):
    slot_name = f"{AAnn.name_prefix}_{AAnn.vep}"
    annotated_gdata = add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)

    assert slot_name in annotated_gdata.uns
    assert isinstance(annotated_gdata.uns[slot_name], pd.DataFrame)

    # Check for specific columns
    expected_columns = [
        AAnn.index,
        AAnn.gene_id,
        AAnn.feature_id,
    ]
    for col in expected_columns:
        assert col in annotated_gdata.uns[slot_name].index.names


def test_combine_annotations(gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)

    combine_annotations(gdata, ["vep"])
    assert AAnn.name_prefix in gdata.uns


def test_aggregate_annotations_for_varm(gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)

    for agg_type in ["first", "unique_list_max", "list", "str"]:
        print(agg_type)
        res = aggregate_annotations_for_varm(
            gdata=gdata, annotation_key="variant_annotation_vep", agg_type=agg_type, return_data=True
        )
        id_cols = [AAnn.index]
        assert len(gdata.uns["variant_annotation_vep"].reset_index()[id_cols].drop_duplicates()) == len(res.index)


def test_get_genomic_region_nosubset(gdata):
    chrom = gdata.var["chrom"].iloc[0]
    start = gdata.var["pos"].min()
    end = gdata.var["pos"].max() + 1

    subset_gdata = subset_genomic_region(gdata, chrom, start, end)

    assert subset_gdata.shape[1] == gdata.shape[1]


def test_get_genomic_region_subset(gdata):
    chrom = gdata.var["chrom"].iloc[0]
    start = gdata.var["pos"].min() + 1
    end = gdata.var["pos"].max()

    subset_gdata = subset_genomic_region(gdata, chrom, start, end)

    assert subset_gdata.shape[1] < gdata.shape[1]


def test_get_genomic_region_quantile_subset(gdata):
    chrom = gdata.var["chrom"].iloc[0]
    start = gdata.var["pos"].quantile(0.25)
    end = gdata.var["pos"].quantile(0.75)

    subset_gdata = subset_genomic_region(gdata, chrom, start, end)
    assert subset_gdata.var["pos"].min() >= start
    assert subset_gdata.var["pos"].max() <= end
    assert subset_gdata.shape[1] < gdata.shape[1]


def test_subset_gene_no_vep(gdata):
    # expect value error if vep annotations have not been added to gdata
    with pytest.raises(ValueError):
        subset_gene(gdata, gene_id="ENSG00000141510")


def test_subset_non_present_gene(gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)

    gene_id = "ENSG00000141510"
    subset_gdata = subset_gene(gdata, gene_id)

    assert subset_gdata.shape[1] == 0


def test_subset_present_gene(gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)
    gene_id = "ENSG2"
    subset_gdata = subset_gene(gdata, gene_id)
    assert subset_gdata.shape[1] < gdata.shape[1]
    assert subset_gdata.shape[1] > 0
    assert all(subset_gdata.uns["variant_annotation_vep"].reset_index()[AAnn.gene_id] == gene_id)


def test_subset_multiple_genes(gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=gdata, dummy_consequence=True)
    gene_ids = ["ENSG1", "ENSG2", "ENSG3"]  # only ENSG1 and ENSG2 are present in the sample vep annos
    subset_gdata = subset_gene(gdata, gene_ids)
    assert subset_gdata.shape[1] < gdata.shape[1]
    assert subset_gdata.shape[1] > 0
    assert all(subset_gdata.uns["variant_annotation_vep"].reset_index()[AAnn.gene_id].isin(gene_ids))
    assert "ENSG1" in subset_gdata.uns["variant_annotation_vep"].reset_index()[AAnn.gene_id].values
    assert "ENSG2" in subset_gdata.uns["variant_annotation_vep"].reset_index()[AAnn.gene_id].values
    assert "ENSG3" not in subset_gdata.uns["variant_annotation_vep"].reset_index()[AAnn.gene_id].values
