import shutil
from pathlib import Path

import anndata as ad
import numpy as np
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
CONFIG = Path("configs")


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


@pytest.mark.skipif(shutil.which("vep") is None, reason="VEP is not installed in the environment")
def test_run_vep(gdata):
    write_variants_to_vcf(gdata, "variants.vcf")
    run_vep(CONFIG / "vep.yaml", "variants.vcf", "variants_annotated.txt")
    for this_file in ["variants.vcf", "variants_annotated.txt"]:
        this_file = Path(this_file)
        assert Path(this_file).exists()
        this_file.unlink()


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


def _gdata_with_genes():
    var = pd.DataFrame({"chrom": ["1"] * 4, "pos": [10, 20, 30, 40]}, index=["v0", "v1", "v2", "v3"])
    gdata = ad.AnnData(X=np.zeros((2, 4)), var=var)
    gdata.uns["variant_annotation_vep"] = pd.DataFrame(
        {"gene_id": ["g1", "g1", "g2", "g3"]}, index=pd.Index(["v0", "v1", "v2", "v3"], name="snp_id")
    )
    return gdata


@pytest.mark.parametrize(
    "genes", [["g1", "g2"], ("g1", "g2"), {"g1", "g2"}, np.array(["g1", "g2"]), pd.Index(["g1", "g2"])]
)
def test_subset_gene_accepts_list_likes(genes):
    assert subset_gene(_gdata_with_genes(), genes).n_vars == 3


def test_subset_gene_single_string():
    assert list(subset_gene(_gdata_with_genes(), "g1").var_names) == ["v0", "v1"]


def test_subset_genomic_region_without_variants_returns_empty_copy():
    var = pd.DataFrame({"chrom": pd.Series([], dtype=object), "pos": pd.Series([], dtype=int)})
    out = subset_genomic_region(ad.AnnData(X=np.zeros((3, 0)), var=var), "1", 0, 100)
    assert out.n_vars == 0
    assert not out.is_view
