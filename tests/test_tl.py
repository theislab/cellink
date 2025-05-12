import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellink._core.data_fields import AAnn
from cellink._core.dummy_data import sim_gdata
from cellink.io import read_sgkit_zarr, write_variants_to_vcf
from cellink.tl import (
    add_vep_annos_to_gdata,
    aggregate_annotations_for_varm,
    combine_annotations,
    dosage_per_strand,
    one_hot_encode_genotypes,
    run_vep,
)

DATA = Path("tests/data")
CONFIG = Path("docs/configs")


def test_one_hot_encode_genotypes():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    one_hot_encode_gdata = one_hot_encode_genotypes(gdata[:100, :100])
    one_hot_encode_gdata_load = np.load(DATA / "simulated_genotype_calls_one_hot_encoded.npy")
    assert np.allclose(one_hot_encode_gdata, one_hot_encode_gdata_load)


def test_dosage_per_strand():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata.X = gdata.X.compute()
    dosage_per_strand_gdata = dosage_per_strand(gdata[:100, :100])
    dosage_per_strand_gdata_load = np.load(DATA / "simulated_genotype_calls_dosage_per_strand.npy")
    assert np.allclose(dosage_per_strand_gdata, dosage_per_strand_gdata_load)

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
    run_vep(CONFIG / "vep_config.yaml", "variants.vcf", "variants_annotated.txt")
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
