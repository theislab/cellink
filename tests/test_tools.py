from pathlib import Path

import pandas as pd
import pytest

from cellink._core.data_fields import AAnn
from cellink.io import read_sgkit_zarr
from cellink.tl import add_vep_annos_to_gdata, aggregate_annotations_for_varm, combine_annotations

# DATA = Path("tests/data_old")
DATA = Path("tests/data")


@pytest.fixture
def sample_gdata():
    # zarr_file_path = DATA / "chr22.dose.filtered.R2_0.8_test.vcz"
    zarr_file_path = DATA / "simulated_genotype_calls.vcz"
    return read_sgkit_zarr(zarr_file_path)


@pytest.fixture
def sample_vep_annos():
    vep_file = DATA / "variants_vep_annotated2.txt"
    return vep_file


def test_add_vep_annos_to_gdata(sample_gdata, sample_vep_annos):
    slot_name = f"{AAnn.name_prefix}_{AAnn.vep}"
    annotated_gdata = add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=sample_gdata, dummy_consequence=True)

    assert slot_name in annotated_gdata.uns
    assert isinstance(annotated_gdata.uns[slot_name], pd.DataFrame)

    # Check for specific columns
    expected_columns = [
        AAnn.chrom,
        AAnn.pos,
        AAnn.a0,
        AAnn.a1,
        AAnn.gene_id,
        AAnn.feature_id,
    ]
    for col in expected_columns:
        assert col in annotated_gdata.uns[slot_name].columns

    output_file = Path("annotated_gdata.h5ad")
    annotated_gdata.write_h5ad(output_file)
    assert output_file.exists()

    # Clean up the output file after the test
    output_file.unlink()
    assert not output_file.exists()


def test_combine_annotations(sample_gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=sample_gdata, dummy_consequence=True)

    combine_annotations(sample_gdata, ["vep"])
    assert AAnn.name_prefix in sample_gdata.uns


def test_aggregate_annotations_for_varm(sample_gdata, sample_vep_annos):
    add_vep_annos_to_gdata(vep_anno_file=sample_vep_annos, gdata=sample_gdata, dummy_consequence=True)

    for agg_type in ["first", "unique_list_max", "list", "str"]:
        print(agg_type)
        res = aggregate_annotations_for_varm(
            gdata=sample_gdata, annotation_key="variant_annotation_vep", agg_type=agg_type, return_data=True
        )
        id_cols = [AAnn.chrom, AAnn.pos, AAnn.a0, AAnn.a1]
        assert len(res.index.drop_duplicates()) == len(res[id_cols].drop_duplicates())


if __name__ == "__main__":
    pytest.main([__file__])
