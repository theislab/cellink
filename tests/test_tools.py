from pathlib import Path

import pandas as pd
import pytest

from cellink.io import read_sgkit_zarr
from cellink.tl import add_vep_annos_to_gdata

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


def test_add_vep_annos_to_gdata(sample_gdata, sample_vep_annos, tmp_path):
    annotated_gdata = add_vep_annos_to_gdata(
        str(sample_vep_annos),
        sample_gdata,
        id_col="#Uploaded_variation",
        cols_to_explode=["Consequence"],
        cols_to_dummy=["Consequence"],
    )

    # Check if annotations were added
    assert "annotations_0" in annotated_gdata.varm
    assert isinstance(annotated_gdata.varm["annotations_0"], pd.DataFrame)

    # Check for specific columns
    expected_columns = [
        "Allele",
        "gnomADe_OTH_AF",
        "Consequence_stop_gained",
        "Feature_type",
        "gnomADe_AMR_AF",
        "CLIN_SIG",
        "gnomADe_AF",
        "FLAGS",
        "Location",
        "gnomADe_EAS_AF",
        "CADD_RAW",
        "gnomADe_ASJ_AF",
        "PHENO",
        "CADD_PHRED",
        "Consequence_intergenic_variant",
        "gnomADe_FIN_AF",
        "SOMATIC",
        "gnomADe_NFE_AF",
        "gnomADe_AFR_AF",
        "Existing_variation",
        "gnomADe_SAS_AF",
        "Gene",
        "Feature",
        "cDNA_position",
        "CDS_position",
        "Protein_position",
        "Amino_acids",
        "Codons",
        "IMPACT",
        "DISTANCE",
        "STRAND",
        "BIOTYPE",
        "CANONICAL",
        "ENSP",
        "SIFT",
        "PolyPhen",
        "TSSDistance",
        "Consequence_3_prime_UTR_variant",
        "Consequence_5_prime_UTR_variant",
        "Consequence_downstream_gene_variant",
        "Consequence_intron_variant",
        "Consequence_missense_variant",
        "Consequence_non_coding_transcript_exon_variant",
        "Consequence_non_coding_transcript_variant",
        "Consequence_splice_polypyrimidine_tract_variant",
        "Consequence_splice_region_variant",
        "Consequence_synonymous_variant",
        "Consequence_upstream_gene_variant",
    ]
    for col in expected_columns:
        assert col in annotated_gdata.varm["annotations_0"].columns

    # Check IMPACT values
    impact_values = annotated_gdata.varm["annotations_0"]["IMPACT"].unique()
    assert set(impact_values).issubset({"MODIFIER", "LOW", "MODERATE", "HIGH"})
    assert isinstance(annotated_gdata.varm["annotations_0"], pd.DataFrame)
    # Save annotated gdata
    output_file = tmp_path / "annotated_gdata.h5ad"
    annotated_gdata.write_h5ad(output_file)
    assert output_file.exists()


def test_consequence_types(sample_gdata, sample_vep_annos):
    annotated_gdata = add_vep_annos_to_gdata(
        str(sample_vep_annos),
        sample_gdata,
        id_col="#Uploaded_variation",
        cols_to_explode=["Consequence"],
        cols_to_dummy=["Consequence"],
    )

    # Check for specific consequence types
    expected_types = ["intron_variant", "missense_variant", "synonymous_variant"]
    for exp_type in expected_types:
        column_name = f"Consequence_{exp_type}"
        assert column_name in annotated_gdata.varm["annotations_0"].columns
        assert annotated_gdata.varm["annotations_0"][column_name].sum() > 0


if __name__ == "__main__":
    pytest.main([__file__])
