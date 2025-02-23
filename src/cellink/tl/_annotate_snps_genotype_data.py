import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cellink._core.data_fields import VAnn
from cellink.tl.utils import (
    _get_vep_start_row,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def setup_snpeff():
    """
    Downloads and sets up the SnpEff tool in the 'deps' directory.

    This function creates a directory called 'deps', downloads the latest version
    of SnpEff from SourceForge, and extracts the contents. It ensures that the
    required tool is available for genome annotation tasks.
    """
    os.makedirs("deps", exist_ok=True)
    subprocess.run(
        [
            "wget",
            "http://sourceforge.net/projects/snpeff/files/snpEff_latest_core.zip",
            "-O",
            "deps/snpEff_latest_core.zip",
        ],
        check=True,
    )
    os.chdir("deps")
    subprocess.run(["unzip", "snpEff_latest_core.zip"], check=True)
    os.chdir("../")



def run_annotation_with_snpeff(vcf_input: str, vcf_output: str, genome: str = "GRCh37.75"):
    """
    Runs genome annotation using the SnpEff tool.

    This function uses the SnpEff tool to annotate a given VCF file with the specified
    genome database. The annotated output is saved to the specified output file.

    Args:
        vcf_input (str): Path to the input VCF file to be annotated.
        vcf_output (str): Path to the output file where the annotated VCF will be saved.
        genome (str): Genome version to be used for annotation (default: "GRCh37.75").
                      Ensure this genome is supported by SnpEff.
    """
    snpeff_path = "./deps/snpEff/snpEff.jar"

    subprocess.run(
        ["java", "-Xmx8g", "-jar", snpeff_path, genome, vcf_input],
        stdout=open(vcf_output, "w"),
        check=True,
    )



def _write_variants_to_vcf(variants, out_file):
    # TODO add check for if file allready exists
    logger.info(f"Writing variants to {out_file}")
    with open(out_file, "w") as f:
        # Write the VCF header
        f.write("##fileformat=VCFv4.0\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write each variant without ID, QUAL, FILTER, or INFO
        for row in variants:
            chrom, pos, ref, alt = row.split("_")
            vcf_row = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n"
            f.write(vcf_row)


def write_variants_to_vcf(gdata, out_file="variants.vcf"):
    """Write unique variants from gdata to vcf file for annotation

    Parameters
    ----------
    gdata : gdata
        gdata object
    out_file : str, optional
        output file. By default "variants.vcf"
    """
    all_variants = list(gdata.var.index)
    logger.info(f"number of variants to annotate: {len(all_variants)}")
    # TODO allow subsetting of variants
    _write_variants_to_vcf(all_variants, out_file)


def run_vep(
    config_file,
    input_vcf="variants.vcf",
    output="variant_vep_annotated.txt",
    return_annos=False,
):
    """Calls the VEP command line tool from within python

    Requires VEP to be installed (Tested with VEP v108)

    Parameters
    ----------
    config_file : _type_
        config file specifying VEP paths as in example/config.yaml
    input_vcf : str, optional
        VCF with variants to annotate. By default "variants.vcf"
    output : str, optional
       File where VEP writes the annotated variants by default "variant_vep_annotated.txt"
    return_annos : bool, optional
        Should the written annotations be loaded into memory.by default False

    Returns
    -------
    None if return_annos=False else the written annotations loaded into a Pandas Data Frame
    """
    # TODO: make VEP options more modular
    logger.info("using config {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    input_fasta = config["input_fasta"]
    vep_dir = config["vep_dir"]
    vep_plugin_dir = config.get("vep_plugin_dir", f"{vep_dir}/Plugins/")
    vep_nfork = config.get("vep_nfork", 5)  # TODO

    genome_assembly = config["genome_assembly"]
    cadd_dir = config.get("cadd_dir", None)

    if cadd_dir is not None:
        cadd_dir = Path(cadd_dir) / genome_assembly
        CADD_WGS_SNV = f"{cadd_dir}/whole_genome_SNVs.tsv.gz"
        CADD_INDEL = {
            "GRCh37": f"{cadd_dir}/InDels.tsv.gz",
            "GRCh38": f"{cadd_dir}/gnomad.genomes.r3.0.indel.tsv.gz",  # TODO check if this was just sth cluster specific
        }[genome_assembly]
        cadd_cmd = f"--plugin CADD,{CADD_WGS_SNV},{CADD_INDEL}"
    else:
        cadd_cmd = ""
    params_af = "--af_gnomade"
    params_offline = "--offline"
    params_cache = "--cache"
    params_dir_cache = f"--dir_cache {vep_dir}"
    species = "homo_sapiens"
    vep_input_format = "vcf"

    cmd = " ".join(
        [
            "vep",
            "--input_file",
            f"{input_vcf}",
            "--output_file",
            f"{output}",
            "--species",
            str(species),
            "--assembly",
            str(genome_assembly),
            "--format",
            str(vep_input_format),
            f"{params_af}",
            f"{params_offline}",
            f"{params_cache}",
            f"{params_dir_cache}",
            "--dir_plugins",
            str(vep_plugin_dir),
            "--fork",
            str(vep_nfork),
            "--fasta",
            f"{input_fasta}",
            "--tab",
            "--total_length",
            "--no_escape",
            "--polyphen s",
            "--sift s",
            "--canonical",
            "--protein",
            "--biotype",
            "--force_overwrite",
            "--no_stats",
            "--per_gene",
            "--check_existing",
            f"{cadd_cmd}",
            "--plugin TSSDistance",
            "--pick_order biotype,canonical,appris,tsl,ccds,rank,length,ensembl,refseq",
        ]
    )

    logger.info(f"running VEP command {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        logger.info("VEP ran successfully!")
        logger.info("Output:\n", result.stdout)  # Standard output of the command
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running VEP: {e}")
        logger.error(f"Error output:\n{e.stderr}")  # Standard error message if command fails

    if return_annos:
        annos = pd.read_csv(output, sep="\t", skiprows=_get_vep_start_row(output))
        return annos


def _change_col_dtype(annos):
    cols_to_replace = list(annos.dtypes[annos.dtypes == "object"].index)
    logger.info(f"Changing dtype of categorical columns {cols_to_replace}")
    for col in cols_to_replace:
        try:
            annos[col] = annos[col].astype(float)
        except:  # TODO: please catch explicit exception
            try:
                annos[col] = annos[col].replace(np.nan, "-")
            except:  # TODO: please catch explicit exception
                logger.warning(f"{col} couldn't be changed")
    return annos


def _prep_vep_annos(
    vep_anno_file: str,
    gdata,
    id_col_vep: str = "#Uploaded_variation",
    cols_to_drop: list = ["Allele", "Feature_type", "Location"],
    dummy_consequence: bool = True,
) -> pd.DataFrame:
    """
    Prepare VEP annotations by processing and formatting data from the input file.

    Parameters
    ----------
    vep_anno_file : str
        Path to the VEP annotation file (e.g., output from run_vep).
    gdata : object
        Object containing genomic data with `var.index` as the list of variants to subset.
    id_col_vep : str, optional
        Column name in the VEP file used for variant IDs, by default "#Uploaded_variation".
    cols_to_drop : list, optional
        List of columns to remove from the final annotations, by default ["Allele", "Feature_type", "Location"].
    dummy_consequence : bool, optional
        If True, one-hot encodes the "Consequence" column into multiple binary columns, by default True.

    Returns
    -------
    pd.DataFrame
        A processed DataFrame with formatted VEP annotations.
    """
    # Define the required identifier columns
    id_cols = [VAnn.chrom, VAnn.pos, VAnn.a0, VAnn.a1]
    unique_identifier_cols = [*id_cols, AAnn.gene_id, AAnn.feature_id]

    # Read VEP annotation file
    logger.info(f"Reading annotation file {vep_anno_file}")
    annos = pd.read_csv(vep_anno_file, sep="\t", skiprows=_get_vep_start_row(vep_anno_file))

    logger.info("Annotation file loaded")

    annos[id_col_vep] = annos[id_col_vep].str.replace("/", "_")
    annos = annos.rename(
        columns={
            id_col_vep: AAnn.index,
            "Gene": AAnn.gene_id,
            "Feature": AAnn.feature_id,
        }
    )

    logger.info(f"Annotation columns: {list(annos.columns)}")

    # subset to make processing faster
    vars_to_keep = list(set(annos[AAnn.index]).intersection(gdata.var.index))
    annos = annos.set_index(AAnn.index).loc[vars_to_keep].reset_index()

    # Replace missing values and split the variant identifier into individual columns
    annos.replace("-", np.nan, inplace=True)
    annos[id_cols] = annos[AAnn.index].str.split("_", expand=True)
    annos = annos.set_index(AAnn.index)

    # Verify that unique identifier cols are chosen correctly
    assert len(annos[unique_identifier_cols].drop_duplicates()) == len(annos)
    # Determine columns to keep
    columns_to_keep = set(unique_identifier_cols).union(set(annos.columns)) - set(cols_to_drop)
    anno_cols = set(columns_to_keep) - set(unique_identifier_cols)

    # Define column order
    col_order = [*unique_identifier_cols, *anno_cols]

    # Optionally, create dummy variables for the "Consequence" column
    if dummy_consequence:
        dummies = annos["Consequence"].str.get_dummies(",").add_prefix("Consequence_")
        dummy_cols = dummies.columns
        anno_cols = anno_cols - {"Consequence"}
        col_order = [*unique_identifier_cols, *dummy_cols, *anno_cols]
        annos[dummy_cols] = dummies
    else:
        col_order = [*unique_identifier_cols, *anno_cols]

    # Reorder columns and return processed DataFrame
    annos = annos[col_order]

    # change dtyps to numeric where possible
    annos = _change_col_dtype(annos)
    return annos


def add_vep_annos_to_gdata(
    vep_anno_file: str,
    gdata,
    id_col_vep: str = "#Uploaded_variation",
    cols_to_drop: list = ["Allele", "Feature_type", "Location"],
    dummy_consequence: bool = True,
) -> object:
    """
    Add VEP annotations to a gdata object.

    Parameters
    ----------
    vep_anno_file : str
        Path to the VEP annotation file (e.g., output from run_vep).
    gdata : object
        Object representing genomic data to which the annotations will be added.
    id_col_vep : str, optional
        Column name in the VEP file used for variant IDs, by default "#Uploaded_variation".
    cols_to_drop : list, optional
        List of columns to drop from the annotations, by default ["Allele", "Feature_type", "Location"].
    dummy_consequence : bool, optional
        If True, one-hot encodes the "Consequence" column into multiple binary columns, by default True.

    Returns
    -------
    object
        The updated gdata object with VEP annotations added as `uns["annotation_vep"]`.
    """
    logger.info("Preparing VEP annotations for addition to gdata")
    # Process the VEP annotations
    vep_data = _prep_vep_annos(vep_anno_file, gdata, id_col_vep, cols_to_drop, dummy_consequence)
    # Subset annotations to match the variants in gdata
    vep_data = pd.DataFrame(index=gdata.var.index).join(vep_data, how="left")
    missing_vars = nan_rows = vep_data[vep_data.isna().all(axis=1)]
    if len(missing_vars) > 0:
        logger.warning(f"VEP annotation missing for {len(missing_vars)}")
    # Add processed annotations to gdata
    gdata.uns[f"{AAnn.name_prefix}_{AAnn.vep}"] = vep_data
    logger.info("VEP annotations added to gdata")
    return gdata


def combine_annotations(
    gdata,
    keys: list = ["vep"],
    unique_identifier_cols=[
        VAnn.chrom,
        VAnn.pos,
        VAnn.a0,
        VAnn.a1,
        AAnn.gene_id,
        AAnn.feature_id,
    ],
):
    """
    Combine multiple annotation datasets into a single unified dataset.

    Parameters
    ----------
    gdata : object
        The genomic data object containing annotations stored in `uns` under specific keys.
    keys : list, optional
        List of annotation keys to combine, by default ["vep"].
        These keys correspond to the annotations stored in `gdata.uns`, with prefixes like `variant_annotation_key}`.
    unique_identifier_cols : list, optional
        List of columns that uniquely identify a variant-context pair, by default
        [VAnn.chrom, VAnn.pos, VAnn.a0, VAnn.a1, AAnn.gene_id, AAnn.feature_id].

    Returns
    -------
    None
        Modifies the `gdata` object in place by adding the combined annotations under the key `variant_annotation`.

    Notes
    -----
    - The function ensures that all unique identifier columns are present in each annotation set.
    - Performs an outer join across all specified annotations based on unique identifiers.
    - Verifies that no annotation columns are duplicated in the resulting dataset.
    - Verifies that the number of unique variant-context combinations remains consistent after merging.

    Raises
    ------
    AssertionError
        If any of the following checks fail:
        - The provided `keys` are a subset of the allowed annotation sources (currently only "vep").
        - Unique identifier columns are present in each annotation dataset.
        - No duplicate annotation columns exist in the combined dataset.
        - The number of unique variant-context combinations is consistent post-merge.

    Examples
    --------
    >>> combine_annotations(gdata, keys=["vep"])
    >>> print(gdata["variant_annotation"])
    # Outputs the combined annotations stored in gdata under the `variant_annotation` key.
    """
    logger.warning("Function still under development until it can be tested with other annotations")

    allowed_keys = [AAnn.vep]  # update once snpeff and favor are implemented as well
    assert set(keys).issubset(allowed_keys)

    combined_annotation_cols = []
    i = 0
    for key in keys:
        this_annotations = gdata.uns[f"{AAnn.name_prefix}_{key}"]
        columns = this_annotations.columns

        # check that identifer columns are present
        assert set(unique_identifier_cols).issubset(columns), f"Missing unique identifiers in {key} annotations."

        annotation_cols = set(columns) - set(unique_identifier_cols)
        combined_annotation_cols = combined_annotation_cols + list(annotation_cols)

        # merge annotations (outer join)
        if i == 0:
            combined_annotations = this_annotations
            unique_contexts = this_annotations[unique_identifier_cols]
        else:
            combined_annotations = combined_annotations.merge(this_annotations, on=unique_identifier_cols, how="outer")
            unique_contexts = pd.concat([unique_contexts, this_annotations[unique_identifier_cols]])
        i += 1
    # check that no annotation columns are duplicated in the data frames
    assert len(set(combined_annotation_cols)) == len(combined_annotation_cols), "Duplicate annotation columns detected."

    # check that number of unique variant-context in combined object is correct
    assert len(combined_annotations) == len(
        unique_contexts.drop_duplicates()
    ), "Mismatch in unique variant-context combinations after merging."

    gdata.uns[AAnn.name_prefix] = combined_annotations


def custom_agg(x, agg_type):
    if agg_type == "unique_list_max":
        if x.dtype == "object":  # Check if column is of string type
            return ",".join([str(i) for i in x.unique()])  # Unique values as comma-separated string
        elif pd.api.types.is_numeric_dtype(x):  # Check if column is numeric
            return x.max()  # Aggregate using max value
        else:
            return ",".join([str(i) for i in x.unique()])  # Fallback for other types
    elif agg_type == "list":
        return list(x)  # Simply aggregate into a list
    elif agg_type == "str":
        return ",".join([str(i) for i in x])  # Aggregate into a comma-separated string
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


def aggregate_annotations_for_varm(
    gdata,
    annotation_key,
    agg_type: str = "unique_list_max",
    return_data: bool = False,
):
    """
    Aggregates a DataFrame containing variant annotations based on the specified aggregation type such
    that there is only row per variant id. This means that annotations are aggregated across different gene/transcript contexts

    Parameters
    ----------
    gdata : object
        The genomic data object containing annotations stored in `uns` under specific keys.
    annotation_key : str
        Key to access the annotations within `gdata.uns`. The annotations are expected to be stored as a pandas DataFrame.
    agg_type : str, optional
        Aggregation type to determine how annotation values are combined. Options are:
            - "unique_list_max": Unique string values are aggregated into a comma-separated string,
                                 and numeric columns are aggregated by their maximum value.
            - "list": Aggregates all values into a list, preserving duplicates.
            - "str": Aggregates all values into a single comma-separated string.
            - "first": Drops duplicates and keeps only the first occurrence for each variant-context pair.
        Default is "unique_list_max".
    return_data : bool, optional
        If True, the aggregated DataFrame is returned in addition to modifying the `gdata` object.
        Default is False.

    Returns
    -------
    pd.DataFrame, optional
        The aggregated DataFrame is returned if `return_data` is True. Otherwise, the function writes the
        aggregated annotations to gdata.varm["variant_annotation"].

    Examples
    --------
    >>> aggregate_annotations(gdata, "variant_annotation_vep",
                            agg_type = "unique_list_max",
                            debug = True)

    """
    # Extract DataFrame
    anno_df = gdata.uns[annotation_key]

    # Validate aggregation type
    allowed_agg_types = ["first", "unique_list_max", "list", "str"]
    if agg_type not in allowed_agg_types:
        raise ValueError(f"Invalid agg_type '{agg_type}'. Allowed types are: {allowed_agg_types}")

    logger.info(f"Aggregating using method: {agg_type}")

    # Handle "first" agg_type
    if agg_type == "first":
        aggregated_df = anno_df.reset_index().drop_duplicates(subset=AAnn.index, keep="first").set_index(AAnn.index)

    else:
        # Identify columns to aggregate
        col_order = anno_df.columns
        cols_with_multiple_values = anno_df.reset_index().groupby(AAnn.index).nunique().max().loc[lambda x: x > 1].index
        logger.info(f"Columns to aggregate: {list(cols_with_multiple_values)}")

        # Aggregate columns with differing values
        aggregated_df = (
            anno_df[cols_with_multiple_values]
            .reset_index()
            .groupby(AAnn.index)
            .agg(lambda x: custom_agg(x, agg_type=agg_type))
        )

        # Keep columns that do not require aggregation
        cols_to_keep = anno_df.columns.difference(cols_with_multiple_values)
        unique_cols_df = anno_df[cols_to_keep].drop_duplicates()

        # Validate and merge results
        if len(unique_cols_df) != len(aggregated_df):
            raise ValueError("Mismatch in row counts after aggregation. Check your input data.")

        aggregated_df = aggregated_df.join(unique_cols_df, how="left", validate="1:1").reindex(columns=col_order)

    aggregated_df = aggregated_df.loc[gdata.var.index]
    gdata.varm[AAnn.name_prefix] = aggregated_df

    if return_data:
        return aggregated_df
