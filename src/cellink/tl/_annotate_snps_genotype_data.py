import anndata as ad
import pandas as pd
import numpy as np
import pickle
import sgkit as sg
from pathlib import Path
import logging
import sys
from cellink.tl.utils import (
    _explode_columns,
    _get_vep_start_row,
    _add_dummy_cols,
    _flatten_single_value,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


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
    """
    Calls the VEP command line tool from within python
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
    vep_version = config["vep_version"]
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
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, shell=True
        )
        logger.info("VEP ran successfully!")
        logger.info("Output:\n", result.stdout)  # Standard output of the command
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running VEP: {e}")
        logger.error(
            f"Error output:\n{e.stderr}"
        )  # Standard error message if command fails

    if return_annos:
        annos = pd.read_csv(output, sep="\t", skiprows=_get_vep_start_row(output))
        return annos


def _change_col_dtype(annos):
    cols_to_replace = list(annos.dtypes[annos.dtypes == "object"].index)
    logger.info(f"Changing dtype of categorical columns {cols_to_replace}")
    for col in cols_to_replace:
        try:
            annos[col] = annos[col].astype(float)
        except:
            try:
                annos[col] = annos[col].replace(np.nan, "-")
            except:
                logger.warning(f"{col} couldn't be changed")
    return annos


def add_vep_annos_to_gdata(
    vep_anno_file,
    gdata,
    id_col="#Uploaded_variation",
    id_col_new="snp_id",
    cols_to_explode=["Consequence"],
    cols_to_dummy=["Consequence"]):
    """Add VEP annotations to gdata 

    Parameters
    ----------
    vep_anno_file : _type_
        File with vep output (as written by run_vep)
    gdata : gdata
        gdata object the annotations should be written to
    id_col : str, optional
        Variant id column used in the vep_anno_file, by default "#Uploaded_variation"
    id_col_new : str, optional
        _description_, by default "snp_id"
    cols_to_explode : list, optional
        Columns which can have multiple comma-separated values per element which should be exploded, by default ["Consequence"]


    Returns
    -------
    gdata
        gdata object with VEP annotations as varm["annotation_0"]-varm["annotation_n"]. 
        Multiple anotation dimensions (0-n) results from multiple contexts different variants can be in (e.g., different effects in overlapping transcripts/gene.)
        N is the maximum number of contexts any variant has. Dimensions >annotation_0 so for simple downstream analyses it's advised to use varm["annotation_0"].
    """    

    # TODO: rename annotation columns
    annos = pd.read_csv(
        vep_anno_file, sep="\t", skiprows=_get_vep_start_row(vep_anno_file)
    )
    
    logger.info(f"renaming id column {id_col} into {id_col_new}")
    annos[id_col] = annos[id_col].str.replace("/", "_")
    annos = annos.rename(columns={id_col: id_col_new})
    id_col = id_col_new

    logger.info("Subsetting annotations to variants that are in gdata")
    annos = annos.set_index(id_col).loc[gdata.var.index].reset_index()
    logger.info(f"{annos.columns}")
    annos.replace("-", np.nan, inplace=True)
    for col in cols_to_explode:
        annos = _explode_columns(annos, col)
    for col in cols_to_dummy:
        annos = _add_dummy_cols(annos, col)

    annos = _change_col_dtype(annos)
    # change dtypes such that they can be written

    logger.info("Expanding annotations from multiple contexts per variant")
    varm = _create_varm_expanded(
        annos, gdata, id_col, key_prefix="annotations_"
    )

    gdata.varm = varm

    return gdata


def _expand_annotations(
    data, id_col, cols_to_exp, max_n_val, key_prefix="annotations_"
):
    """Expand annotations with multiple context per variant into mulitple data frames

    Parameters
    ----------
    data : Pandas.DataFrame
        annotation data frame
    id_col : str
        Variant id col
    cols_to_exp : list
        Columns to expand into multiple dimensions of the annotations_0 - n e.g., Gene id and the  corresponding effect column
    max_n_val : _type_
        Maximum number of context any variant has (i.e., n)
    key_prefix : str, optional
        prefix of resulting annotation data frames, by default "annotations_"

    Returns
    -------
    dict
        dictionary of annotation data frames 
    """    
    data_exp = data.groupby(id_col).agg(lambda x: list(x)).reset_index()
    obj_cols =list(data.dtypes[(data.dtypes == "category") | (data.dtypes == "object")].index)
    obj_cols = list(set(obj_cols).intersection(set(cols_to_exp)))
    fill_dict = {col: "-" if col in obj_cols else 0 for col in cols_to_exp}

    # Iterate over each row in agg_res to populate the new lists for DataFrames
    col_dict = {i: {col: [] for col in cols_to_exp} for i in range(max_n_val)}
    for index, row in data_exp.iterrows():
        # Get the list of values
        for col in cols_to_exp:
            values_flat = row[col]
            while len(values_flat) < max_n_val:
                values_flat.append(fill_dict[col])  # Fill with zeros
            for i in range(max_n_val):
                col_dict[i][col].append(values_flat[i])

    for i in range(max_n_val):
        col_dict[i][id_col] = data_exp[id_col]
    df_dict = {
        f"{key_prefix}{i}": pd.DataFrame(col_dict[i]).set_index(id_col)
        for i in range(max_n_val)
    }
    return df_dict


def _create_varm_expanded(
    annos, gdata, id_col, key_prefix="annotations_"
):
    """Expand annotations with multiple contexts per variant into multiple annotation data frames

    Parameters
    ----------
    annos : Pd.DataFrame
        Annotations
    gdata : gdata
        gdata 
    id_col : str
        Variant id col
    key_prefix : str, optional
        prefix of resulting annotation data frames, by default "annotations_"


    Returns
    -------
    dict
        varm dictionary of max_n_val annotation data frames  
    """    

    logger.info("getting unique counts")
    unique_counts = (
        annos.groupby(id_col).agg(lambda x: x.nunique(dropna=False)).max(axis=0)
    )

    cols_to_expand = unique_counts[unique_counts > 1].index
    cols_to_keep = list(set(annos.columns) - set(cols_to_expand))
    print(f"Columns with multiple values per variant: {cols_to_expand}")

    max_n_vals = unique_counts.max()
    logger.info(f"Maximum number of distinct annotations per variant {max_n_vals}")

    annos_exp = _expand_annotations(
        annos[[id_col, *cols_to_expand]].copy(), id_col, cols_to_expand, max_n_vals
    )

    logger.info("merging non-expanded annos into first data frame")
    anno_0 = annos_exp[f"{key_prefix}0"]
    assert len(annos[cols_to_keep].drop_duplicates()) == len(anno_0)

    anno_0_merged = (
        annos[cols_to_keep]
        .drop_duplicates()
        .set_index(id_col)
        .join(anno_0, how="left", validate="1:1")
    )
    annos_exp[f"{key_prefix}0"] = anno_0_merged
    id_order = gdata.var.index

    varm = {key: val.loc[id_order] for key, val in annos_exp.items()}

    return varm