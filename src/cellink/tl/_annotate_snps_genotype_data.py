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
    """Write unique variants to vcf file for annotation

    Args:
        gdata (_type_): gdata object
        out_file (str, optional): output file. Defaults to "variants.vcf".
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
    """Run VEP
    Calls the VEP command line tool from within python
    Requires VEP to be installed
    Args:
        config_file (_type_): _description_
        input_vcf (str, optional): _description_. Defaults to "variants.vcf".
        output (str, optional): _description_. Defaults to "variant_vep_annotated.txt".
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
        annos = pd.read_csv(output, sep="\t", skiprows=58)
        return annos


def _aggregate_dup_rows(annos, id_col="#Uploaded_variation"):

    annos_cond = annos.copy()
    annos_cond = annos_cond.set_index(id_col)
    indices_to_agg = list(
        annos_cond.index.to_frame()[annos_cond.index.value_counts() > 1].index.unique()
    )

    logger.info(f"Number of variant ids with >1 annotation {len(indices_to_agg)}")

    annos_cond_sub = annos_cond.loc[indices_to_agg].copy()

    annos_cond_sub = annos_cond_sub.groupby(id_col).agg(lambda x: list(x))
    annos_cond_sub = annos_cond_sub.applymap(
        lambda x: _flatten_single_value(x) if isinstance(x, list) else x
    )
    # TODO fix  FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
    indices_to_keep = list(set(annos_cond.index) - set(indices_to_agg))
    annos_cond = pd.concat([annos_cond.loc[indices_to_keep], annos_cond_sub])

    assert len(annos_cond) == len(annos[id_col].unique())
    return annos_cond


def read_vep_annos(
    vep_anno_file, cols_to_explode=["Consequence"], cols_to_dummy=["Consequence"]
):

    # TODO: rename annotation columns
    annos = pd.read_csv(
        vep_anno_file, sep="\t", skiprows=_get_vep_start_row(vep_anno_file)
    )
    logger.info(f"{annos.columns}")
    annos.replace("-", np.nan, inplace=True)
    for col in cols_to_explode:
        annos = _explode_columns(annos, col)
    for col in cols_to_dummy:
        annos = _add_dummy_cols(annos, col)
    # TODO: make function to collapse such that only one row per variant
    logger.info(
        "Aggregating annotations from multiple contexts to get one row per variant"
    )
    annos = _aggregate_dup_rows(annos, id_col="#Uploaded_variation")
    return annos


def merge_annos_into_gdata(annos, gdata, id_col="#Uploaded_variation"):

    annos = annos.reset_index().rename(columns={id_col: "variant_id"})

    annos["variant_id"] = annos["variant_id"].str.replace("/", "_")
    annos = annos.set_index("variant_id")
    assert len(set(gdata.var.index) - set(annos.index)) == 0

    var_merged = gdata.var.copy()
    logger.info("Joining gdata.var with annos on index")
    var_merged = var_merged.join(annos, how="left", validate="1:1")
    gdata.var = var_merged
    return gdata
