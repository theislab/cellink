import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cellink.tl.utils import (
    _add_dummy_cols,
    _explode_columns,
    _flatten_single_value,
    _get_vep_start_row,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

def setup_snpeff():

    os.makedirs("deps", exist_ok=True)
    subprocess.run(["wget", "http://sourceforge.net/projects/snpeff/files/snpEff_latest_core.zip", "-O", "deps/snpEff_latest_core.zip"], check=True)
    os.chdir("deps")
    subprocess.run(["unzip", "snpEff_latest_core.zip"], check=True)
    os.chdir("../")

def load_snpeff():
    return 0

def run_annotation_with_snpeff():

    snpeff_path = "./deps/snpEff/snpEff.jar"

    chromosome = "1"
    vcf_input = f"/sc-projects/sc-proj-dh-ukb-intergenics/raw_data/genotype_single_cell/yazar_powell/genotype/filter_vcf_r08/chr{chromosome}.dose.filtered.R2_0.8.vcf"
    vcf_output = f"/sc-projects/sc-proj-dh-ukb-intergenics/raw_data/genotype_single_cell/yazar_powell/genotype/filter_vcf_r08/chr{chromosome}.dose.filtered.R2_0.8.ann.vcf"

    subprocess.run(["java", "-Xmx8g", "-jar", snpeff_path, "GRCh37.75", vcf_input], stdout=open(vcf_output, 'w'), check=True)

    return 0

def setup_favor():

    subprocess.run(["wget", "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/1VGTJI/17fe155b1d0-76967428f313?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27chr22.tar.gz&response-content-type=application%2Fx-gzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241008T072613Z&X-Amz-SignedHeaders=host&X-Amz-Expires=7200&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20241008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2d9e2bb3962a7c0d14534edc0758015f355683a52bc3303679df5241b2e76f87", "-O", "favor_chr22.zip"], check=True)

def run_annotation_with_favor():

    df = pl.DataFrame(
        data = {
            "rsid": ["rs0", "rs1", "rs2"],
            "annotation": [1241, 1242, 1251]
        }
    )

    annotation = pl.read_csv("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/theis/favor/n/holystore01/LABS/xlin/Lab/xihao_zilin/FAVORDB/chr22_1.csv")

    import polars as pl

    import datetime as dt

    df = pl.DataFrame(
        {
            "name": ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
            "birthdate": [
                dt.date(1997, 1, 10),
                dt.date(1985, 2, 15),
                dt.date(1983, 3, 22),
                dt.date(1981, 4, 30),
            ],
            "weight": [57.9, 72.5, 53.6, 83.1],  # (kg)
            "height": [1.56, 1.77, 1.65, 1.75],  # (m)
        }
    )

    df2 = pl.DataFrame(
        {
            "name": ["Ben Brown", "Daniel Donovan", "Alice Archer", "Chloe Cooper"],
            "parent": [True, False, False, False],
            "siblings": [1, 2, 3, 4],
        }
    )

    print(df.join(df2, on="name", how="left"))

    "Rscript convertVCFtoGDS.r chrnumber"
    "Rscript FAVORannotatorv2aGDS.r chrnumber"

    "Rscript /sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/theis/favor/FAVORannotator/Scripts/CSV/convertVCFtoGDS.r /sc-projects/sc-proj-dh-ukb-intergenics/raw_data/genotype_single_cell/yazar_powell/genotype/filter_vcf_r08/chr22.dose.filtered.R2_0.8.vcf.gz"

    #https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/1VGTJI/17fdc14a662-18dc7770dec1?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27Annotator22.sql.gz&response-content-type=application%2Fgzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241008T130642Z&X-Amz-SignedHeaders=host&X-Amz-Expires=7199&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20241008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e3b427d85584e310f8445097fe34cc18ee85c4ba46e46a8f8c6d9a53127aa20e

    return 0


###############################################

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        logger.info("VEP ran successfully!")
        logger.info("Output:\n", result.stdout)  # Standard output of the command
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running VEP: {e}")
        logger.error(f"Error output:\n{e.stderr}")  # Standard error message if command fails

    if return_annos:
        annos = pd.read_csv(output, sep="\t", skiprows=58)
        return annos


def _aggregate_dup_rows(annos, id_col="#Uploaded_variation"):
    annos_cond = annos.copy()
    annos_cond = annos_cond.set_index(id_col)
    indices_to_agg = list(annos_cond.index.to_frame()[annos_cond.index.value_counts() > 1].index.unique())

    logger.info(f"Number of variant ids with >1 annotation {len(indices_to_agg)}")

    annos_cond_sub = annos_cond.loc[indices_to_agg].copy()

    annos_cond_sub = annos_cond_sub.groupby(id_col).agg(lambda x: list(x))
    annos_cond_sub = annos_cond_sub.applymap(lambda x: _flatten_single_value(x) if isinstance(x, list) else x)
    # TODO fix  FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
    indices_to_keep = list(set(annos_cond.index) - set(indices_to_agg))
    annos_cond = pd.concat([annos_cond.loc[indices_to_keep], annos_cond_sub])

    assert len(annos_cond) == len(annos[id_col].unique())
    return annos_cond


def read_vep_annos(vep_anno_file, cols_to_explode=["Consequence"], cols_to_dummy=["Consequence"]):
    # TODO: rename annotation columns
    annos = pd.read_csv(vep_anno_file, sep="\t", skiprows=_get_vep_start_row(vep_anno_file))
    logger.info(f"{annos.columns}")
    annos.replace("-", np.nan, inplace=True)
    for col in cols_to_explode:
        annos = _explode_columns(annos, col)
    for col in cols_to_dummy:
        annos = _add_dummy_cols(annos, col)
    # TODO: make function to collapse such that only one row per variant
    logger.info("Aggregating annotations from multiple contexts to get one row per variant")
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
