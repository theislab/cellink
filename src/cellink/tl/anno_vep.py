import os
import numpy as np
from anndata import AnnData
import pandas as pd
import subprocess


## Function to install VEP (low prio)

## Function to write variants to VCF


def _write_variants_to_vcf(variants, out_file):
    with open(output_file, "w") as f:
        # Write the VCF header
        f.write("##fileformat=VCFv4.0\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write each variant without ID, QUAL, FILTER, or INFO
        for row in list(all_variants["id"]):
            chrom, rest = row.split(":")
            pos, ref, alt = rest.split("_")
            vcf_row = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n"
            f.write(vcf_row)


def write_variants_to_vcf(gdata, out_file="variants.vcf"):
    """Write unique variants to vcf file for annotation

    Args:
        gdata (_type_): _description_
        out_file (str, optional): _description_. Defaults to "variants.vcf".
    """
    all_variants = gdata.var
    # TODO allow subsetting of variants
    _write_variants_to_vcf(gdata.var, out_file)


## Function to call VEP


def run_vep(config):

    input_fasta = config["input_fasta"]
    vep_dir = config["vep_dir"]
    vep_plugin_dir = cofig.get("vep_plugin_dir", f"{vep_dir}/Plugins/")
    vep_nfork = cofig.get("vep_nfork", 5)  # TODO

    genome_assembly = config["genome_assembly"]
    cadd_dir = config["cadd_dir"]
    vep_version = config["vep_version"]
    input_vcf = "variants.vcf"
    output = "variant_vep_annotated.txt"

    CADD_WGS_SNV = f"{cadd_dir}/whole_genome_SNVs.tsv.gz"
    CADD_INDEL = {
        "GRCh37": f"{cadd_dir}/InDels.tsv.gz",
        "GRCh38": f"{cadd_dir}/gnomad.genomes.r3.0.indel.tsv.gz",
    }[genome_assembly]

    params_af = "--af_gnomade"
    params_offline = "--offline"
    params_cache = "--cache"
    params_dir_cache = f"--dir_cache {vep_dir}"
    species = "homo_sapiens"
    vep_input_format = "vcf"
    cadd_cmd = f"--plugin CADD,{CADD_WGS_SNV},{CADD_INDEL}"

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
            "--check-existing" f"{cadd_cmd}",
            "--plugin TSSDistance",
            "--pick_order biotype,canonical,appris,tsl,ccds,rank,length,ensembl,refseq",
        ]
    )

    # Execute the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("VEP ran successfully!")
        print("Output:\n", result.stdout)  # Standard output of the command
    except subprocess.CalledProcessError as e:
        print(f"Error running VEP: {e}")
        print(f"Error output:\n{e.stderr}")  # Standard error message if command fails
