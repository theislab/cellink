import anndata as ad
import pandas as pd
import numpy as np
import pickle
import sgkit as sg
from pathlib import Path
import logging 


def setup_snpeff():
    return 0

def run_annotation_with_snpeff():
    return 0

def setup_ensemblvep():
    return 0

def run_annotation_with_ensemblvep():
    return 0

def setup_favor():
    return 0

def run_annotation_with_favor():
    return 0

def _write_variants_to_vcf(variants, out_file):
    #TODO add check for if file allready exists
    with open(out_file, "w") as f:
        # Write the VCF header
        f.write("##fileformat=VCFv4.0\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write each variant without ID, QUAL, FILTER, or INFO
        for row in variants:
            logger.info(row)
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