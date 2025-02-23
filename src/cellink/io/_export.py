import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def _generate_bim_df(gdata, chrom_col="chrom", cm_col="cm", pos_col="pos", a1_col="a1", a2_col="a2"):
    """
    Generate a BIM DataFrame from genetic data.

    Parameters
    ----------
    gdata : object
        A genetic data object with a `var` attribute containing SNP information.
    chrom_col : str, optional
        The column name in `gdata.var` representing the chromosome. Default is "chrom".
    cm_col : str, optional
        The column name in `gdata.var` representing the centimorgan position. Default is "cm".
    pos_col : str, optional
        The column name in `gdata.var` representing the base pair position. Default is "pos".
    a1_col : str, optional
        The column name in `gdata.var` representing allele 1. Default is "a1".
    a2_col : str, optional
        The column name in `gdata.var` representing allele 2. Default is "a2".

    Returns
    -------
    pd.DataFrame
        A BIM-formatted DataFrame with the following columns:
        - "CHR": Chromosome number.
        - "SNP": SNP identifier (from the index of `gdata.var`).
        - "CM": Centimorgan position (default 0 if column is missing).
        - "BP": Base pair position (default 0 if column is missing).
        - "A1": Allele 1 (default 0 if column is missing).
        - "A2": Allele 2 (default 0 if column is missing).
    """
    bim_data = pd.DataFrame(
        {
            "CHR": gdata.var[chrom_col],
            "SNP": gdata.var.index,
            "CM": gdata.var.get(cm_col, 0),
            "BP": gdata.var.get(pos_col, 0),
            "A1": gdata.var.get(a1_col, 0),
            "A2": gdata.var.get(a2_col, 0),
        }
    )

    return bim_data


def _generate_fam_df(gdata, fid_col="fid", pid_col="pid", mid_col="mid", sex_col="sex", phenotype_col="phenotype"):
    """
    Generate a FAM DataFrame from genetic data.

    Parameters
    ----------
    gdata : object
        A genetic data object with an `obs` attribute containing individual information.
    fid_col : str, optional
        The column name in `gdata.obs` representing family IDs. Default is "fid".
    pid_col : str, optional
        The column name in `gdata.obs` representing paternal IDs. Default is "pid".
    mid_col : str, optional
        The column name in `gdata.obs` representing maternal IDs. Default is "mid".
    sex_col : str, optional
        The column name in `gdata.obs` representing the sex of individuals. Default is "sex".
    phenotype_col : str, optional
        The column name in `gdata.obs` representing phenotypes. Default is "phenotype".

    Returns
    -------
    pd.DataFrame
        A FAM-formatted DataFrame with the following columns:
        - "FID": Family ID (default 0 if column is missing).
        - "IID": Individual ID (from the index of `gdata.obs`).
        - "PID": Paternal ID (default 0 if column is missing).
        - "MID": Maternal ID (default 0 if column is missing).
        - "SEX": Sex (default 0 if column is missing).
        - "PHENOTYPE": Phenotype (default 0 if column is missing).
    """
    fam_data = pd.DataFrame(
        {
            "FID": gdata.obs.get(fid_col, 0),
            "IID": gdata.obs.index,
            "PID": gdata.obs.get(pid_col, 0),
            "MID": gdata.obs.get(mid_col, 0),
            "SEX": gdata.obs.get(sex_col, 0),
            "PHENOTYPE": gdata.obs.get(phenotype_col, 0),
        }
    )

    return fam_data


def to_plink(
    gdata,
    output_prefix,
    snp_per_byte=4,
    num_patients_chunk=100,
    chrom_col="chrom",
    cm_col="cm",
    pos_col="pos",
    a1_col="a1",
    a2_col="a2",
    fid_col="fid",
    pid_col="pid",
    mid_col="mid",
    sex_col="sex",
    phenotype_col="phenotype",
):
    """
    Export genotype data in Dask array format to PLINK binary format.

    Parameters
    ----------
    gdata : object
        A genetic data object with an `obs` attribute containing individual information.
    output_prefix: str
        Prefix for the output PLINK files (.bed, .bim, .fam).
    snp_per_byte: int
        Number of SNPs to pack into a single byte. Options are 1, 2, or 4.
    num_patients_chunk: int
        Number of patients in chunk
    chrom_col : str, optional
        The column name in `gdata.var` representing the chromosome. Default is "chrom".
    cm_col : str, optional
        The column name in `gdata.var` representing the centimorgan position. Default is "cm".
    pos_col : str, optional
        The column name in `gdata.var` representing the base pair position. Default is "pos".
    a1_col : str, optional
        The column name in `gdata.var` representing allele 1. Default is "a1".
    a2_col : str, optional
        The column name in `gdata.var` representing allele 2. Default is "a2".
    fid_col : str, optional
        The column name in `gdata.obs` representing family IDs. Default is "fid".
    pid_col : str, optional
        The column name in `gdata.obs` representing paternal IDs. Default is "pid".
    mid_col : str, optional
        The column name in `gdata.obs` representing maternal IDs. Default is "mid".
    sex_col : str, optional
        The column name in `gdata.obs` representing the sex of individuals. Default is "sex".
    phenotype_col : str, optional
        The column name in `gdata.obs` representing phenotypes. Default is "phenotype".
    """
    bim_df = _generate_bim_df(gdata)
    fam_df = _generate_fam_df(gdata)
    dask_genotype_array = gdata.X

    num_individuals, num_snps = dask_genotype_array.shape
    dask_genotype_array = dask_genotype_array.rechunk((num_patients_chunk, num_snps))

    if len(bim_df) != num_snps:
        raise ValueError("Number of SNPs in BIM file does not match genotype matrix.")
    if len(fam_df) != num_individuals:
        raise ValueError("Number of individuals in FAM file does not match genotype matrix.")

    if len(dask_genotype_array.chunks) != 2:
        raise ValueError("Dask array is not 2D. Please ensure the input is (individuals x SNPs).")

    bim_file = f"{output_prefix}.bim"
    bim_df.to_csv(bim_file, sep="\t", index=False, header=False)

    fam_file = f"{output_prefix}.fam"
    fam_df.to_csv(fam_file, sep="\t", index=False, header=False)

    bed_file = f"{output_prefix}.bed"
    with open(bed_file, "wb") as bed:
        bed.write(bytearray([108, 27, 1]))

        for delayed_chunk_row in dask_genotype_array.to_delayed():
            for delayed_chunk in delayed_chunk_row:
                chunk = delayed_chunk.compute()
                if len(chunk.shape) != 2:
                    raise ValueError(f"Chunk is not 2D. Got shape: {chunk.shape}")

                chunk = np.nan_to_num(chunk, nan=3).astype(np.uint8)

                bed_data = []
                for row in chunk:
                    packed_row = []
                    for i in range(0, len(row), 4):
                        genotypes = row[i : i + 4]
                        byte = 0
                        for j, genotype in enumerate(genotypes):
                            byte |= genotype << (j * 2)
                        packed_row.append(byte)
                    bed_data.extend(packed_row)

                bed.write(bytearray(bed_data))

    logger.info(f"Exported: {output_prefix}.bed, {output_prefix}.bim, {output_prefix}.fam")


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
