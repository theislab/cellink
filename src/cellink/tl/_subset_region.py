import pandas as pd
import numpy as np
from cellink._core.data_fields import AAnn
def subset_genomic_region(gdata, chrom, start, end):
    """
    Subset the gdata object to only include variants within the specified genomic region.
    
    Parameters:
    gdata: Anndata object containing the genomic data.
    chrom: Chromosome name (e.g., 'chr1').
    start: Start position of the genomic region (0-based).
    end: End position of the genomic region (0-based).
    
    Returns:
    Subsetted Anndata object containing only variants within the specified genomic region.
    """
    # Ensure that the 'chrom', 'pos' columns are present in gdata.var
    if AAnn.chrom not in gdata.var.columns or AAnn.pos not in gdata.var.columns:
        raise ValueError("gdata.var must contain 'chrom' and 'pos' columns.")
    # Trying to cast chrom to the same dtype as gdata.var['chrom'] to ensure that subsetting works correctly
    chrom_column_dtype = type(gdata.var[AAnn.chrom].iloc[0])  

    if type(chrom) != chrom_column_dtype:
        print(f"Attempting to convert chrom value {chrom} from type {type(chrom)} to the same dtype as gdata.var['chrom'] which is {chrom_column_dtype}")
        try:
            chrom = chrom_column_dtype(chrom)
        except Exception as e:  
            raise ValueError(f"Failed to convert chromosome {chrom} to the expected dtype {chrom_column_dtype}. Error: {e}")
    # Subset the variants based on the specified genomic region
    subset_mask = (gdata.var[AAnn.chrom] == chrom) & (gdata.var[AAnn.pos] >= start) & (gdata.var[AAnn.pos] < end)
    # Check if any variants are found in the specified region
    if not subset_mask.any():
        print(f"No variants found in the specified region: {chrom}:{start}-{end}. Returning an empty Anndata object.")
        return gdata[:, subset_mask].copy()  # Return an empty Anndata object with the same structure

    # Subset the Anndata object
    subset_gdata = gdata[:, subset_mask].copy()
    
    return subset_gdata

def subset_gene(gdata, gene_id: str or list):
    """
    Subset the gdata object to only include variants that are annotated to a specific gene.
    It is required that vep annotations have already been added to gdata and that  'gene_id' is present in gdata.uns["variant_annotation_vep"]
    Parameters:
    gdata: Anndata object containing the genomic data.
    gene_id: Gene identifier (e.g., Ensembl gene ID) to subset by.
    
    Returns:
    Subsetted Anndata object containing only variants annotated to the specified gene.
    """
    if "variant_annotation_vep" not in gdata.uns:
        raise ValueError("VEP annotations must be added to gdata before subsetting by gene. Please run add_vep_annos_to_gdata() first.")
    # Ensure that the 'gene_id' column is present in gdata.var
    if 'gene_id' not in gdata.uns["variant_annotation_vep"].reset_index().columns:
        raise ValueError("gdata.uns['variant_annotation_vep'] must contain 'gene_id' column. Have you added VEP annotations to gdata?")
    
    # Subset the variants based on the specified gene_id
    if isinstance(gene_id, list):
        subset_mask_uns = gdata.uns["variant_annotation_vep"].reset_index()["gene_id"].isin(gene_id)
    else:
        subset_mask_uns = gdata.uns["variant_annotation_vep"].reset_index()["gene_id"] == gene_id
    subset_variants = gdata.uns["variant_annotation_vep"].reset_index()[subset_mask_uns]["snp_id"].unique()
    subset_mask_var = gdata.var_names.isin(subset_variants)
    # Subset the Anndata object
    subset_gdata = gdata[:, subset_mask_var].copy()
    subset_gdata.uns["variant_annotation_vep"] = gdata.uns["variant_annotation_vep"].reset_index().loc[subset_mask_uns].copy()
    #subset_gdata.uns = gdata.uns[subset_mask_uns].copy()  # Copy uns to preserve annotations
    return subset_gdata