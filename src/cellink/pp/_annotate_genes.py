import logging

import pandas as pd
from anndata import AnnData


def annotate_genes(adata: AnnData, gene_annotation_path: str) -> AnnData:
    """
    Annotate genes in an AnnData object with information from a gene annotation file.

    This function loads gene annotation data from a CSV file, adds it to the
    var attribute of the provided AnnData object, and subsets the AnnData object
    to only include genes with annotations. It matches genes based on their
    Ensembl ID and adds all available annotation information.

    Args:
        adata (AnnData): The AnnData object to annotate.
        gene_annotation_path (str): Path to the gene annotation CSV file.

    Returns
    -------
        AnnData: A new AnnData object with added gene annotations in the var attribute,
                 containing only the genes with annotations.

    Raises
    ------
        FileNotFoundError: If the gene annotation file is not found.
        ValueError: If no matching genes are found between the AnnData object and the annotation file.
    """
    # Load gene annotation
    gene_annotation = pd.read_csv(gene_annotation_path, index_col="ensembl_gene_id")

    # Find matching genes
    matching_genes = adata.var_names.intersection(gene_annotation.index)
    non_matching_genes = adata.var_names.difference(gene_annotation.index)

    if len(matching_genes) == 0:
        raise ValueError("No matching genes found between AnnData and gene annotation file.")

    # Subset the AnnData object to include only matching genes
    adata_subset = adata[:, matching_genes].copy()

    # Add gene annotations to adata_subset.var
    adata_subset.var = adata_subset.var.join(gene_annotation.loc[matching_genes])

    # Rename 'chromosome_name' to 'chrom'
    if "chromosome_name" in adata_subset.var.columns:
        adata_subset.var = adata_subset.var.rename(columns={"chromosome_name": "chrom"})
    else:
        logging.warning("Column 'chromosome_name' not found in gene annotations.")

    # Log information about the annotation process
    logging.info(f"Added annotations for {len(matching_genes)} genes.")
    logging.warning(f"Removed {len(non_matching_genes)} genes without annotations.")
    logging.info(f"Original AnnData had {adata.n_vars} genes, new AnnData has {adata_subset.n_vars} genes.")

    return adata_subset
