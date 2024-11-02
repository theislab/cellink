import pandas as pd
import requests
from tqdm.auto import tqdm


def _split_snp_id(snp_str):
    chrom = snp_str.split(":")[0]
    pos = snp_str.split(":")[1].split("_")[0]
    a0 = snp_str.split(":")[1].split("_")[1]
    a1 = snp_str.split(":")[1].split("_")[2]
    return chrom, pos, a0, a1


def get_snp_df(variant_codes, server="https://grch37.rest.ensembl.org/"):
    """
    Retrieve SNP (Single Nucleotide Polymorphism) information and overlap with genes from Ensembl.

    This function takes a list of SNP identifiers, queries the Ensembl REST API to retrieve information about
    the SNPs and their overlapping genes, and returns this data in the form of two dataframes. The first dataframe
    contains SNP-related data, including whether each SNP is located within a protein-coding gene and its
    clinical significance. The second dataframe provides gene-related information for the overlapping genes.

    Parameters
    ----------
    variant_codes : list of str
        A list of SNP identifiers in the format of chromosome, position, and alleles (e.g., `1_55516888_T_C`).
    server : str, optional
        The URL of the Ensembl REST API server to query. Defaults to the GRCh37 Ensembl server.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing:
        - var_df: A dataframe with SNP-related information, including whether the SNP is within a gene and its
          clinical significance. Each row corresponds to an SNP.
        - gene_df: A dataframe containing information about the genes that overlap with the SNPs, with genes as the index.

    Notes
    -----
    The function uses the Ensembl REST API to query data, specifically querying for overlapping regions between
    SNPs and genes. The results include SNPs sourced from dbSNP, and the genes returned are limited to
    protein-coding genes.

    Example
    -------
    >>> variant_codes = ["1_55516888_T_C", "2_117900001_A_G"]
    >>> var_df, gene_df = get_snp_df(variant_codes)
    >>> var_df.head()
          snp_id  is_in_gene    genes  ...  clinical_significance
    0  1_55516888_T_C        True    GENE1  ...          pathogenic
    1  2_117900001_A_G       False     GENE2 ...                benign
    >>> gene_df.head()
          biotype    start     end  ...       strand
    id
    GENE1  protein_coding  5500000  5600000  ...          1
    GENE2  protein_coding  11780000  11800000 ...         -1
    """
    results = []
    genes_list = []
    for snp in tqdm(variant_codes, desc="SNPs"):
        chrom, pos, a0, a1 = _split_snp_id(snp)
        ext = f"/overlap/region/human/{chrom}:{pos}-{pos}?feature=variation;feature=gene"
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})

        if not r.ok:
            r.raise_for_status()

        decoded = r.json()
        is_in_gene = any(entry["feature_type"] == "gene" and entry["biotype"] == "protein_coding" for entry in decoded)
        genes = ",".join([entry["id"] for entry in decoded if entry["feature_type"] == "gene"])
        for entry in decoded:
            if entry["feature_type"] == "gene":
                genes_list.append(entry)
        for entry in decoded:
            if (
                entry["feature_type"] != "gene"
                and entry["source"] == "dbSNP"
                and (a0 in entry["alleles"] and a1 in entry["alleles"])
            ):
                entry["rs_id"] = entry.pop("id")
                entry["ref"], entry["alt"] = (allele for allele in entry["alleles"] if allele in [a0, a1])
                entry["alleles"] = "".join(entry["alleles"])
                entry["clinical_significance"] = ",".join(entry.get("clinical_significance", []))
                results.append(
                    {
                        "snp_id": snp,
                        "is_in_gene": is_in_gene,
                        "genes": genes,
                        **entry,
                    }
                )
    var_df = pd.DataFrame(results)

    def process_df(group, **kwargs):
        lengths = group.alleles.apply(len)
        group["rs_ids"] = ",".join(group.rs_id.unique().tolist())
        if 2 in list(lengths):
            return group[lengths == 2]

        return group

    var_df = var_df.groupby("snp_id", sort=False).apply(process_df, include_groups=False)
    var_df = var_df.drop_duplicates(subset=var_df.columns.difference(["rs_id"]))
    gene_df = pd.DataFrame(genes_list).set_index("id")
    return var_df, gene_df
