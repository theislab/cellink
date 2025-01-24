import copy 
from tqdm import tqdm
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.preprocessing import quantile_transform
from collections.abc import Callable, Mapping, Sequence
from cellink._core import DonorData
from cellink.tl._eqtl import GWAS, _dump_results, _get_pb_data, quantile_transform, _apply_transforms_seq, _column_normalize
from pathlib import Path 
from functools import partial
from anndata.utils import asarray
import logging
import requests


logger = logging.getLogger(__name__)


def _get_burden(gd_gene, weight_col):
    this_weights = np.array(gd_gene.varm['annotations_0'][weight_col])
    g_weigthed = gd_gene.X * this_weights
    this_burdens = np.nansum(g_weigthed, axis = 1) #TODO implement alternative weighting functions
    return this_burdens


def _postprocess_results(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """"""
    return None


def _get_gene_location(ensembl_id):
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        #print(data)
        if data.get('strand') == 1: 
            # forward: this means start ----> end (start < end)
            chrom = data.get('seq_region_name')
            start = data.get('start')
            end = data.get('end')
            return chrom, start, end
        else:
            # reverse: this means end <---- start (start > end, on the scale of the forward strand)
            chrom = data.get('seq_region_name')
            start = data.get('end')
            end = data.get('start')
            return chrom, start, end
    else:
        chrom = np.nan
        start = np.nan
        end = np.nan
        return chrom, start, end


def _calc_tss_distance_per_gene(variants_df,
                                gene_start,
                                col_name_distance,
                                col_name_saige=""):
    """
    Calculate absolute TSS Distance and if set also TSS Distance Saige

    Parameters:
        variants_df (pd.DataFrame): DataFrame with SNP ID as index and contains variant position in column "Position"
        gene_start (int): Start of Gene aka TS
        col_name_distance (str): name of TSS column
        col_name_saige (str, optional): determines whether saige formula should be calculated as well.

    Returns:
        distances_df (pd.DataFrame): Dataframe with tss distance per variant
    """
    distances = {"snp_id": [], col_name_distance: []}
    saige_distances = []

    for i, row in variants_df.iterrows():
        # Calculate absolute distances to start and end positions
        distance_to_start = abs(row["Position"] - gene_start)

        distances["snp_id"].append(i)
        distances[col_name_distance].append(distance_to_start)

        # calculate saige if parameter is set
        if col_name_saige != "":
            distance_saige = np.exp(-1e-5 * distance_to_start)
            saige_distances.append(distance_saige)

    # add saige to final dataframe if parameter is set
    if col_name_saige != "":
        distances[col_name_saige] = saige_distances

    # return dataframe with tss distance and if applied tss distance saige per variant
    distance_df = pd.DataFrame(distances)
    distance_df.set_index("snp_id", inplace=True)
    return distance_df


def _find_snps_near_gene(gdata, gene_chrom, gene_start, gene_end, bp_range=10000):
    """
    Finds SNPs within a specified range of a gene's location.

    Parameters:
        gdata (pd.DataFrame): DataFrame with a 'Location' column in the format "chromosome:position".
        gene_chrom, gene_start, gene_end (int)
        bp_range (int): Range in base pairs to search upstream and downstream.

    Returns:
        pd.DataFrame: With data of SNPs within the specified range.
    """
    # Parse the gene location
    #import ipdb; ipdb.set_trace()

    # Extract chromosome and position from the SNPs
    gene_chrom = str(gene_chrom)
    gdata_df = gdata.copy()
    gdata_df[['Chromosome', 'Position']] = gdata_df['Location'].str.split(':', expand=True)
    gdata_df['Position'] = gdata_df['Position'].astype(int)

    # Filter for SNPs within the range
    # snps_in_range = gdata_df[
    #     (gdata_df['Chromosome'] == gene_chrom) &
    #     (gdata_df['Position'] >= gene_start - bp_range) &
    #     (gdata_df['Position'] <= gene_end + bp_range)
    # ]

    if gene_start < gene_end:
        # forward strand
        snps_upstream = gdata_df[
            (gdata_df['Chromosome'] == gene_chrom) &
            (gdata_df['Position'] >= gene_start - bp_range) &
            (gdata_df['Position'] <= gene_start)

        ]
        snps_downstream = gdata_df[
            (gdata_df['Chromosome'] == gene_chrom) &
            (gdata_df['Position'] <= gene_start + bp_range) &
            (gdata_df['Position'] > gene_start)
        ]
    else:
        # reverse strand
        snps_upstream = gdata_df[
            (gdata_df['Chromosome'] == gene_chrom) &
            (gdata_df['Position'] <= gene_start + bp_range) &
            (gdata_df['Position'] >= gene_start)
        ]
        snps_downstream = gdata_df[
            (gdata_df['Chromosome'] == gene_chrom) &
            (gdata_df['Position'] >= gene_start - bp_range) &
            (gdata_df['Position'] < gene_start)
        ]

    # return snps_in_range.index
    return snps_upstream, snps_downstream


def _compute_burdens_for_gene(this_gd,
                              this_gene,
                              gene_chrom,
                              gene_start,
                              gene_end,
                              weight_cols,
                              annotation_varm="annotations_0",
                              window_size=100000,
                              DNA_LM_up="",
                              DNA_LM_down="",
                              DNA_LM_mixed="DNA_LM_mixed",
                              GENE_TSS_DISTANCE="",
                              GENE_TSS_DISTANCE_SAIGE=""):
    """
    Compute burdenscores for a given gene and given annotations

    Parameters:
        this_gd (pd.DataFrame): ddata.gdata.
        this_gene (str): Ensemble ID.
        gene_chrom, gene_start, gene_end (int)
        weight_cols (list): colnames of variant annotations to compute burden scores for.
        annotation_varm (str): key for pd.DataFrame (gdata.varm[key])
        window_size (int)
        DNA_LM_up (str): colname for DNA_LM upstream model
                    if empty, mixed model is not computed
        DNA_LM_down (str): colname for DNA_LM downstream model
                    if empty, mixed model is not computed
        DNA_LM_mixed (str): name of mixed model column
        GENE_TSS_DISTANCE (str): name for tss distance column
        GENE_TSS_DISTANCE_SAIGE (str): name for tss distance saige column

    Returns:
        pd.DataFrame containing burden scores for this_gene across the weight_cols
    """
    # Return a dataframe with None entries for weight cols, if gene location is nan (this means it could not be found in ensembl)
    if np.isnan(gene_chrom):
        print(f"Failed to retrieve location for gene {this_gene}. No Burden scores computed.")
        # Create a DataFrame with None for all the weight columns
        empty_burdens = pd.DataFrame(
            None,
            index=this_gd.obs.index,  # Assuming these are the sample indices
            columns=weight_cols
        )
        # Add the Geneid column
        empty_burdens["Geneid"] = this_gene
        return empty_burdens

    # Filter the variants using the SNP location and gene location
    this_vars_up_df, this_vars_down_df = _find_snps_near_gene(this_gd.varm[annotation_varm], gene_chrom, gene_start, gene_end, bp_range=window_size)

    # get snps IDs
    this_vars_down = this_vars_down_df.index
    this_vars_up = this_vars_up_df.index

    gd_gene = this_gd[:, this_vars_up.append(this_vars_down).unique()].copy()

    # if mixed model is computed, add column for DNA_LM mixed model
    if DNA_LM_up != "":
        # Add the "DNA_LM_mixed" column to the annotations_0 DataFrame
        gd_gene.varm["annotations_0"][DNA_LM_mixed] = np.nan  # Initialize the column with NaN

        # Assign values to the "DNA_LM_mixed" column based on the conditions
        gd_gene.varm["annotations_0"].loc[this_vars_up, DNA_LM_mixed] = gd_gene.varm["annotations_0"].loc[this_vars_up, DNA_LM_up]
        gd_gene.varm["annotations_0"].loc[this_vars_down, DNA_LM_mixed] = gd_gene.varm["annotations_0"].loc[this_vars_down, DNA_LM_down]

    if GENE_TSS_DISTANCE_SAIGE != "": # calc GENE_TSS_DISTANCE and GENE_TSS_DISTANCE_SAIGE
        gd_gene.varm["annotations_0"][GENE_TSS_DISTANCE] = np.nan # Initialize the column with NaN
        gd_gene.varm["annotations_0"][GENE_TSS_DISTANCE_SAIGE] = np.nan  # Initialize the column with NaN

        # calculate GENE_TSS_DISTANCE and GENE_TSS_DISTANCE_SAIGE independent of up or downstream
        all_variants = pd.concat([this_vars_up_df, this_vars_down_df], axis=0)
        distances = _calc_tss_distance_per_gene(all_variants, gene_start, GENE_TSS_DISTANCE, GENE_TSS_DISTANCE_SAIGE)

        # add GENE_TSS_DISTANCE and GENE_TSS_DISTANCE_SAIGE to annotation 0
        gd_gene.varm["annotations_0"].loc[distances.index, GENE_TSS_DISTANCE] = distances[GENE_TSS_DISTANCE]
        gd_gene.varm["annotations_0"].loc[distances.index, GENE_TSS_DISTANCE_SAIGE] = distances[GENE_TSS_DISTANCE_SAIGE]

    elif GENE_TSS_DISTANCE != "":  # calc only GENE_TSS_DISTANCE
        gd_gene.varm["annotations_0"][GENE_TSS_DISTANCE] = np.nan  # Initialize the column with NaN

        # calculate tss distance independent of up or downstream
        all_variants = pd.concat([this_vars_up_df, this_vars_down_df])
        tss_distances = _calc_tss_distance_per_gene(all_variants, gene_start, GENE_TSS_DISTANCE)

        # add to gd_gene
        gd_gene.varm["annotations_0"].loc[tss_distances.index, GENE_TSS_DISTANCE] = tss_distances[GENE_TSS_DISTANCE]

    all_burdens_this_gene = []
    for weight_col in weight_cols:
        this_burden = _get_burden(gd_gene, weight_col)
        all_burdens_this_gene.append(this_burden)

    all_burdens_this_gene = np.stack(all_burdens_this_gene, axis=1)
    all_burdens_this_gene = pd.DataFrame(all_burdens_this_gene, index=gd_gene.obs.index, columns=weight_cols)
    all_burdens_this_gene["Geneid"] = this_gene

    return all_burdens_this_gene


def compute_burdens(ddata,
                    max_af=0.05,
                    weight_cols=["DISTANCE", "CADD_PHRED"],
                    annotations_varm="annotations_0",
                    window_size=1000000,
                    DNA_LM_up="",
                    DNA_LM_down="",
                    DNA_LM_mixed="DNA_LM_mixed",
                    GENE_TSS_DISTANCE="",
                    GENE_TSS_DISTANCE_SAIGE=""):
    """Compute gene burdens for each gene and sample using different variant annotations

    Parameters
    ----------
    ddata : ddata
        _description_
    max_af : float, optional
        maximum variant minor allele frequency, by default 0.05
    weight_cols : list, optional
        variant annotations used for weighting (columns of gdata.varm.annotations_0), by default ["DISTANCE", "CADD_PHRED"]
    annotations_varm: str, optional
        key for gdata.varm dataframe (eg: gdata.varm["annotations_0"])
    window_size: int, optional
        range around gene TSS, in which variants are regarded for gene burden scores
    DNA_LM_up: str, optional, if DNA_LM mixed model should be computed
        colname of DNA_LM score upstream model
    DNA_LM_down: str, optional, if DNA_LM mixed model should be computed
        colname of DNA_LM score downstream model
    DNA_LM_mixed: str, optional
        colname of DNA_LM score mixed model
    GENE_TSS_DISTANCE: str, optional
        colname for TSS distances and also flag to compute its burden 
    GENE_TSS_DISTANCE_SAIGE:str, optional
        colname for TSS distances using saige formula and also flag to compute its burden 
    Returns
    -------
    pandas.DataFrame
        gene burdens for all genes, individuals and all annotations in weightcols
    """
    if (DNA_LM_up != "" and DNA_LM_down == "") or (DNA_LM_up == "" and DNA_LM_down != ""):
        raise ValueError("If you want to compute the burden scores using the DNA_LM mixed model, you must set both DNA_LM_up and DNA_LM_down, else leave both empty.")

    this_gd = ddata.gdata.copy()
    this_ad = ddata.adata.copy()
    this_gd = this_gd[:, this_gd.var["maf"] < max_af]
    all_burdens = []

    if not all(col in this_ad.var.columns for col in ["chromosome", "start", "end"]):
        # compute all the gene locations
        this_ad.var[['chromosome', 'start', 'end']] = this_ad.var.index.to_series().apply(
            lambda x: pd.Series(_get_gene_location(x))
        )

    # add mixed model to the weight cols for which burden score is computed
    if DNA_LM_up != "" and DNA_LM_down != "" and DNA_LM_mixed not in weight_cols:
        weight_cols.append(DNA_LM_mixed)
        if DNA_LM_up not in weight_cols:
            weight_cols.append(DNA_LM_up)
        if DNA_LM_down not in weight_cols:
            weight_cols.append(DNA_LM_down)

    # add tss distance to the weight cols to initialize burden score computations
    if GENE_TSS_DISTANCE != "" and GENE_TSS_DISTANCE not in weight_cols:
        weight_cols.append(GENE_TSS_DISTANCE)

    # add tss distance saige to the weight cols to initialize burden score computations
    if GENE_TSS_DISTANCE_SAIGE != "":
        if GENE_TSS_DISTANCE_SAIGE not in weight_cols:
            weight_cols.append(GENE_TSS_DISTANCE_SAIGE)
        if GENE_TSS_DISTANCE == "":  # if saige is set then tss distance has to be calculated too
            GENE_TSS_DISTANCE = "GENE_TSS_DISTANCE"

    for gene in tqdm(this_ad.var.index):  # add subsetting [0:500] for test purposes
        gene_chrom = int(this_ad.var.loc[gene, "chromosome"])
        gene_start = int(this_ad.var.loc[gene, "start"])
        gene_end = int(this_ad.var.loc[gene, "end"])

        this_b = _compute_burdens_for_gene(this_gd, gene, gene_chrom, gene_start, gene_end, weight_cols, annotations_varm, window_size, DNA_LM_up, DNA_LM_down, DNA_LM_mixed, GENE_TSS_DISTANCE, GENE_TSS_DISTANCE_SAIGE)
        all_burdens.append(this_b)

    all_burdens = pd.concat(all_burdens)

    return all_burdens


def _run_burden_testing_on_gene(pb_data, burden_gene, target_gene, normalize_burdens):
    # target gene: gene whose expression is tested against
    # burden gene: gene whose gene burden score is tested against the target gene 
    # for cis tests, burden_gene = target_gene
    F = pb_data.adata.obsm["F"]
    Y = pb_data.adata[:, [target_gene]].layers["transformed_mean"]

    print(f"burden_gene: {burden_gene}")
    
    #TODO don't hard code the gene_id column
    G = pb_data.gdata.uns["gene_burdens"].query("Geneid ==@burden_gene")\
            .drop(columns = "Geneid").loc[pb_data.gdata.obs_names]
    burden_types = G.columns
    G = np.array(G)
    
    #TODO check if normalizing is really needed 
    if normalize_burdens:
        G = _column_normalize(G)
    gwas = GWAS(np.array(Y).reshape(-1, 1), F)
    gwas.process(G)
    beta = gwas.getBetaSNP().reshape(-1)
    pv = gwas.getPv().reshape(-1)
    pv[np.isnan(pv)] = 1

    out_df = pd.DataFrame({"burden_gene":burden_gene, 
                            "target_gene": target_gene,
                            "burden_type": burden_types, 
                            "pvalue": pv, 
                            "beta": beta})
    
    return out_df



def _burden_test(
    donor_data: DonorData,
    gene_burdens,
    target_cell_type: str,
    target_chromosome: str,
    target_genes: Sequence[str] | None,
    donor_key_in_scdata: str,
    sex_key_in_scdata: str,
    age_key_in_scdata: str,
    pseudobulk_aggregation_type: str,
    min_individuals_threshold: int,
    n_top_genes: int,
    n_sc_comps: int,
    n_genetic_pcs: int,
    n_cellstate_comps: int,
    transforms_seq: Sequence[Callable],
    pv_transforms: Mapping[str, Callable],
    prog_bar: bool,
    normalize_burdens: bool,
    eigenvector_df: pd.DataFrame
    
) -> pd.DataFrame:
    """Runs the burden testing pipeline on a given pair of (`target_cell_type`, `target_chromosome`) over all genes
       Currently only implemented for cis testing (gene burden of gene_A vs expression of gene_A)

    Parameters
    ----------
        `gene_burdens: pd.DataFrame`
            Gene burdens

    Returns
    -------
        `pd.DataFrame`
            The output data with the parsed statistics to be stored for all genes
    """
    # output results
    output = []
    # retrieving tmp data for current cell_type and chromosome
    # generate pseudbulks
    pb_data = _get_pb_data(
        donor_data.adata,
        donor_data.gdata,
        target_cell_type,
        target_chromosome,
        donor_key_in_scdata,
        sex_key_in_scdata,
        age_key_in_scdata,
        pseudobulk_aggregation_type,
        min_individuals_threshold,
        n_top_genes,
        n_sc_comps,
        n_genetic_pcs,
        n_cellstate_comps,
        eigenvector_df
    )
    
    for gene in target_genes:
        if gene not in pb_data.adata.var_names:
            print(f"This is missing after pb calc: {gene}")
            
    Y = pb_data.adata.layers["mean"]  # pseudobulk expression
    # defining transform function
    transform_fn = partial(_apply_transforms_seq, transforms_seq=transforms_seq)
    # applying transformation
    Y = asarray(Y)
    if transform_fn is not None:
        logger.info("transforming phenotype")
        Y = transform_fn(Y)
    pb_data.adata.layers["transformed_mean"] = Y
    # retrieving current genes
    current_genes = pb_data.adata.var_names

    print(f"current_genes: {current_genes}")
    print(f"target_genes: {target_genes}")
    
    # optionally setting all genes by default
    if target_genes is None:
        # logging message
        msg = f"`target_genes` not defined, running the EQTL agains all the {len(current_genes)} genes for current combination of {target_cell_type=}, {target_chromosome=}."
        logger.info(msg)
        target_genes = current_genes
    # early return if pseudo bulked data is None
    pb_data.gdata.uns["gene_burdens"] = gene_burdens.query("Geneid in @target_genes")

    if pb_data is None:
        # logging message
        msg = "Filtering the pseudo-bulked data retrieved an empty dataset."
        logger.info(msg)
        return output
    # defining optional iterator
    iterator = tqdm(range(len(target_genes))) if prog_bar else None
    # iterating over the target genes
    for target_gene in target_genes:
        # checking that the target gene appears in the current genes
        if target_gene not in current_genes:
            # logging message
            msg = f"The gene {target_gene} does not appear in `current_genes`, skipping iteration"
            logger.info(msg)
            continue
        # running the gwas on gene
        burden_test_out = _run_burden_testing_on_gene(
            pb_data,
            burden_gene = target_gene, 
            target_gene = target_gene,
            normalize_burdens = normalize_burdens,
        )
        output.append(burden_test_out)

        if iterator is not None:
            iterator.update()
    output = pd.concat(output)

    return output


def burden_test(
    donor_data: DonorData,
    gene_burdens: pd.DataFrame,
    target_cell_type: str,
    target_chromosome: str,
    eigenvector_df: pd.DataFrame,
    target_genes: Sequence[str] | None = None,
    donor_key_in_scdata: str = "individual",
    sex_key_in_scdata: str = "sex",
    age_key_in_scdata: str = "age",
    pseudobulk_aggregation_type: str = "mean",
    min_individuals_threshold: int = 10,
    n_top_genes: int = 5_000,
    n_sc_comps: int = 15,
    n_genetic_pcs: int = 20,
    n_cellstate_comps: int = 50,
    transforms_seq: Sequence[Callable] | None = (quantile_transform,),
    pv_transforms: Mapping[str, Callable] | None = None,
    prog_bar: bool = True,
    dump_results: bool = True,
    dump_dir: str | None = None,
    file_prefix: str | None = "burden",
    use_cell_type_chrom_specific_dir: bool = True,
    normalize_burdens: bool = True,
) -> [pd.DataFrame, pd.DataFrame]:
    """Runs the EQTL pipeline on a given pair of (`target_cell_type`, `target_chromosome`) over all genes and
    stores the results to a `pd.DataFrame` object and optionally to disk

    Parameters
    ----------
        `target_cell_type: str`
            Target chromosome which GWAS experiment was ran on
        `target_chromosome: str`
            Target chromosome which GWAS experiment was ran on
        `eigenvector_df: pd.DataFrame`
            sampleId x eigenvector

    Returns
    -------
        `pd.DataFrame`
            The output data in a `pd.DataFrame` object
    """
    for gene in target_genes:
        if gene not in donor_data.adata.var_names:
            print(f"This is missing before pb calc: {gene}")
    
    # ensuring the type of target chromose is a string
    if isinstance(target_chromosome, int):
        target_chromosome = str(target_chromosome)
    # optionally creating a directory to store the results for
    # the current `target_cell_type`, `target_chromosome` combination
    # running the pipeline and constructing results DataFrame
    results_df = _burden_test(
        donor_data,
        gene_burdens,
        target_cell_type,
        target_chromosome,
        target_genes,
        donor_key_in_scdata,
        sex_key_in_scdata,
        age_key_in_scdata,
        pseudobulk_aggregation_type,
        min_individuals_threshold,
        n_top_genes,
        n_sc_comps,
        n_genetic_pcs,
        n_cellstate_comps,
        transforms_seq,
        pv_transforms,
        prog_bar,
        normalize_burdens,
        eigenvector_df
    )

    results_df["cell_type"] = target_cell_type
    results_df["chromosome"] = target_chromosome
    # postprocessing the results
    postprocessed_dfs = _postprocess_results(results_df)
    # optionally saving the results to disk

    if dump_results:
        target_cell_type_file = target_cell_type.replace(" ", "-")
        logger.warning(f"{dump_dir=} {file_prefix=}")
        _dump_results(
            results_df,
            postprocessed_dfs,
            0, #cis_window place holder
            target_cell_type_file,
            target_chromosome,
            None,
            file_prefix,
            dump_dir,
        )
    # constructing out dictionary
    return results_df, postprocessed_dfs


