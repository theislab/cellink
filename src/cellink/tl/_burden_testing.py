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


logger = logging.getLogger(__name__)


def _get_burden(gd_gene, weight_col):
    this_weights = np.array(gd_gene.varm['annotations_0'][weight_col])
    g_weigthed = gd_gene.X * this_weights
    this_burdens = np.nansum(g_weigthed, axis = 1) #TODO implement alternative weighting functions
    return this_burdens

def _postprocess_results(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """"""
    return None

def _compute_burdens_for_gene(this_gd, 
                              this_gene, 
                              weight_cols,
                              annotation_varm = "annotations_0"):
    this_vars = this_gd.varm["annotations_0"][this_gd.varm["annotations_0"]["Gene"] == this_gene].index
    gd_gene = this_gd[:, this_vars]
    
    all_burdens_this_gene = []
    for weight_col in weight_cols: 
        this_burden = _get_burden(gd_gene, weight_col)
        all_burdens_this_gene.append(this_burden)
    all_burdens_this_gene = np.stack(all_burdens_this_gene, axis = 1)
    all_burdens_this_gene = pd.DataFrame(all_burdens_this_gene, index = gd_gene.obs.index,
            columns = weight_cols)
    all_burdens_this_gene["Geneid"] = this_gene
    return all_burdens_this_gene

def compute_burdens(ddata, max_af=0.05, weight_cols=["DISTANCE", "CADD_PHRED"]):
    """Compute gene burdens for each gene and sample using different variant annotations

    Parameters
    ----------
    ddata : ddata
        _description_
    max_af : float, optional
        maximum variant minor allele frequency, by default 0.05
    weight_cols : list, optional
        variant annotations used for weighting (columns of gdata.varm.annotations_0), by default ["DISTANCE", "CADD_PHRED"]

    Returns
    -------
    pandas.DataFrame
        gene burdens for all genes, individuals and all annotations in weightcols
    """    
    this_gd = ddata.gdata.copy()
    this_gd = this_gd[:, this_gd.var["AF"] < max_af]
    all_burdens = []
    for gene in tqdm(ddata.adata.var.index):
        this_b = _compute_burdens_for_gene(this_gd, gene, weight_cols)
        all_burdens.append(this_b)
    all_burdens = pd.concat(all_burdens)

    return all_burdens

def _run_burden_testing_on_gene(pb_data, burden_gene, target_gene, normalize_burdens):
    #target gene: gene whose expression is tested against
    #burden gene: gene whose gene burden score is tested against the target gene 
    #for cis tests, burden_gene = target_gene
    F = pb_data.adata.obsm["F"]
    Y = pb_data.adata[:, [target_gene]].layers["transformed_mean"]

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
    normalize_burdens: bool
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
    )
    Y = pb_data.adata.layers["mean"] #pseudobulk expression
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

    Returns
    -------
        `pd.DataFrame`
            The output data in a `pd.DataFrame` object
    """
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
    )

    results_df["cell_type"] = target_cell_type
    results_df["chrom"] = target_chromosome 
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


