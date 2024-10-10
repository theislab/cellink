import copy 
from tqdm import tqdm
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.preprocessing import quantile_transform

from cellink import DonorData
from cellink.tl._gwas import GWAS

def _get_burden(gd_gene, weight_col):
    this_weights = np.array(gd_gene.varm['annotations_0'][weight_col])
    g_weigthed = gd_gene.X * this_weights
    this_burdens = np.nansum(g_weigthed, axis = 1)
    return this_burdens
    
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
    all_burdens_this_gene["Gene"] = this_gene
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

def _my_quantile_transform(x, seed=1):
    """
    Gaussian quantile transform for values in a pandas Series.

    :param x: Input pandas Series.
    :type x: pd.Series
    :param seed: Random seed.
    :type seed: int
    :return: Transformed Series.
    :rtype: pd.Series

    .. note::
        "nan" values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()

    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)

    x_transform[~is_nan] = quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]
    
    x_transform = pd.Series(x_transform, index = x.index)
    return x_transform


def _run_burden_testing_on_gene(ddata_ct, this_gene, 
                               covar_cols=["sex", "age"]):
    F = ddata_ct.gdata.obs[covar_cols]
    Y = ddata_ct.gdata.obsm["pseudo_mean"][this_gene]
    Y = _my_quantile_transform(Y)
    G = ddata_ct.gdata.uns["gene_burdens"].query("Gene ==@this_gene")\
        .drop(columns = "Gene").loc[Y.index]

    gwas = GWAS(np.array(Y).reshape(-1, 1), F)
    gwas.process(G)
    beta = gwas.getBetaSNP().reshape(-1)
    pv = gwas.getPv().reshape(-1)
    pv[np.isnan(pv)] = 1

    res = pd.DataFrame({"Gene_burden":this_gene, 
                        "burden_type": G.columns, 
                        "pvalue": pv, 
                        "beta": beta})
    
    return res

def run_burden_testing(ddata_ct):
    all_res = []
    for this_gene in tqdm(ddata_ct.gdata.obsm["pseudo_mean"].columns):
        this_res = _run_burden_testing_on_gene(ddata_ct, this_gene)
        all_res.append(this_res)
    all_res = pd.concat(all_res)
    return(all_res)



def compute_pseudobulks(ddata, target_cell_type, cell_type_col = "cell_label"):
    """Compute pseudobulks for a given cell type

    Parameters
    ----------
    ddata : ddata
        _description_
    target_cell_type : str
        cell type to subset for and compute pseudobulk
    cell_type_col : str, optional
        column which has the cell type, by default "cell_label"

    Returns
    -------
    ddata
        ddata subset for the selected cell type, with pseudobulk counts in gdata.obsm.pseudo_mean 
    """    
    ddata_ct = copy.deepcopy(ddata)
    ddata_ct.adata = ddata_ct.adata[ddata_ct.adata.obs[cell_type_col] == target_cell_type]

    kept_donors = list(ddata_ct.adata.obs[ddata_ct.donor_key_in_sc_adata].unique())
    #if some cell types are not present in a donor that donor gets removed from adata so it also has to be removed from gdata
    ddata_ct.gdata = ddata_ct.gdata[kept_donors, :]
    print("Normalizing and log1p transforming data")
    sc.pp.normalize_total(ddata_ct.adata)
    sc.pp.log1p(ddata_ct.adata)
    sc.pp.normalize_total(ddata_ct.adata)

    ddata_ct.aggregate("X", "pseudo_mean", agg_func = "mean") 


    genes_to_keep = ddata_ct.gdata.obsm["pseudo_mean"].columns[(ddata_ct.gdata.obsm["pseudo_mean"] > 0).sum(axis = 0) > 10]
    
    ddata_ct.gdata.obsm["pseudo_mean"] = ddata_ct.gdata.obsm["pseudo_mean"][genes_to_keep]

    return ddata_ct

    
    

