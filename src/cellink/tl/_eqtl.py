import logging
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.utils import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import quantile_transform as sk_quantile_transform
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from dask.array import Array as DaskArray

from cellink._core import DonorData
from cellink.tl._gwas import GWAS

logger = logging.getLogger(__name__)

__all__ = ["eqtl", "quantile_transform", "bonferroni_adjustment", "q_value"]


def _column_normalize(X: np.ndarray) -> np.ndarray:
    """"""
    assert X.ndim == 2
    return (X - X.mean(0)) / (X.std(0) * np.sqrt(X.shape[1]))


def _register_fixed_effects(
    pbdata: ad.AnnData,
    gdata: ad.AnnData,
    n_top_genes: int,
    n_epcs: int,
    n_gpcs: int,
    sex_key_in_scdata: str,
    age_key_in_scdata: str,
    gen_pc_path: str | None,
) -> ad.AnnData:
    """Registers the fixed effect matrix for the given pseudo-bulked data

    Parameters
    ----------
        `pbdata: ad.AnnData`
            The `ad.AnnData` object holding the pseudo-bulked single cell data

    Returns
    -------
        `ad.AnnData` containing with the updated `obsm` with the fixed effects
    """
    # compute expression PCs
    sc.pp.highly_variable_genes(pbdata, n_top_genes=n_top_genes)
    sc.tl.pca(pbdata, use_highly_variable=True, n_comps=n_epcs)
    pbdata.obsm["ePCs"] = _column_normalize(pbdata.obsm["X_pca"])
    # load genetic PCs
    if gen_pc_path is None:
        gen_pcs = sc.tl.pca(gdata.X, n_comps=n_gpcs)
    else:
        # load genetic PCs  
        gen_pcs = pd.read_csv(gen_pc_path, sep=" ", header=None, index_col=1).drop(columns=[0])
        gen_pcs = gen_pcs.loc[pbdata.obs.index, :].iloc[:, :n_gpcs].values
    # load patient covariates
    sex_encoded = pd.Categorical(pbdata.obs[sex_key_in_scdata]).codes[:, None]
    age_standardized = StandardScaler().fit_transform(pbdata.obs[age_key_in_scdata].values.reshape(-1, 1))
    constant_col = np.ones_like(sex_encoded)
    # store fixed effects in pb_adata
    pbdata.obsm["F"] = np.concatenate(
        (constant_col, sex_encoded, age_standardized, gen_pcs, pbdata.obsm["ePCs"]), axis=1
    )
    # final normalization
    pbdata.obsm["F"][:, 2:] = _column_normalize(pbdata.obsm["F"][:, 2:])
    return pbdata


def _map_col_scdata_obs_to_pbdata(
    scdata: ad.AnnData, pbdata: ad.AnnData, donor_key_in_scdata: str, column: str
) -> ad.AnnData:
    """Maps the selected column, assumed to have only one unique value for each patient (i.e.: age, sex, etc.) from the base single cell data to the pseudo-bulked one, as not all the columns are returned after aggregation

    Parameters
    ----------
        `pbdata: ad.AnnData`
            The `ad.AnnData` object holding the pseudo-bulked single cell data
        `column: str`
            Column in `self.scdata.obs` of patient covariates that we want to map back

    Returns
    -------
        `ad.AnnData` containing with the updated `obs` containing the required column
    """
    # mapping over the individuals
    reference_data = scdata.obs[[donor_key_in_scdata, column]]
    reference_data = reference_data.groupby(donor_key_in_scdata).agg(["unique"])

    # function for making sure the values are unique
    def retrieve_unique_value_safe(row):
        assert len(row) == 1
        return row[0]

    # retrieving the unique values for each donor
    reference_data[column] = reference_data[column].map(retrieve_unique_value_safe)
    # merging the data and updating column names
    pbdata.obs = pd.merge(pbdata.obs, reference_data[column], left_on=donor_key_in_scdata, right_index=True)
    pbdata.obs[column] = pbdata.obs["unique"]
    pbdata.obs = pbdata.obs.drop(columns=["unique"], axis=1)
    return pbdata


def _pseudobulk_scdata(
    scdata_cell: ad.AnnData,
    donor_key_in_scdata: str,
    sex_key_in_scdata: str,
    age_key_in_scdata: str,
    pseudobulk_aggregation_type: str,
) -> ad.AnnData:
    """Pseudobulks the single cell data

    Parameters
    ----------
        `scdata_cell: ad.AnnData`
            The `ad.AnnData` object holding the cells with the given type

    Returns
    -------
        `ad.AnnData` containing with the single cell data aggregated by patient
    """
    # aggregating the data
    pbdata = sc.get.aggregate(
        adata=scdata_cell,
        by=donor_key_in_scdata,
        func=pseudobulk_aggregation_type,
    )
    # storing data lost in the aggregation
    pbdata.X = pbdata.layers.pop("mean")
    pbdata = _map_col_scdata_obs_to_pbdata(scdata_cell, pbdata, donor_key_in_scdata, sex_key_in_scdata)
    pbdata = _map_col_scdata_obs_to_pbdata(scdata_cell, pbdata, donor_key_in_scdata, age_key_in_scdata)
    return pbdata


def _get_pb_data(
    scdata: ad.AnnData,
    gdata: ad.AnnData,
    cell_type: str,
    target_chromosome: str,
    donor_key_in_scdata: str,
    sex_key_in_scdata: str,
    age_key_in_scdata: str,
    cell_type_key_in_scdata: str,
    pseudobulk_aggregation_type: str,
    min_individuals_threshold: int,
    n_top_genes: int,
    n_epcs: int,
    n_gpcs: int,
    n_cellstate_comps: int,
    gen_pc_path: str | None,
) -> ad.AnnData | None:
    """Registers the fixed effect matrix for the given pseudo-bulked data

    Parameters
    ----------
        `pbdata: ad.AnnData`
            The `ad.AnnData` object holding the pseudo-bulked single cell data

    Returns
    -------
        `ad.AnnData` containing with the updated `obsm` with the fixed effects
    """
    # filtering cells
    scdata_cell = scdata[scdata.obs[cell_type_key_in_scdata] == cell_type]
    # early return if no cells left
    if scdata_cell.shape[0] == 0:
        logger.info(f"No cells found for the given cell type {cell_type} ({scdata_cell.shape=})")
        return None
    # filtering chromosomes
    scdata_cell = scdata_cell[:, scdata_cell.var["chrom"] == target_chromosome]
    if scdata_cell.shape[1] == 0:
        logger.info(f"No genes found for the given chromosome {target_chromosome} ({scdata_cell.shape=})")
        return None
    # pseudobulk aggregation
    pbdata = _pseudobulk_scdata(
        scdata_cell,
        donor_key_in_scdata,
        sex_key_in_scdata,
        age_key_in_scdata,
        pseudobulk_aggregation_type,
    )
    # filter out genes least expressed genes
    sc.pp.filter_genes(pbdata, min_cells=min_individuals_threshold)
    # retrieving the wanted number of sc PCs
    if "X_pca" not in pbdata.obsm.keys():
        sc.tl.pca(pbdata, n_comps=n_cellstate_comps)
    pbdata.obsm["X_pca"] = _column_normalize(pbdata.obsm["X_pca"][:, :n_cellstate_comps])
    # early return if no genes left
    if scdata_cell.shape[1] == 0:
        logger.info(f"No genes found in more than {min_individuals_threshold} individuals ({scdata_cell.shape=})")
        return None
    # registering fixed effects
    pbdata = _register_fixed_effects(
        pbdata,
        gdata,
        n_top_genes,
        n_epcs,
        n_gpcs,
        sex_key_in_scdata,
        age_key_in_scdata,
        gen_pc_path,
    )
    # synchronizing sc and genetics data using DonorData DS
    data = DonorData(adata=pbdata, gdata=gdata, donor_key_in_sc_adata=donor_key_in_scdata)
    return data


def _prepare_gwas_data(
    pb_data: DonorData, target_gene: str, target_chromosome: str, cis_window: int
) -> Sequence[np.ndarray] | None:
    """Prepares the data used to run GWAS on

    Parameters
    ----------
        `pb_data: DonorData`
            Donor Data containing the pseudo bulked data for the current cell type and chromosome
        `target_gene: str`
            Target gene which to run GWAS on
        `target_chromosome: str`
            Target chromosome which to run GWAS on
        `cis_window: int`
            The window for retrieving the neighboring variants
        `transforms: Callable | None`
            The transformation to be applied to the input data before estimating the linear model

    Returns
    -------
        `Y: np.ndarray`
            Array containing the input data to be fed to the linear model
        `F: np.ndarray`
            Array containing the data for the fixed effects
        `G: np.ndarray`
            Array containing the genetics data for the variants falling within the window
    """
    # retrieving the pseudo-bulked data
    Y = pb_data.adata[:, [target_gene]].layers["transformed_mean"]
    # retrieving start and end position for each gene
    start = pb_data.adata.var.loc[target_gene].start
    end = pb_data.adata.var.loc[target_gene].end
    # retrieving the variants within the cis window
    subgadata = pb_data.gdata[
        :,
        (pb_data.gdata.var.chrom == target_chromosome)
        & (pb_data.gdata.var.pos >= start - cis_window)
        & (pb_data.gdata.var.pos <= end + cis_window),
    ]
    # early return if no cis_snip_found
    if subgadata.shape[1] == 0:
        return None
    G = subgadata.X.compute()
    F = pb_data.adata.obsm["F"]
    if isinstance(F, DaskArray):
        F = F.compute()
    return Y, F, G


def _parse_gwas_results(gwas: GWAS) -> Sequence[np.ndarray]:
    """Parses the results of a ran GWAS and cleanes it from potential nan or infinity values

    Parameters
    ----------
        `gwas: GWAS`
            The ran gwas experiment which to retrieve the results from

    Returns
    -------
        `pv: np.ndarray`
            The array containing the p-values for each variant
        `betasnp: np.ndarray`
            The array containing the coefficient of the LM for each variant
        `betasnp_ste: np.ndarray`
            The array containing the standard error of coefficient of the LM for each variant
        `lrt: np.ndarray`
            The array containing the results of the likelihood ration test for each variant
    """
    # retrieving the results
    pv = gwas.getPv()
    betasnp = gwas.getBetaSNP()
    betasnp_ste = gwas.getBetaSNPste()
    lrt = gwas.getLRT()
    # retrieving gwas results
    pv = np.squeeze(pv)
    betasnp = np.squeeze(betasnp)
    betasnp_ste = np.squeeze(betasnp_ste)
    lrt = np.squeeze(lrt)
    # removing nan
    pv[np.isnan(pv)] = 1
    betasnp[np.isnan(betasnp)] = 0
    betasnp_ste[np.isnan(betasnp_ste)] = 0
    lrt[np.isnan(lrt)] = 0
    # removing infinity
    pv[np.isinf(pv)] = 1
    betasnp[np.isinf(betasnp)] = 0
    betasnp_ste[np.isinf(betasnp_ste)] = 0
    lrt[np.isinf(lrt)] = 0
    return pv, betasnp, betasnp_ste, lrt


def _apply_transforms_seq(Y: np.ndarray, transforms_seq: Sequence[Callable] | None = None, **kwargs) -> np.ndarray:
    """Applies in order the transformations defined in `transform_seq` to the input array

    Parameters
    ----------
        `Y: np.ndarray`
            Array containing the pseudo-bulked single cell data

    Returns
    -------
        `np.ndarray`
            Array with the transformed pseudo-bulked single cell data
    """
    Y_transformed = Y.copy()
    if transforms_seq is not None:
        for transform in transforms_seq:
            if transform is not None:
                Y_transformed = transform(Y_transformed, **kwargs)
    return Y_transformed


def _apply_transforms_map(
    pv: np.ndarray, transforms_map: Sequence[Callable] | None = None, **kwargs
) -> dict[str, np.ndarray]:
    """Applies the transformations defined in `transfroms_map` and stores them in a dictionary

    Parameters
    ----------
        `pc: np.ndarray`
            Array containing the computed p-values for each variant

    Returns
    -------
        `dict[str, np.ndarray]`
            Dictionary mapping names (under the form of strings) to Arrays with the transformed p-value
    """
    pv_transformed_results = {}
    if transforms_map is not None:
        for pv_transform_id, pv_transform_fn in transforms_map.items():
            pv_transformed = pv.copy()
            if pv_transform_fn is not None:
                pv_transformed_results[pv_transform_id] = pv_transform_fn(pv_transformed, **kwargs)
    return pv_transformed_results


def _best_eqtl(
    pb_data: DonorData,
    target_cell_type: str,
    target_chromosome: str,
    target_gene: str,
    gwas: GWAS,
    no_tested_variants: int,
    pv_transforms: Mapping[str, Callable],
) -> Sequence[dict[str, float]]:
    """Postprocesses the GWAS results to report only the best variants in terms of p-value

    Parameters
    ----------
        `pb_data: DonorData`
            Donor Data containing the pseudo bulked data for the current cell type and chromosome
        `target_cell_type: str`
            Target cell type which GWAS experiment was ran on
        `target_chromosome: str`
            Target chromosome which GWAS experiment was ran on
        `target_gene: str`
            Target gene which GWAS experiment was ran on
        `gwas: GWAS`
            The ran gwas experiment which to retrieve the results from
        `no_tested_variants: int`
            The number of tested variants for the current combination of (`target_cell_type`, `target_chromosome`, `target_gene`)

    Returns
    -------
        `Sequence[dict[str, float]]`
            The output data with the parsed statistics to be stored (only for best variant in terms of p-value)
    """
    # retrieving gwas results
    pv, betasnp, betasnp_ste, lrt = _parse_gwas_results(gwas)
    # transforming the pvalues
    pv_transformed = _apply_transforms_map(pv, pv_transforms, no_tested_variants=no_tested_variants)
    # retrieving the results for the variant with the lowest p-value
    min_pv = pv.min()
    min_pv_idx = pv.argmin()
    min_pv_variant = pb_data.gdata.var.index[min_pv_idx]
    min_pv_variant_beta = betasnp[min_pv_idx]
    min_pv_variant_beta_ste = betasnp_ste[min_pv_idx]
    min_pv_variant_lrt = lrt[min_pv_idx]
    # constructing output dictionary
    out_dict = {
        "cell_type": target_cell_type,
        "chrom": target_chromosome,
        "gene": target_gene,
        "no_tested_variants": no_tested_variants,
        "min_pv": min_pv,
        "min_pv_variant": min_pv_variant,
        "min_pv_variant_beta": min_pv_variant_beta,
        "min_pv_variant_beta_ste": min_pv_variant_beta_ste,
        "min_pv_variant_lrt": min_pv_variant_lrt,
        **pv_transformed,
    }
    return [out_dict]


def _all_eqtls(
    pb_data: DonorData,
    target_cell_type: str,
    target_chromosome: str,
    target_gene: str,
    gwas: GWAS,
    no_tested_variants: int,
    pv_transforms: Mapping[str, Callable],
) -> Sequence[dict[str, float]]:
    """Postprocesses the GWAS results to report all variants

    Parameters
    ----------
        `pb_data: DonorData`
            Donor Data containing the pseudo bulked data for the current cell type and chromosome
        `target_cell_type: str`
            Target cell type which GWAS experiment was ran on
        `target_chromosome: str`
            Target chromosome which GWAS experiment was ran on
        `target_gene: str`
            Target gene which GWAS experiment was ran on
        `gwas: GWAS`
            The ran gwas experiment which to retrieve the results from
        `no_tested_variants: int`
            The number of tested variants for the current combination of (`target_cell_type`, `target_chromosome`, `target_gene`)

    Returns
    -------
        `Sequence[dict[str, float]]`
            The output data with the parsed statistics to be stored
    """
    # retrieving gwas results
    pv, betasnp, betasnp_ste, lrt = _parse_gwas_results(gwas)
    # transforming the pvalues
    pv_transformed = _apply_transforms_map(pv, pv_transforms, no_tested_variants=no_tested_variants)
    # defining the output object
    results = [
        {
            "cell_type": target_cell_type,
            "chrom": target_chromosome,
            "gene": target_gene,
            "no_tested_variants": no_tested_variants,
            "pv": pv[idx],
            "variant": pb_data.gdata.var.index[idx],
            "betasnp": betasnp[idx],
            "betasnp_ste": betasnp_ste[idx],
            "lrt": lrt[idx],
            **{transform_id: transformed_pv[idx] for transform_id, transformed_pv in pv_transformed.items()},
        }
        for idx in range(no_tested_variants)
    ]
    return results


def _gwas(
    pb_data: DonorData,
    target_cell_type: str,
    target_chromosome: str,
    target_gene: str,
    cis_window: int,
    pv_transforms: Mapping[str, Callable],
    mode: str,
) -> Sequence[dict[str, float | str]]:
    """"""
    # preparing gwas data
    gwas_data = _prepare_gwas_data(pb_data, target_gene, target_chromosome, cis_window)
    if gwas_data is None:
        logger.info(
            f"No cis snips found for {target_cell_type=}, {target_chromosome=}, {target_gene=} for cis window of {cis_window}"
        )
        return [
            {
                "cell_type": target_cell_type,
                "chrom": target_chromosome,
                "gene": target_gene,
                "no_tested_variants": 0,
            }
        ]
    Y, F, G = gwas_data
    # retrieving the no of cis snips
    no_cis_snps = G.shape[1]
    # processing the found snips
    gwas = GWAS(Y, F=F)
    gwas.process(G)
    # retrieving the results
    if mode == "best":
        results = _best_eqtl(
            pb_data, target_cell_type, target_chromosome, target_gene, gwas, no_cis_snps, pv_transforms
        )
    elif mode == "all":
        results = _all_eqtls(
            pb_data, target_cell_type, target_chromosome, target_gene, gwas, no_cis_snps, pv_transforms
        )
    else:
        raise ValueError(f"{mode=} not supported, try either 'best' or 'all'")
    return results


def _run_eqtl(
    donor_data: DonorData,
    target_cell_type: str,
    target_chromosome: str,
    target_genes: Sequence[str] | None,
    donor_key_in_scdata: str,
    sex_key_in_scdata: str,
    age_key_in_scdata: str,
    cell_type_key_in_scdata: str,
    pseudobulk_aggregation_type: str,
    min_individuals_threshold: int,
    n_top_genes: int,
    n_epcs: int,
    n_gpcs: int,
    n_cellstate_comps: int,
    cis_window: int,
    transforms_seq: Sequence[Callable],
    pv_transforms: Mapping[str, Callable],
    mode: str,
    prog_bar: bool,
    dump_intermediate_results: bool,
    file_prefix: str | None,
    dump_dir: str | None,
    gen_pc_path: str | None,
) -> Sequence[dict[str, float]]:
    """Runs the EQTL pipeline on a given pair of (`target_cell_type`, `target_chromosome`) over all genes

    Parameters
    ----------
        `target_cell_type: str`
            Target chromosome which GWAS experiment was ran on
        `target_chromosome: str`
            Target chromosome which GWAS experiment was ran on
        `cis_window: int`
            The window used for running the GWAS experiment

    Returns
    -------
        `Sequence[dict[str, float]]`
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
        cell_type_key_in_scdata,
        pseudobulk_aggregation_type,
        min_individuals_threshold,
        n_top_genes,
        n_epcs,
        n_gpcs,
        n_cellstate_comps,
        gen_pc_path,
    )
    # retrieving the pseudo-bulked data
    Y = pb_data.adata.X
    # defining transform function
    transform_fn = partial(_apply_transforms_seq, transforms_seq=transforms_seq)
    # applying transformation
    Y = asarray(Y)
    if transform_fn is not None:
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
        gwas_out = _gwas(
            pb_data,
            target_cell_type,
            target_chromosome,
            target_gene,
            cis_window,
            pv_transforms,
            mode,
        )
        output = [*output, *gwas_out]

        if dump_intermediate_results:
            gwas_results_df = pd.DataFrame(gwas_out)
            gwas_postprocessed_dfs = _postprocess_results(gwas_results_df)
            _dump_results(
                gwas_results_df,
                gwas_postprocessed_dfs,
                cis_window,
                target_cell_type,
                target_chromosome,
                target_gene,
                file_prefix,
                dump_dir,
            )

        # updating the iterator
        if iterator is not None:
            iterator.update()
    return output


def _postprocess_results(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """"""
    return None


def quantile_transform(
    x: np.ndarray,
    seed: int = 1,
) -> np.ndarray:
    """
    Gaussian quantile transform for values in a pandas Series.    :param x: Input pandas Series.
    :type x: pd.Series
    :param seed: Random seed.
    :type seed: int
    :return: Transformed Series.
    :rtype: pd.Series    .. note::
        “nan” values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()
    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)
    x_transform[~is_nan] = sk_quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]
    return x_transform


def bonferroni_adjustment(
    pv: np.ndarray,
    no_tested_variants: int,
) -> np.ndarray:
    """"""
    return np.clip(pv * no_tested_variants, 0, 1)


def q_value(pv: np.ndarray, no_tested_variants: int) -> np.ndarray:
    """"""
    bf_pv = bonferroni_adjustment(pv, no_tested_variants)
    return fdrcorrection(bf_pv)[1]


def _dump_results(
    results_df: pd.DataFrame,
    postprocessed_dfs: dict[str, pd.DataFrame] | None,
    cis_window: int,
    target_cell_type: str,
    target_chromosome: str,
    target_gene: str | None,
    file_prefix: str | None,
    dump_dir: str | None,
) -> None:
    """"""
    # setting default file prefix and dump path if not defined
    if file_prefix is None:
        file_prefix = "EQTL"
    if dump_dir is None:
        dump_dir = "./"
    # defining the path and saving the model
    dump_path = Path(dump_dir) / f"{file_prefix}_CellType{target_cell_type}_Chrom{target_chromosome}.csv"
    dump_path = Path(dump_dir) / f"{file_prefix}_CellType{target_cell_type}_Chrom{target_chromosome}.csv"
    if target_gene is not None:
        dump_path = (
            Path(dump_dir) / f"{file_prefix}_CellType{target_cell_type}_Chrom{target_chromosome}_Gene{target_gene}.csv"
        )
    results_df.to_csv(dump_path, index=False)
    # saving post processed results df to disk
    if postprocessed_dfs is not None:
        for post_processing_id, post_processed_df in postprocessed_dfs.items():
            dump_file = f"{file_prefix}_PP{post_processing_id}_CellType{target_cell_type}_Chrom{target_chromosome}.csv"
            if target_gene is not None:
                dump_file = f"{file_prefix}_PP{post_processing_id}_CellType{target_cell_type}_Chrom{target_chromosome}_Gene{target_gene}.csv"
            dump_path = Path(dump_dir) / dump_file
            post_processed_df.to_csv(dump_path, index=False)


def eqtl(
    donor_data: DonorData,
    target_cell_type: str,
    target_chromosome: str,
    target_genes: Sequence[str] | None = None,
    donor_key_in_scdata: str = "individual",
    sex_key_in_scdata: str = "sex",
    age_key_in_scdata: str = "age",
    cell_type_key_in_scdata: str = "cell_label",
    pseudobulk_aggregation_type: str = "mean",
    min_individuals_threshold: int = 10,
    n_top_genes: int = 5_000,
    n_epcs: int = 15,
    n_gpcs: int = 20,
    n_cellstate_comps: int = 50,
    cis_window: int = 1_000_000,
    transforms_seq: Sequence[Callable] | None = (quantile_transform,),
    pv_transforms: Mapping[str, Callable] | None = None,
    mode: str = "all",
    prog_bar: bool = True,
    all_genes: bool = True,
    dump_results: bool = True,
    dump_dir: str | None = None,
    file_prefix: str | None = None,
    use_cell_type_chrom_specific_dir: bool = True,
    dump_intermediate_results: bool = False,
    gen_pc_path: str | None = None,
) -> Sequence[dict[str, float]]:
    """Runs the EQTL pipeline on a given pair of (`target_cell_type`, `target_chromosome`) over all genes and stores the results to a `pd.DataFrame` object and optionally to disk

    Parameters
    ----------
        `target_cell_type: str`
            Target chromosome which GWAS experiment was ran on
        `target_chromosome: str`
            Target chromosome which GWAS experiment was ran on
        `cis_window: int`
            The window used for running the GWAS experiment

    Returns
    -------
        `pd.DataFrame`
            The output data in a `pd.DataFrame` object
    """
    # ensuring the type of target chromose is a string
    # TODO: Understand why we need to do this when loading confs with hydra
    if isinstance(target_chromosome, int):
        target_chromosome = str(target_chromosome)
    # optionally creating a directory to store the results for
    # the current `target_cell_type`, `target_chromosome` combination
    dump_dir_cell = None
    if use_cell_type_chrom_specific_dir:
        dump_dir_base = "./" if dump_dir is None else dump_dir
        dump_dir_cell = Path(dump_dir_base) / f"Cell{target_cell_type}_Chrom{target_chromosome}"
        dump_dir_cell.mkdir(exist_ok=True)
        dump_dir_cell = str(dump_dir_cell)
    # running the pipeline and constructing results DataFrame
    results = _run_eqtl(
        donor_data,
        target_cell_type,
        target_chromosome,
        target_genes,
        donor_key_in_scdata,
        sex_key_in_scdata,
        age_key_in_scdata,
        cell_type_key_in_scdata,
        pseudobulk_aggregation_type,
        min_individuals_threshold,
        n_top_genes,
        n_epcs,
        n_gpcs,
        n_cellstate_comps,
        cis_window,
        transforms_seq,
        pv_transforms,
        mode,
        prog_bar,
        dump_intermediate_results,
        file_prefix,
        dump_dir_cell,
        gen_pc_path,
    )
    results_df = pd.DataFrame(results)
    # postprocessing the results
    postprocessed_dfs = _postprocess_results(results_df)
    # optionally saving the results to disk
    if dump_results:
        logger.warning(f"{dump_dir=} {file_prefix=}")
        _dump_results(
            results_df,
            postprocessed_dfs,
            cis_window,
            target_cell_type,
            target_chromosome,
            None,
            file_prefix,
            dump_dir,
        )
    # constructing out dictionary
    return results_df, postprocessed_dfs
