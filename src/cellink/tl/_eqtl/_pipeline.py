import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from anndata.utils import asarray
from tqdm import tqdm

from cellink._core import DonorData
from cellink.tl._eqtl._data import EQTLDataManager
from cellink.tl._eqtl._gwas import GWAS
from cellink.tl._eqtl._utils import quantile_transform

logger = logging.getLogger(__name__)

__all__ = [
    "EQTLPipeline",
]


@dataclass
class EQTLPipeline:
    """Class handling the EQTL Pipeline

    Parameters
    ----------
        `data: EQTLDataManager`
            Data Manager for the EQTL studies
        `cis_window: int`
            Window used to considering the variants as neighboring to a gene
        `transforms: Sequence[Callable] | None`
            The transformations applied to the pseudo-bulked single cell data before estimating the linear model (Optional, defaults to `cellink.tl._eqtl._utils.quantile_trasform`).
        `pv_transforms: Sequence[Callable] | None`
            The transformations applied to the computed pvalues before creating the tables (Optional, defaults to `None`).
        `mode: Literal["best", "all"]`
            The mode in which to run the eqtl pipeline ("best": report only best variants in terms of p-value, "all": report all variants, defaults to `"all"`)
        `dump_results: bool`
            Whether to dump results to an output CSV file (defaults to `False`).
        `dump_dir: str | None`
            The directory where to optionally dump the files (defaults to the current working directory if not provided).
        `file_prefix: str`
            The prefix to the file name used to save the results to disk (defaults to `"eqtl"`).
    """

    data: EQTLDataManager
    cis_window: int = 1_000_000
    transforms: Sequence[Callable] | None = (quantile_transform,)
    pv_transforms: dict[str, Callable] | None = None
    mode: Literal["best", "all"] = "all"
    dump_results: bool = False
    dump_dir: str | None = None
    file_prefix: str = "eqtl"
    prog_bar: bool = True

    @staticmethod
    def _prepare_gwas_data(
        pb_data: DonorData, target_gene: str, target_chrom: str, cis_window: int, transforms: Callable | None = None
    ) -> Sequence[np.ndarray]:
        """Prepares the data used to run GWAS on

        Parameters
        ----------
            `pb_data: DonorData`
                Donor Data containing the pseudo bulked data for the current cell type and chromosome
            `target_gene: str`
                Target gene which to run GWAS on
            `target_chrom: str`
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
        ## retrieving the pseudo-bulked data
        Y = pb_data.adata[:, [target_gene]].layers["mean"]
        Y = asarray(Y)
        if transforms is not None:
            Y = transforms(Y)
        ## retrieving start and end position for each gene
        start = pb_data.adata.var.loc[target_gene].start
        end = pb_data.adata.var.loc[target_gene].end
        chrom = pb_data.adata.var.loc[target_gene].chrom
        ## retrieving the variants within the cis window
        subgadata = pb_data.gdata[
            :,
            (pb_data.gdata.var.chrom == target_chrom)
            & (pb_data.gdata.var.pos >= start - cis_window)
            & (pb_data.gdata.var.pos <= end + cis_window),
        ]
        G = subgadata.X.compute()
        F = pb_data.adata.obsm["F"]
        return Y, F, G

    @staticmethod
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
        ## retrieving gwas results
        pv = np.squeeze(gwas.getPv())
        betasnp = np.squeeze(gwas.getBetaSNP())
        betasnp_ste = np.squeeze(gwas.getBetaSNPste())
        lrt = np.squeeze(gwas.getLRT())
        ## removing nan
        pv[np.isnan(pv)] = 1
        betasnp[np.isnan(betasnp)] = 0
        betasnp_ste[np.isnan(betasnp_ste)] = 0
        lrt[np.isnan(lrt)] = 0
        ## removing infinity
        pv[np.isinf(pv)] = 1
        betasnp[np.isinf(betasnp)] = 0
        betasnp_ste[np.isinf(betasnp_ste)] = 0
        lrt[np.isinf(lrt)] = 0
        return pv, betasnp, betasnp_ste, lrt

    def _best_eqtl(
        self,
        pb_data: DonorData,
        target_cell_type: str,
        target_chrom: str,
        target_gene: str,
        gwas: GWAS,
        no_tested_variants: int,
    ) -> Sequence[dict[str, float]]:
        """Postprocesses the GWAS results to report only the best variants in terms of p-value

        Parameters
        ----------
            `pb_data: DonorData`
                Donor Data containing the pseudo bulked data for the current cell type and chromosome
            `target_cell_type: str`
                Target cell type which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on
            `target_gene: str`
                Target gene which GWAS experiment was ran on
            `gwas: GWAS`
                The ran gwas experiment which to retrieve the results from
            `no_tested_variants: int`
                The number of tested variants for the current combination of (`target_cell_type`, `target_chrom`, `target_gene`)

        Returns
        -------
            `Sequence[dict[str, float]]`
                The output data with the parsed statistics to be stored (only for best variant in terms of p-value)
        """
        ## retrieving gwas results
        pv, betasnp, betasnp_ste, lrt = self._parse_gwas_results(gwas)
        ## transforming the pvalues
        pv_transformed = self._apply_pv_transforms(pv, no_tested_variants)
        ## retrieving the results for the variant with the lowest p-value
        min_pv = pv.min()
        min_pv_idx = pv.argmin()
        min_pv_variant = pb_data.gdata.var.index[min_pv_idx]
        min_pv_variant_beta = betasnp[min_pv_idx]
        min_pv_variant_beta_ste = betasnp_ste[min_pv_idx]
        min_pv_variant_lrt = lrt[min_pv_idx]
        ## constructing output dictionary
        out_dict = {
            "cell_type": target_cell_type,
            "chrom": target_chrom,
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
        self,
        pb_data: DonorData,
        target_cell_type: str,
        target_chrom: str,
        target_gene: str,
        gwas: GWAS,
        no_tested_variants: int,
    ) -> Sequence[dict[str, float]]:
        """Postprocesses the GWAS results to report all variants

        Parameters
        ----------
            `pb_data: DonorData`
                Donor Data containing the pseudo bulked data for the current cell type and chromosome
            `target_cell_type: str`
                Target cell type which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on
            `target_gene: str`
                Target gene which GWAS experiment was ran on
            `gwas: GWAS`
                The ran gwas experiment which to retrieve the results from
            `no_tested_variants: int`
                The number of tested variants for the current combination of (`target_cell_type`, `target_chrom`, `target_gene`)

        Returns
        -------
            `Sequence[dict[str, float]]`
                The output data with the parsed statistics to be stored
        """
        ## retrieving gwas results
        pv, betasnp, betasnp_ste, lrt = self._parse_gwas_results(gwas)
        ## transforming the pvalues
        pv_transformed = self._apply_pv_transforms(pv, no_tested_variants)
        ## defining the output object
        results = [
            {
                "cell_type": target_cell_type,
                "chrom": target_chrom,
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

    def _apply_transforms(self, Y: np.ndarray) -> np.ndarray:
        """Applies in order the transformations defined in `self.transforms` to the pseudo-bulked single cell data

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
        if self.transforms is not None:
            for transform in self.transforms:
                if transform is not None:
                    Y_transformed = transform(Y_transformed)
        return Y_transformed

    def _apply_pv_transforms(self, pv: np.ndarray, no_tested_variants: int) -> dict[str, np.ndarray]:
        """Applies the transformations on the computed p-values and stores them in a dictionary

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
        if self.pv_transforms is not None:
            for pv_transform_id, pv_transform_fn in self.pv_transforms.items():
                pv_transformed = pv.copy()
                if pv_transform_fn is not None:
                    pv_transformed_results[pv_transform_id] = pv_transform_fn(pv_transformed, no_tested_variants)
        return pv_transformed_results

    def _fit_gwas(self, pb_data: DonorData, target_gene: str, target_chrom: str) -> Sequence[GWAS, int] | None:
        """Runs the GWAS experiment on the given gene and chromosomes for the passed window

        Parameters
        ----------
            `pb_data: DonorData`
                Donor Data containing the pseudo bulked data for the current cell type and chromosome
            `target_gene: str`
                Target gene which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on

        Returns
        -------
            `GWAS`
                A `GWAS` object with the ran experiment
            `int`
                The number of tested variant for the current iteration
        """
        ## preparing gwas data
        Y, F, G = self._prepare_gwas_data(pb_data, target_gene, target_chrom, self.cis_window, self._apply_transforms)
        ## early return if no cis snips found
        if G.shape[1] == 0:
            return None
        gwas = GWAS(Y, F=F)
        ## processing the found snips
        gwas.process(G)
        return gwas, G.shape[1]

    def _run_gwas(
        self,
        pb_data: DonorData,
        target_cell_type: str,
        target_chrom: str,
        target_gene: str,
        gwas: GWAS,
        no_tested_variants: int,
    ) -> Sequence[dict[str, float | str]]:
        """Postprocesses the GWAS results to report either all variants or only the best ones in terms of p-value

        Parameters
        ----------
            `pb_data: DonorData`
                Donor Data containing the pseudo bulked data for the current cell type and chromosome
            `target_cell_type: str`
                Target cell type which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on
            `target_gene: str`
                Target gene which GWAS experiment was ran on
            `gwas: GWAS`
                The ran gwas experiment which to retrieve the results from
            `no_tested_variants: int`
                The number of tested variants for the current combination of (`target_cell_type`, `target_chrom`, `target_gene`)

        Returns
        -------
            `Sequence[dict[str, float]]`
                The output data with the parsed statistics to be stored
        """
        if self.mode == "best":
            return self._best_eqtl(pb_data, target_cell_type, target_chrom, target_gene, gwas, no_tested_variants)
        elif self.mode == "all":
            return self._all_eqtls(pb_data, target_cell_type, target_chrom, target_gene, gwas, no_tested_variants)
        else:
            raise ValueError(f"{self.mode=} not supported, try either 'best' or 'all'")

    def _gwas(
        self, pb_data: DonorData, target_cell_type: str, target_chrom: str, target_gene: str
    ) -> Sequence[dict[str, float | str]]:
        """"""
        ## fitting the gwas on gene
        gwas_out = self._fit_gwas(pb_data, target_gene, target_chrom)
        ## early return if no cis snip is found
        if gwas_out is None:
            return []
        ## parsing the output results
        (gwas, no_tested_variants) = gwas_out
        ## retrieving the results the results
        results = self._run_gwas(pb_data, target_cell_type, target_chrom, target_gene, gwas, no_tested_variants)
        return results

    def _run_pipeline(self, target_cell_type: str, target_chrom: str) -> Sequence[dict[str, float]]:
        """Runs the EQTL pipeline on a given pair of (`target_cell_type`, `target_chrom`) over all genes

        Parameters
        ----------
            `target_cell_type: str`
                Target chromosome which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on
            `cis_window: int`
                The window used for running the GWAS experiment

        Returns
        -------
            `Sequence[dict[str, float]]`
                The output data with the parsed statistics to be stored for all genes
        """
        ## output results
        output = []
        ## retrieving tmp data for current cell_type and chromosome
        pb_data = self.data.get_pb_data(target_cell_type, target_chrom)
        ## early return if pseudo bulked data is None
        if pb_data is None:
            return output
        ## retrieving current genes
        current_genes = pb_data.adata.var_names
        ## defining optional iterator
        iterator = tqdm(range(len(current_genes))) if self.prog_bar else None
        ## iterating over the genes to test
        for idx, target_gene in enumerate(current_genes):
            ## running the gwas on gene
            results = self._gwas(pb_data, target_cell_type, target_chrom, target_gene)
            ## storing the results for the current gene
            output += results
            ## updating the iterator
            if iterator is not None:
                iterator.update()
        return output

    def _postprocess_results(self, results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """"""
        return None

    def run(self, target_cell_type: str, target_chrom: str, postprocess_results: bool = False):
        """Runs the EQTL pipeline on a given pair of (`target_cell_type`, `target_chrom`) over all genes and
        stores the results to a `pd.DataFrame` object and optionally to disk

        Parameters
        ----------
            `target_cell_type: str`
                Target chromosome which GWAS experiment was ran on
            `target_chrom: str`
                Target chromosome which GWAS experiment was ran on
            `cis_window: int`
                The window used for running the GWAS experiment

        Returns
        -------
            `pd.DataFrame`
                The output data in a `pd.DataFrame` object
        """
        ## ensuring the type of target chromose is a string
        ## TODO: Understand why we need to do this when loading confs with hydra
        if isinstance(target_chrom, int):
            target_chrom = str(target_chrom)
        ## running the pipeline and constructing results DataFrame
        results = self._run_pipeline(target_cell_type, target_chrom)
        results_df = pd.DataFrame(results)
        ## postprocessing the results
        postprocessed_dfs = self._postprocess_results(results_df)
        ## optionally saving the results to disk
        if self.dump_results:
            dump_dir = "./" if self.dump_dir is None else self.dump_dir
            dump_path = Path(dump_dir) / f"{self.file_prefix}_{target_cell_type}_{target_chrom}_{self.cis_window}.csv"
            results_df.to_csv(dump_path, index=False)
            ## saving post processed results df to disk
            if postprocessed_dfs is not None:
                for post_processing_id, post_processed_df in postprocessed_dfs.items():
                    dump_path = (
                        Path(dump_dir)
                        / f"{self.file_prefix}_{post_processing_id}_{target_cell_type}_{target_chrom}_{self.cis_window}.csv"
                    )
                    post_processed_df.to_csv(dump_path, index=False)
        ## constructing out dictionary
        return results_df, postprocessed_dfs
