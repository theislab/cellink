import logging
from collections.abc import Sequence
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from cellink._core import DonorData

logger = logging.getLogger(__name__)

__all__ = ["EQTLDataManager"]


@dataclass
class EQTLDataManager:
    """Data Manager for EQTL pipeline.

    Parameters
    ----------
        `donor_data: DonorData`
            A `DonorData` instance holding the data to run the experiments on.
        `n_sc_comps: int`
            The number of principal components for the single cell data (defaults to 500).
        `donor_key_in_scdata: str`
            The column name in `self.donor_data.obs` containing the information about the donor (defaults to `"individual"`).
        `sex_key_in_scdata: str`
            The column name in `self.donor_data.obs` containing the information about the donor's sex (defaults to `"sex"`).
        `age_key_in_scdata: str`
            The column name in `self.donor_data.obs` containing the information about the donor's age (defaults to `"age"`).
        `pseudobulk_aggregation_type`
            The type of aggregation used for pseudo-bulking the single cell data (defaults to `"mean"`).
        `n_top_genes: int`
            The number of highly variable genes used to compute RNA expression PCs (defaults to 5000).
        `min_individuals_threshold: int`
            The threshold of the number of individuals for filtering out genes (defaults to 10).

    """

    donor_data: DonorData
    n_sc_comps: int = 500
    n_genetic_pcs: int = 500
    donor_key_in_scdata: str = "individual"
    sex_key_in_scdata: str = "sex"
    age_key_in_scdata: str = "age"
    pseudobulk_aggregation_type: str = "mean"
    n_top_genes: int = 5000
    min_individuals_threshold: int = 10

    @staticmethod
    def _column_normalize(X: np.ndarray) -> np.ndarray:
        """"""
        assert X.ndim == 2
        return (X - X.mean(0)) / (X.std(0) * np.sqrt(X.shape[1]))

    @staticmethod
    def _filter_cells_by_type(scdata: ad.AnnData, cell_type: str) -> ad.AnnData:
        """Filters cells by cell type

        Parameters
        ----------
            `scdata: ad.AnnData`
                The `ad.AnnData` object holding the single cell data
            `cell_type: str`
                The cell type to retrieve for the current iteration

        Returns
        -------
            `ad.AnnData` containing only the cells of the given type type
        """
        scdata_cell = scdata[scdata.obs.cell_label == cell_type]
        return scdata_cell

    @staticmethod
    def _filter_genes_by_chromosome(scdata: ad.AnnData, target_chromosome: str) -> ad.AnnData:
        """Filters genes by chromosome

        Parameters
        ----------
            `scdata: ad.AnnData`
                The `ad.AnnData` object holding the single cell data
            `target_chromosome: str`
                The target chromosome to retrieve for the current iteration

        Returns
        -------
            `ad.AnnData` containing only the genes associated to the current chromosome
        """
        scdata = scdata[:, scdata.var["chrom"] == target_chromosome]
        return scdata

    def _map_col_scdata_obs_to_pbdata(self, pbdata: ad.AnnData, column: str) -> ad.AnnData:
        """Maps the selected column, assumed to have only one unique value for each patient (i.e.: age, sex, etc.)
        from the base single cell data to the pseudo-bulked one, as not all the columns are returned after aggregation

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
        ## mapping over the individuals
        individuals = pbdata.obs[self.donor_key_in_scdata]
        reference_data = self.scdata.obs[[self.donor_key_in_scdata, column]]
        reference_data = reference_data.groupby(self.donor_key_in_scdata).agg(["unique"])

        ## function for making sure the values are unique
        def retrieve_unique_value_safe(row):
            assert len(row) == 1
            return row[0]

        ## retrieving the unique values for each donor
        reference_data[column] = reference_data[column].map(retrieve_unique_value_safe)
        ## merging the data and updating column names
        pbdata.obs = pd.merge(pbdata.obs, reference_data[column], left_on=self.donor_key_in_scdata, right_index=True)
        pbdata.obs[column] = pbdata.obs["unique"]
        pbdata.obs = pbdata.obs.drop(columns=["unique"], axis=1)
        return pbdata

    def _pseudobulk_scdata(self, scdata_cell: ad.AnnData) -> ad.AnnData:
        """Pseudobulks the single cell data

        Parameters
        ----------
            `scdata_cell: ad.AnnData`
                The `ad.AnnData` object holding the cells with the given type

        Returns
        -------
            `ad.AnnData` containing with the single cell data aggregated by patient
        """
        ## aggregating the data
        pbdata = sc.get.aggregate(
            scdata_cell,
            self.donor_key_in_scdata,
            self.pseudobulk_aggregation_type,
        )
        ## storing data lost in the aggregation
        pbdata.X = pbdata.layers["mean"]
        pbdata = self._map_col_scdata_obs_to_pbdata(pbdata, self.sex_key_in_scdata)
        pbdata = self._map_col_scdata_obs_to_pbdata(pbdata, self.age_key_in_scdata)
        return pbdata

    def _register_fixed_effects(self, pbdata: ad.AnnData) -> ad.AnnData:
        """Registers the fixed effect matrix for the given pseudo-bulked data

        Parameters
        ----------
            `pbdata: ad.AnnData`
                The `ad.AnnData` object holding the pseudo-bulked single cell data

        Returns
        -------
            `ad.AnnData` containing with the updated `obsm` with the fixed effects
        """
        ## compute expression PCs
        sc.pp.highly_variable_genes(pbdata, n_top_genes=self.n_top_genes)
        sc.tl.pca(pbdata, use_highly_variable=True, n_comps=self.n_sc_comps)
        pbdata.obsm["E_dpc"] = self._column_normalize(pbdata.obsm["X_pca"])
        ## load genetic PCs
        ## TODO: Probably this is wrong
        gen_pcs = sc.tl.pca(self.gdata.X, n_comps=self.n_genetic_pcs)
        ## load patient covariates
        sex_one_hot = np.eye(2)[(pbdata.obs[self.sex_key_in_scdata].values - 1)]
        age_standardized = StandardScaler().fit_transform(pbdata.obs[self.age_key_in_scdata].values.reshape(-1, 1))
        covariates = np.concatenate((sex_one_hot, age_standardized), axis=1)
        ## store fixed effects in pb_adata
        pbdata.obsm["F"] = np.concatenate((covariates, gen_pcs, pbdata.obsm["E_dpc"]), axis=1)
        return pbdata

    def _get_pb_data(self, cell_type: str, target_chromosome: str) -> ad.AnnData | None:
        """Registers the fixed effect matrix for the given pseudo-bulked data

        Parameters
        ----------
            `pbdata: ad.AnnData`
                The `ad.AnnData` object holding the pseudo-bulked single cell data

        Returns
        -------
            `ad.AnnData` containing with the updated `obsm` with the fixed effects
        """
        ## filtering cells
        scdata_cell = self._filter_cells_by_type(self.scdata, cell_type)
        ## early return if no cells left
        if scdata_cell.shape[0] == 0:
            logger.info(f"No cells found for the given cell type {cell_type} ({scdata_cell.shape=})")
            return None
        ## filtering chromosomes
        scdata_cell = self._filter_genes_by_chromosome(scdata_cell, target_chromosome)
        if scdata_cell.shape[1] == 0:
            logger.info(f"No genes found for the given chromosome {target_chromosome} ({scdata_cell.shape=})")
            return None
        ## pseudobulk aggregation
        pbdata = self._pseudobulk_scdata(scdata_cell)
        ## filter out genes least expressed genes
        sc.pp.filter_genes(pbdata, min_cells=self.min_individuals_threshold)
        ## early return if no genes left
        if scdata_cell.shape[1] == 0:
            logger.info(
                f"No genes found in more than {self.min_individuals_threshold} individuals ({scdata_cell.shape=})"
            )
            return None
        ## registering fixed effects
        pbdata = self._register_fixed_effects(pbdata)
        return pbdata

    def get_pb_data(self, cell_type: str, target_chromosome: str) -> DonorData | None:
        """Gets the pseudo bulked data for the given target cell type and chromosome

        Parameters
        ----------
            `cell_type: str`
                The cell type to retrieve for the current study
            `target_chromosome: str`
                The chromosome to retrieve for the current study

        Returns
        -------
            `DonorData` object with the the pseudo-bulked single cell data and the corresponding genetics data
        """
        pbdata = self._get_pb_data(cell_type, target_chromosome)
        ## early return if pseudo bulked data is None
        if pbdata is None:
            return None
        data = DonorData(adata=pbdata, gdata=self.gdata, donor_key_in_sc_adata=self.donor_key_in_scdata)
        return data

    @property
    def cell_types(self) -> Sequence[str]:
        """Gets the unique cell types in the underlying single cell data

        Returns
        -------
            `Sequence[str]` of cell type identifiers
        """
        return self.scdata.obs.cell_label.unique()

    @property
    def chroms(self) -> Sequence[str]:
        """Gets the unique chromosomes in the underlying single cell data

        Returns
        -------
            `Sequence[str]` of chromosomes identifiers
        """
        return self.scdata.var.chrom.unique()

    @property
    def genes(self) -> Sequence[str]:
        """Gets the unique genes in the underlying single cell data

        Returns
        -------
            `Sequence[str]` of gene identifiers
        """
        return self.scdata.var_names

    @property
    def scdata(self) -> ad.AnnData:
        """Gets the underlying single cell data

        Returns
        -------
            `ad.AnnData` containing the underlying single cell data
        """
        return self.donor_data.adata

    @property
    def gdata(self) -> ad.AnnData:
        """Gets the underlying genetics data

        Returns
        -------
            `ad.AnnData` containing the underlying genetics data
        """
        return self.donor_data.gdata
