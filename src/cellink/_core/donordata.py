from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass
class DonorData:
    """Store and manage donor-related data with single-cell readouts.

    This class allows donor-level, especially genetic, analysis with single-cell datasets.
    It holds AnnData objects for single-cell (adata) and genetic (gdata) data.
    The donor key in adata.obs must be a categorical column.

    Raises
    ------
        ValueError: If the specified donor_key_in_sc is not found in adata.obs
        ValueError: If the specified donor_key_in_sc is not categorical
        ValueError: _description_
        ValueError: _description_

    Returns
    -------
        _type_: DonorData object
    """

    adata: AnnData
    gdata: AnnData
    donor_key_in_adata: str

    def __post_init__(self):
        self._validate_data()

        # TODO: get overlap of donors and sync data

    def _validate_data(self):
        """Validates that the donor key exists in the single-cell data and is categorical.

        Raises
        ------
            ValueError: If the donor_key_in_sc is not found
            ValueError: If adata.obs[self.donor_key_in_sc] is not categorical
        """
        if self.donor_key_in_adata not in self.adata.obs.columns:
            raise ValueError(f"'{self.donor_key_in_adata}' not found in adata.obs")
        if not self.adata.obs[self.donor_key_in_adata].dtype.name == "category":
            raise ValueError(
                f"'{self.donor_key_in_adata}' in adata.obs is not categorical"
            )

    def get_donor_adata(self, donor: str) -> AnnData:
        """Retrieve single-cell data for a specific donor.

        Args:
            donor (str): The name of the donor.

        Raises
        ------
            ValueError: If the donor is not found in the data.

        Returns
        -------
            AnnData: AnnData object containing the donor's single-cell data.
        """
        if donor not in self.adata.obs[self.donor_key_in_adata].cat.categories:
            raise ValueError(f"Donor '{donor}' not found in adata")
        return self.adata[self.adata.obs[self.donor_key_in_adata] == donor]

    def get_donor_gdata(self, donor: str) -> AnnData:
        """Retrieve genetic data for a specific donor.

        Args:
            donor (str): The name of the donor.

        Raises
        ------
            ValueError: If the donor is not found in the data.

        Returns
        -------
            AnnData: AnnData object containing the donor's genetic data.
        """
        if donor not in self.gdata.obs_names:
            raise ValueError(f"Donor '{donor}' not found in gdata")
        return self.gdata[donor]

    def _sync_data(self, valid_donors: np.ndarray):
        """
        Syncs the adata and gdata objects by filtering to only include valid donors.

        Args:
            valid_donors (np.ndarray): An array of valid donor names.
        """
        self.adata = self.adata[
            self.adata.obs[self.donor_key_in_adata].isin(valid_donors)
        ]
        self.gdata = self.gdata[self.gdata.obs_names.isin(valid_donors)]

    def slice_cells(self, cell_condition) -> DonorData:
        """
        Returns a new DonorData object with single-cell data sliced based on the provided cell condition.

        Args:
            cell_condition: A boolean mask, condition, or slice to apply to the single-cell data.

        Returns
        -------
            DonorData: A new DonorData object with sliced single-cell data.
        """
        new_adata = self.adata[cell_condition]
        valid_donors = new_adata.obs[self.donor_key_in_adata].unique()
        self._sync_data(valid_donors)

        return DonorData(new_adata, self.gdata, self.donor_key_in_adata)

    def slice_donors(self, donors: list[str]) -> DonorData:
        """
        Returns a new DonorData object with both single-cell and genetic data sliced to only include the specified donors.

        Args:
            donors (list[str]): A list of donor names to retain.

        Returns
        -------
            DonorData: A new DonorData object with sliced data.
        """
        valid_donors = np.intersect1d(
            self.adata.obs[self.donor_key_in_adata].unique(), donors
        )
        self._sync_data(valid_donors)

        return DonorData(self.adata, self.gdata, self.donor_key_in_adata)
