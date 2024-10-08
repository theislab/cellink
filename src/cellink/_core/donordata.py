import logging
from dataclasses import dataclass

from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass
class DonorData:
    """
    A class to store and manage donor-related data with single-cell readouts to allow donor-level, especially genetic, analysis with single-cell datasets.

    This class holds AnnData objects for single-cell (adata) and genetic (gdata) data.

    Attributes
    ----------
        adata (AnnData): Single-cell data in AnnData format.
        gdata (AnnData): Genetic data in AnnData format.
        donor_key_in_sc (str): Column name in adata.obs that identifies donors.

    Raises
    ------
        ValueError: If the specified donor_key_in_sc is not found in adata.obs.
    """

    adata: AnnData
    gdata: AnnData
    donor_key_in_sc: str

    def __post_init__(self):
        self._validate_data()

    def _validate_data(self):
        """
        Validates that the donor key exists in the single-cell data.

        Raises
        ------
            ValueError: If the donor_key_in_sc is not found in adata.obs.
        """
        if self.donor_key_in_sc not in self.adata.obs.columns:
            raise ValueError(f"'{self.donor_key_in_sc}' not found in adata.obs")

    def get_donor_adata(self, donor: str) -> AnnData | None:
        """
        Retrieves single-cell data for a specific donor.

        Args:
            donor (str): The name of the donor.

        Returns
        -------
            Optional[AnnData]: AnnData object containing the donor's single-cell data,
                               or None if the donor is not found.
        """
        if donor not in self.adata.obs[self.donor_key_in_sc].unique():
            return None
        return self.adata[self.adata.obs[self.donor_key_in_sc] == donor]

    def get_donor_gdata(self, donor: str) -> AnnData | None:
        """
        Retrieves genetic data for a specific donor.

        Args:
            donor (str): The name of the donor.

        Returns
        -------
            Optional[AnnData]: AnnData object containing the donor's genetic data,
                               or None if the donor is not found.
        """
        if donor not in self.gdata.obs_names:
            return None
        return self.gdata[donor]
