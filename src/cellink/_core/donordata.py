import logging
from dataclasses import dataclass

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
    donor_key_in_sc: str

    def __post_init__(self):
        self._validate_data()

    def _validate_data(self):
        """Validates that the donor key exists in the single-cell data and is categorical.

        Raises
        ------
            ValueError: If the donor_key_in_sc is not found
            ValueError: If adata.obs[self.donor_key_in_sc] is not categorical
        """
        if self.donor_key_in_sc not in self.adata.obs.columns:
            raise ValueError(f"'{self.donor_key_in_sc}' not found in adata.obs")
        if not self.adata.obs[self.donor_key_in_sc].dtype.name == "category":
            raise ValueError(f"'{self.donor_key_in_sc}' in adata.obs is not categorical")

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
        if donor not in self.adata.obs[self.donor_key_in_sc].cat.categories:
            raise ValueError(f"Donor '{donor}' not found in adata")
        return self.adata[self.adata.obs[self.donor_key_in_sc] == donor]

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
