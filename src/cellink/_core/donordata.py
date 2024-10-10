from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from anndata import AnnData
from anndata.utils import asarray
from pandas import Index
from rich.console import Console
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


@dataclass
class DonorData:
    """Store and manage donor-related data with single-cell readouts.

    This class allows donor-level, especially genetic, analysis with single-cell datasets.
    It holds AnnData objects for single-cell (adata) and genetic (gdata) data,
    as well as gene annotation information.
    The donor key in adata.obs must be a categorical column.

    Attributes
    ----------
        adata (AnnData): Single-cell data
        gdata (AnnData): Genetic data
        donor_key_in_sc_adata (str): Key for donor information in adata.obs
        gene_annotation (pd.DataFrame): Gene annotation information

    Raises
    ------
        ValueError: If the specified donor_key_in_adata is not found in adata.obs
        ValueError: If the specified donor_key_in_adata is not categorical

    Returns
    -------
        _type_: DonorData object
    """

    adata: AnnData
    gdata: AnnData
    donor_key_in_sc_adata: str
    gene_annotation: pd.DataFrame = field(init=False)

    def __init__(
        self,
        adata: AnnData,
        gdata: AnnData,
        donor_key_in_sc_adata: str,
        gene_annotation_path: str = "data/Yazar_OneK1K/gene_annotation.csv",
    ):
        self.adata = adata
        self.gdata = gdata
        self.donor_key_in_sc_adata = donor_key_in_sc_adata
        self._load_gene_annotation(gene_annotation_path)
        self._validate_data()
        self._match_donors()

    def _load_gene_annotation(self, gene_annotation_path: str):
        """Loads the gene annotation data from the specified file."""
        if not os.path.exists(gene_annotation_path):
            error_message = f"Gene annotation file not found at {gene_annotation_path}"
            raise FileNotFoundError(error_message)

        self.gene_annotation = pd.read_csv(gene_annotation_path, index_col="ensembl_gene_id")
        logger.info(f"Gene annotation loaded from {gene_annotation_path}")

    def _validate_data(self):
        """Validates that the donor key exists in the single-cell data and is categorical.

        Raises
        ------
            ValueError: If the donor_key_in_adata is not found
            ValueError: If adata.obs[self.donor_key_in_adata] is not categorical
        """
        if self.donor_key_in_sc_adata not in self.adata.obs.columns:
            raise ValueError(f"'{self.donor_key_in_sc_adata}' not found in adata.obs")
        if not self.adata.obs[self.donor_key_in_sc_adata].dtype.name == "category":
            raise ValueError(f"'{self.donor_key_in_sc_adata}' in adata.obs is not categorical")

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
        if donor not in self.adata.obs[self.donor_key_in_sc_adata].cat.categories:
            raise ValueError(f"Donor '{donor}' not found in adata")
        return self.adata[self.adata.obs[self.donor_key_in_sc_adata] == donor]

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

    def slice_cells(self, cell_condition) -> DonorData:
        """Returns a new DonorData object with single-cell data sliced based on the provided cell condition.

        Args:
            cell_condition: A boolean mask, condition, or slice to apply to the single-cell data.

        Returns
        -------
            DonorData: A new DonorData object with sliced single-cell data.
        """
        return DonorData(self.adata[cell_condition], self.gdata, self.donor_key_in_sc_adata)

    def slice_donors(self, donors: list[str]) -> DonorData:
        """Returns a new DonorData object with both single-cell and genetic data sliced to only include the specified donors.

        Args:
            donors (list[str]): A list of donor names to retain.

        Returns
        -------
            DonorData: A new DonorData object with sliced data.
        """
        new_gdata = self.gdata[self.gdata.obs.index.isin(donors)]
        return DonorData(self.adata, new_gdata, self.donor_key_in_sc_adata)

    def _match_donors(self) -> None:
        """Match donors between genetic and single-cell data.

        This method aligns the donors in genetic and single-cell data,
        keeping only the donors' data that is present in both datasets, and sorts the single-cell adata on the donors.

        Raises
        ------
            None

        Returns
        -------
            None

        Notes
        -----
        - Donor's data where a donor is not present in both datasets is dropped.
        - Warnings are logged about the number of donors kept and dropped.
        """
        # Sort single-cell data by the specified column
        self.adata = self.adata[self.adata.obs[self.donor_key_in_sc_adata].sort_values().index]

        # Get unique sample identifiers from both datasets
        sc_index: Index = pd.Index(self.adata.obs[self.donor_key_in_sc_adata].unique())
        g_index: Index = self.gdata.obs.index

        # Find common donors and all unique donors
        keep_donors: Index = sc_index.intersection(g_index)
        all_donors: Index = sc_index.union(g_index)

        # Log warnings about sample matching
        logger.info("Keeping %s/%s donors", len(keep_donors), len(all_donors))
        logger.info(
            "Dropping %s/%s donors from genetic data",
            len(g_index) - len(keep_donors),
            len(g_index),
        )
        logger.info(
            "Dropping %s/%s donors from single-cell data",
            len(sc_index) - len(keep_donors),
            len(sc_index),
        )

        # Filter both datasets to keep only matched donors
        self.gdata = self.gdata[keep_donors]
        self.adata = self.adata[self.adata.obs[self.donor_key_in_sc_adata].isin(keep_donors)]

    def __repr__(self) -> str:
        """String representation of DonorData showing side-by-side adata and gdata views."""
        # Create a console for rich
        console = Console()

        # Create a table
        table = Table(show_header=True, header_style="bold magenta")

        # Add two columns to represent adata and gdata
        table.add_column("Cells (adata)", max_width=100, justify="left")
        table.add_column("Donors (gdata)", max_width=100, justify="left")

        # Prepare the string representation of adata and gdata
        adata_repr = str(self.adata)
        gdata_repr = str(self.gdata)

        # Split the representations into lines for easy columnization
        adata_lines = adata_repr.splitlines()
        gdata_lines = gdata_repr.splitlines()

        # Ensure both have the same number of lines by padding with empty lines if necessary
        max_lines = max(len(adata_lines), len(gdata_lines))
        adata_lines += [""] * (max_lines - len(adata_lines))
        gdata_lines += [""] * (max_lines - len(gdata_lines))

        # Prepare the lines with highlighted donor key for adata
        highlighted_donor_key = self.donor_key_in_sc_adata  # The donor key to highlight
        adata_text_lines = []

        for line in adata_lines:
            if highlighted_donor_key in line:
                # If the donor key is found in the line, highlight it
                parts = line.split(highlighted_donor_key)
                highlighted_line = (
                    Text(parts[0]) + Text(highlighted_donor_key, style="bold blue underline") + Text(parts[1])
                )
            else:
                highlighted_line = Text(line)
            adata_text_lines.append(highlighted_line)

        for adata_line, gdata_line in zip(adata_text_lines, gdata_lines, strict=False):
            table.add_row(adata_line, gdata_line)
        # Use the console to print the table and return the string
        console.print(table)
        return ""

    def aggregate(
        self,
        key,
        target_key,
        filter_key: str = None,
        filter_value: str = None,
        agg_func: str | Callable = "mean",
    ) -> None:
        """Aggregate single-cell data to donor-level.

        Args:
            key:
                The key in adata to aggregate.
            target_key:
                The key in gdata to store the aggregated data.
            filter_key:
                The key in adata.obs to filter by. Defaults to None.
            filter_value:
                The value in adata.obs[filter_key] to filter by. Defaults to None.
            agg_func:
                The aggregation function to use. Defaults to "mean".
        """
        assert (filter_key is None) == (
            filter_value is None
        ), "filter_key and filter_value both have to be provided or None"

        adata = self.adata
        if filter_key is not None:
            adata = adata[adata.obs[filter_key] == filter_value]

        if key == "X":
            data = adata.X
            columns = adata.var_names
        elif key in adata.layers:
            data = adata.layers[key]
            columns = adata.var_names
        elif key in adata.obs.columns:
            data = adata.obs[[key]]
            columns = [key]
        elif key in adata.var.index:
            data = adata[:, [key]].X
            columns = [key]
        elif key in adata.obsm:
            data = adata.obsm[key]
            columns = getattr(data, "columns", range(data.shape[1]))
        else:
            raise ValueError(f"Key '{key}' not found in adata")

        data = pd.DataFrame(asarray(data), columns=columns)
        data["group"] = adata.obs[self.donor_key_in_sc_adata].values
        data = data.groupby("group", observed=True).agg(agg_func)
        data = data.loc[self.gdata.obs_names]
        if data.shape[1] == 1:
            self.gdata.obs[target_key] = data.iloc[:, 0]
        else:
            self.gdata.obsm[target_key] = data

    def _slice_genomic_region(self, chrom, start, end, window_size=1e6):
        """Returns a new DonorData object with genetic data sliced to a specific genomic region.

        Args:
            chrom: The chromosome to slice.
            start: The start position of the region.
            end: The end position of the region.
            window_size: Additional window size to include around the region. Defaults to 0.

        Returns
        -------
            A new DonorData object with sliced genetic data.

        Raises
        ------
            ValueError: If the required columns are not present in gdata.var.
        """
        required_columns = ["chrom", "pos"]
        if not all(col in self.gdata.var.columns for col in required_columns):
            raise ValueError(f"gdata.var must contain the following columns: {required_columns}")

        # Slice the genetic data
        mask = (
            (self.gdata.var.chrom == chrom)
            & (self.gdata.var.pos >= start - window_size)
            & (self.gdata.var.pos <= end + window_size)
        )
        new_gdata = self.gdata[:, mask]

        # Create a new DonorData object with the sliced genetic data
        return DonorData(self.adata, new_gdata, self.donor_key_in_sc_adata)

    def slice_genomic_region(self, ensembl_id: str, window_size: float = 1e6, tss: bool = True) -> DonorData:
        """
        Slice the genetic data to a specific genomic region based on an Ensembl gene ID.

        Args:
            ensembl_id: The Ensembl gene ID to slice around.
            window_size: Additional window size to include around the region. Defaults to 1e6.
            tss: If True, use only the transcription start site. If False, use the entire gene body. Defaults to True.

        Returns
        -------
            A new DonorData object with sliced genetic data.

        Raises
        ------
            ValueError: If the Ensembl ID is not found in the gene annotation.
        """
        if ensembl_id not in self.gene_annotation.index:
            raise ValueError(f"Ensembl ID '{ensembl_id}' not found in gene annotation.")

        gene_info = self.gene_annotation.loc[ensembl_id]
        chrom = gene_info["chromosome_name"]
        strand = gene_info["strand"]

        if tss:
            if strand == 1:
                start = end = gene_info["start_position"]
            else:
                start = end = gene_info["end_position"]
        else:
            start = gene_info["start_position"]
            end = gene_info["end_position"]

        return self._slice_genomic_region(chrom, start, end, window_size)
