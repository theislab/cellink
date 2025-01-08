from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import scanpy as sc
from anndata import AnnData
from rich.console import Console
from rich.table import Table
from rich.text import Text

from cellink._core.annotation import DAnn

logger = logging.getLogger(__name__)


@dataclass
class DonorData:
    """Store and manage donor-related data with single-cell readouts.

    This class allows donor-level, especially genetic, analysis with single-cell datasets.
    It holds AnnData objects for single-cell (C) and genetic (D) data

    Attributes
    ----------
        C (AnnData): Cell-level AnnData
        D (AnnData): Donor-level AnnData (e.g.: GenoAnnData)
        donor_key (str): Key for donor information in C.obs

    Returns
    -------
        _type_: DonorData object
    """

    def __init__(
        self,
        C: AnnData,
        D: AnnData,
        donor_key: str = DAnn.donor,
    ):
        if donor_key not in C.obs.columns:
            raise ValueError(f"'{donor_key}' not found in C.obs")
        if donor_key not in D.obs.columns and donor_key != D.obs.index.name:
            raise ValueError(f"'{donor_key}' must be in gdata.obs or set as index")
        if donor_key != D.obs.index.name:
            D.obs = D.obs.set_index(donor_key)

        self.donor_key = donor_key
        self._match_donors(C, D)

    def _match_donors(self, C: AnnData | None = None, D: AnnData | None = None) -> None:
        if C is None:
            C = self._C
        if D is None:
            D = self._D

        # Get unique sample identifiers from both datasets
        C_idx = pd.Index(C.obs[self.donor_key].unique())
        D_idx = D.obs.index
        keep_donors = D_idx.intersection(C_idx)

        C = C[C.obs[self.donor_key].isin(keep_donors)]
        D = D[keep_donors]

        # Sort cells by donor order
        sorted_cells = C.obs.iloc[
            pd.Categorical(C.obs[self.donor_key], categories=keep_donors, ordered=True).argsort()
        ].index

        self._C = C[sorted_cells]
        self._D = D

    @property
    def C(self) -> AnnData:
        return self._C

    @C.setter
    def C(self, value: AnnData) -> None:
        if not isinstance(value, AnnData):
            raise ValueError("C must be an AnnData object")
        self._match_donors(C=value, D=self._D)

    @property
    def D(self) -> AnnData:
        return self._D

    @D.setter
    def D(self, value: AnnData) -> None:
        if not isinstance(value, AnnData):
            raise ValueError("D must be an AnnData object")
        self._match_donors(C=self._C, D=value)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > 4:
            raise ValueError(f"Invalid slice length: {len(key)}")

        _D = self.D[key[0]]
        _C = self.C
        if len(key) >= 2:
            _D = _D[:, key[1]]
        if len(key) >= 3:
            _C = _C[key[2]]
        if len(key) == 4:
            _C = _C[:, key[3]]

        return DonorData(_C, _D, self.donor_key)

    def aggregate(
        self,
        key_added: None | str = None,
        layer: None | str = None,
        obs: None | str = None,
        obsm: None | str = None,
        filter_key: None | str = None,
        filter_value: None | str = None,
        add_to_obs: bool = False,
        func: str | Callable = "mean",
    ) -> None:
        """Aggregate single-cell data to donor-level.

        If neither layer, obsm, or obs is provided, adata.X is aggregated.
        Args:
            key_added:
                The key in gdata to store the aggregated data.
            layer:
                The layer in adata to aggregate. Defaults to None.
            obs:
                The key in adata.obs to aggregate. Defaults to None.
            obsm:
                The key in adata.obsm to aggregate. Defaults to None.
            filter_key:
                The key in adata.obs to filter by. Defaults to None.
            filter_value:
                The value in adata.obs[filter_key] to filter by. Defaults to None.
            func:
                The aggregation function to use. Defaults to "mean".
        """
        assert (filter_key is None) == (
            filter_value is None
        ), "filter_key and filter_value both have to be provided or None"

        assert (layer is None) + (obsm is None) + (
            obs is None
        ) >= 2, "Only one of layer, obsm, varm, or obs can be provided"

        adata = self.C
        if filter_key is not None:
            adata = adata[adata.obs[filter_key] == filter_value]

        slot = layer or obsm or obs or "X"
        if key_added is None:
            key_added = f"{slot}_{filter_value}_{func}"

        if obs is not None:
            obs = obs if isinstance(obs, list) else [obs]
            _data = adata.obs.groupby(self.donor_key, observed=True)[obs].agg(func)
            data = pd.DataFrame(index=self.D.obs_names, columns=obs)
            data.loc[data.index, obs] = _data
            if add_to_obs:
                self.D.obs.loc[data.index, obs] = data
            else:
                self.D.obsm[key_added] = data
        else:
            aggdata = sc.get.aggregate(adata, by=self.donor_key, func=func, layer=layer, obsm=obsm)
            if slot == "obsm":
                # use columns of obsm dataframe if provided
                columns = getattr(getattr(adata, slot), "columns", None)
            else:
                columns = adata.var_names

            data = pd.DataFrame(index=self.D.obs_names, columns=columns)
            data.loc[aggdata.obs_names] = aggdata.layers[func]
            self.D.obsm[key_added] = data

    def __repr__(self) -> str:
        """String representation of DonorData showing side-by-side adata and gdata views."""
        # Create a console for rich
        console = Console()

        # Create a table
        table = Table(show_header=True, header_style="bold deep_pink2")

        # Add two columns to represent adata and gdata
        table.add_column("D (donors)", max_width=50, justify="left")
        table.add_column("C (cells)", max_width=50, justify="left")

        # Split the representations into lines for easy columnization
        D_lines = str(self.D).splitlines()
        C_lines = str(self.C).splitlines()

        # Ensure both have the same number of lines by padding with empty lines if necessary
        max_lines = max(len(D_lines), len(C_lines))
        D_lines += [""] * (max_lines - len(D_lines))
        C_lines += [""] * (max_lines - len(C_lines))

        # Prepare the lines with highlighted donor key for adata
        highlighted_donor_key = self.donor_key  # The donor key to highlight
        adata_text_lines = []

        for line in C_lines:
            if highlighted_donor_key in line:
                # If the donor key is found in the line, highlight it
                parts = line.split(highlighted_donor_key)
                highlighted_line = (
                    Text(parts[0]) + Text(highlighted_donor_key, style="bold orange_red1 underline") + Text(parts[1])
                )
            else:
                highlighted_line = Text(line)
            adata_text_lines.append(highlighted_line)

        for adata_line, gdata_line in zip(adata_text_lines, D_lines, strict=False):
            table.add_row(gdata_line, adata_line)
        # Use the console to print the table and return the string
        console.print(table)
        return ""

    @property
    def shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return *self.D.shape, *self.C.shape
