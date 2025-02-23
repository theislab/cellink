from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import scanpy as sc
from anndata import AnnData

if TYPE_CHECKING:
    from mudata import MuData

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cellink._core.data_fields import DAnn

logger = logging.getLogger(__name__)

HIGHLIGHT_COLOR = "bold deep_pink2"


@dataclass
class DonorData:
    """Store and manage donor-related data with single-cell readouts.

    This class allows donor-level, especially genetic, analysis with single-cell datasets.
    It holds AnnData objects for donor-level/genetic (G) and single-cell (C) data

    Attributes
    ----------
        G (AnnData): Donor-level AnnData
        C (AnnData): Cell-level AnnData
        donor_id (str): Key for donor information in C.obs

    Returns
    -------
        DonorData object
    """

    def __init__(
        self,
        *,
        G: AnnData | MuData,
        C: AnnData | MuData,
        donor_id: str = DAnn.donor,
        var_dims_to_sync: list[str] = None,
    ):
        if donor_id not in C.obs.columns:
            raise ValueError(f"'{donor_id}' not found in C.obs")
        if donor_id not in G.obs.columns and donor_id != G.obs.index.name:
            raise ValueError(f"'{donor_id}' must be in gdata.obs or set as index")
        if donor_id != G.obs.index.name:
            G.obs = G.obs.set_index(donor_id)

        self._var_dims_to_sync = [] if var_dims_to_sync is None else var_dims_to_sync
        self.donor_id = donor_id
        self._match_donors(G, C)

    def _match_donors(self, G: AnnData | MuData, C: AnnData | MuData) -> None:
        G_idx = G.obs.index
        C_idx = pd.Index(C.obs[self.donor_id].unique())
        keep_donors = G_idx.intersection(C_idx)

        if len(keep_donors) == 0:
            raise ValueError(
                (
                    "No overlapping donors between dd.G and dd.C,"
                    " make sure dd.C.obs['%s'] exists and corresponds to dd.G.obs_names"
                ),
                self.donor_id,
            )
        if not keep_donors.equals(G.obs_names):
            G = G[keep_donors]

        keep_cells = C.obs[self.donor_id].isin(keep_donors)
        if not keep_cells.all():
            C = C[keep_cells]

        # Sort cells by donor order
        sorted_cells = C.obs.iloc[
            pd.Categorical(C.obs[self.donor_id], categories=keep_donors, ordered=True).argsort()
        ].index
        if not sorted_cells.equals(C.obs_names):
            C = C[sorted_cells]

        self._C = C
        self._G = G

    def copy(self) -> DonorData:
        if self._G.is_view:
            self._G = self._G.copy()
        if self._C.is_view:
            self._C = self._C.copy()
        return self

    @property
    def C(self) -> AnnData:
        return self._C

    @C.setter
    def C(self, value: AnnData | MuData) -> None:
        if not isinstance(value, AnnData | MuData):
            raise ValueError("C must be an AnnData or MuData object")
        self._C = value
        self._match_donors(self._G, self._C)

    @property
    def G(self) -> AnnData:
        return self._G

    @G.setter
    def G(self, value: AnnData | MuData) -> None:
        if not isinstance(value, AnnData | MuData):
            raise ValueError("G must be an AnnData or MuData object")
        self._G = value
        self._match_donors(self._G, self._C)

    def _sync_var_dims(self, G: AnnData | MuData, C: AnnData | MuData) -> None:
        """Sync _D.obsm if needed, drop deleted keys from _var_dims_to_sync"""
        keys_to_sync = []
        for v in self._var_dims_to_sync:
            if v in G.obsm:
                keys_to_sync.append(v)
                if not C.var.index.equals(G.obsm[v].columns):
                    if G.is_view:
                        G = G.copy()
                    G.obsm[v] = G.obsm[v].loc[:, C.var.index]
        self._var_dims_to_sync = keys_to_sync
        return G

    def sel(
        self,
        *,
        G_obs: slice = slice(None),
        G_var: slice = slice(None),
        C_obs: slice = slice(None),
        C_var: slice = slice(None),
    ):
        _G = self.G[G_obs]
        _G = _G[:, G_var]
        _C = self.C[C_obs]
        _C = _C[:, C_var]

        _G = self._sync_var_dims(_G, _C)
        return DonorData(G=_G, C=_C, donor_id=self.donor_id, var_dims_to_sync=self._var_dims_to_sync)

    # TODO: properly support ellipsis
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 4:
            raise IndexError("DonorData only supports 4 dimensions")

        key = self._expand_ellipsis(key)
        key = tuple(
            (idx,) if isinstance(idx, str) else idx for idx in key
        )  # needed because Mudata[str] looks up modalities
        key = key + (slice(None),) * (4 - len(key))
        # Only slice if key is not slice(None)
        _G = self.G[key[0]] if key[0] is not slice(None) else self.G
        _G = _G[:, key[1]] if key[1] is not slice(None) else _G
        _C = self.C[key[2]] if key[2] is not slice(None) else self.C
        _C = _C[:, key[3]] if key[3] is not slice(None) else _C

        _G = self._sync_var_dims(_G, _C)
        return DonorData(
            G=_G,
            C=_C,
            donor_id=self.donor_id,
            var_dims_to_sync=self._var_dims_to_sync,
        )

    def _expand_ellipsis(self, index_tuple):
        ellipsis_count = sum(1 for idx in index_tuple if idx is Ellipsis)
        if ellipsis_count == 0:
            return index_tuple
        if ellipsis_count > 1:
            raise IndexError("DonorData only supports a single ellipsis ('...').")

        ellipsis_index = index_tuple.index(Ellipsis)
        provided = len(index_tuple) - 1  # minus the ellipsis
        num_full_slices = 4 - provided  # dd has 4 dimensions

        new_index = index_tuple[:ellipsis_index] + (slice(None),) * num_full_slices + index_tuple[ellipsis_index + 1 :]
        return new_index

    def aggregate(
        self,
        *,
        key_added: None | str = None,
        layer: None | str = None,
        obs: None | str = None,
        obsm: None | str = None,
        filter_key: None | str = None,
        filter_value: None | str = None,
        add_to_obs: bool = False,
        func: str | Callable = "mean",
        sync_var: bool = False,
        verbose: bool = False,
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
            add_to_obs:
                Whether to add the aggregated data to adata.obs. Defaults to False.
            func:
                The aggregation function to use. Defaults to "mean".
            sync_var:
                Whether to set the variable dimensions of the aggregated data to sync with
                the variable dimensions of the single-cell data. Defaults to False.
            verbose:
                Whether to print verbose output. Defaults to False.
        """
        if filter_key is not None:
            assert filter_value is not None, "filter_value must be provided if filter_key is provided"

        assert (layer is None) + (obsm is None) + (
            obs is None
        ) >= 2, "Only one of layer, obsm, varm, or obs can be provided"

        adata = self.C
        if filter_key is not None:
            adata = adata[adata.obs[filter_key] == filter_value]

        slot = layer or obsm or obs or "X"
        if key_added is None:
            key_added = f"{slot}_{func}"
            if filter_value is not None:
                key_added += f"_{filter_value}"

        if obs is not None:
            columns = obs if isinstance(obs, list) else [obs]
            aggres = adata.obs.groupby(self.donor_id, observed=True)[columns].agg(func)
            dtype = aggres.dtypes.iloc[0]
            target = "obs" if add_to_obs else "obsm"
        else:
            aggdata = sc.get.aggregate(adata, by=self.donor_id, func=func, layer=layer, obsm=obsm)
            if slot == "obsm":
                # use columns of obsm dataframe if provided
                columns = getattr(getattr(adata, slot), "columns", None)
            else:
                columns = adata.var_names
                if sync_var:
                    self._var_dims_to_sync.append(key_added)

            dtype = aggdata.layers[func].dtype
            # Convert the aggregated layer to a DataFrame.
            aggres = pd.DataFrame(aggdata.layers[func], index=aggdata.obs_names)
            target = "obsm"

        if verbose:
            logger.info(f"Aggregated {slot} to {key_added}")
            logger.info("Observation found for %s donors.", aggres.shape[0])
        data = pd.DataFrame(index=self.G.obs_names, columns=columns, dtype=dtype)
        data.loc[aggres.index] = aggres

        if self.G.is_view:  # because we will write to self.G
            self.G = self.G.copy()

        if target == "obs":
            self.G.obs.loc[data.index, columns] = data
        else:
            self.G.obsm[key_added] = data

    def prep_repr(self) -> str:
        """String representation of DonorData showing side-by-side dd.G and dd.C views."""
        # Split the representations into lines
        G_lines, G_highlight = _anndata_repr(self.G, self.G.n_obs, self.G.n_vars, self._var_dims_to_sync)
        C_lines, C_highlight = _anndata_repr(self.C, self.C.n_obs, self.C.n_vars, [self.donor_id])

        def pad_lists(l1, l2):
            max_lines = max(len(l1), len(l2))
            l1 += [""] * (max_lines - len(l1))
            l2 += [""] * (max_lines - len(l2))
            return l1, l2

        # Ensure both have the same number of lines by padding with empty lines if necessary
        G_lines, C_lines = pad_lists(G_lines, C_lines)
        G_highlight, C_highlight = pad_lists(G_highlight, C_highlight)

        def highlight_lines(lines, highlights):
            _lines = []
            for line, highlight in zip(lines, highlights, strict=True):
                text = Text()
                for idx, (el, hl) in enumerate(zip(line, highlight, strict=True)):
                    _el = f"{el}, " if idx not in [0, len(line) - 1] else el
                    text.append(_el, HIGHLIGHT_COLOR if hl else None)
                _lines.append(text)
            return _lines

        G_lines = highlight_lines(G_lines, G_highlight)
        C_lines = highlight_lines(C_lines, C_highlight)

        # Create table
        table = Table(show_header=True, header_style=HIGHLIGHT_COLOR)
        table.add_column("G (donors)", max_width=50, justify="left")
        table.add_column("C (cells)", max_width=50, justify="left")
        for gdata_line, adata_line in zip(G_lines, C_lines, strict=True):
            table.add_row(gdata_line, adata_line)

        n_donors = self.G.shape[0]
        n_cells_per_donor = self.C.obs[self.donor_id].value_counts()
        min_n_cells = n_cells_per_donor.min()
        max_n_cells = n_cells_per_donor.max()
        header_line = (
            f"    DonorData({n_donors} x "
            f"n_donors={self.G.shape[1]:,}, n_cells_per_donor=[{min_n_cells:,}-{max_n_cells:,}], "
            f"donor_id = '{self.donor_id}')"
        )
        spanning_header = Panel(Text(header_line, style=HIGHLIGHT_COLOR, justify="center"), width=100)
        spanning_header = Align.center(spanning_header)

        return Group(spanning_header, table)

    def __repr__(self) -> str:
        table = self.prep_repr()
        Console().print(table)
        return ""

    def __str__(self) -> str:
        n_donors, n_donor_vars, n_cells, n_cell_vars = self.shape
        return (
            f"{self.__class__.__name__}"
            f"(n_G_obs={n_donors:,}, n_G_vars={n_donor_vars:,}, "
            f"n_C_obs={n_cells:,}, n_C_vars={n_cell_vars:,})"
        )

    @property
    def shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return *self.G.shape, *self.C.shape


def _anndata_repr(adata, n_obs, n_vars, highlight_keys=None) -> str:
    if adata.isbacked:
        backed_at = f" backed at {str(adata.filename)!r}"
    else:
        backed_at = ""
    lines = [[f"AnnData object with n_obs × n_vars = {n_obs:,} × {n_vars:,} {backed_at}"]]
    if adata.is_view:
        lines[0][0] = "View of " + lines[0][0]
    highlight = [[False]]
    for attr in ["obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp"]:
        keys = getattr(adata, attr).keys()

        if len(keys) > 0:
            line = [f"    {attr}: "]
            line_highlight = [False]
            for key in keys:
                if attr in ["obsm", "obs"] and key in highlight_keys:
                    line_highlight.append(True)
                else:
                    line_highlight.append(False)
                line.append(f"'{key}'")
            lines.append(line)
            highlight.append(line_highlight)
    return lines, highlight
