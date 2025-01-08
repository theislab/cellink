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

HIGHLIGHT_COLOR = "bold deep_pink2"


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
        var_dims_to_sync: list[str] = None,
    ):
        if donor_key not in C.obs.columns:
            raise ValueError(f"'{donor_key}' not found in C.obs")
        if donor_key not in D.obs.columns and donor_key != D.obs.index.name:
            raise ValueError(f"'{donor_key}' must be in gdata.obs or set as index")
        if donor_key != D.obs.index.name:
            D.obs = D.obs.set_index(donor_key)

        self._var_dims_to_sync = [] if var_dims_to_sync is None else var_dims_to_sync
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

        if not keep_donors.equals(D.obs_names):
            D = D[keep_donors]

        keep_cells = C.obs[self.donor_key].isin(keep_donors)
        if not keep_cells.all():
            C = C[keep_cells]

        # Sort cells by donor order
        sorted_cells = C.obs.iloc[
            pd.Categorical(C.obs[self.donor_key], categories=keep_donors, ordered=True).argsort()
        ].index
        if not sorted_cells.equals(C.obs_names):
            C = C[sorted_cells]

        self._C = C
        self._D = D

    def copy(self) -> DonorData:
        self._C = self._C.copy()
        self._D = self._D.copy()
        return self

    @property
    def C(self) -> AnnData:
        return self._C

    @C.setter
    def C(self, value: AnnData) -> None:
        if not isinstance(value, AnnData):
            raise ValueError("C must be an AnnData object")
        self._C = value
        self._match_donors()

    @property
    def D(self) -> AnnData:
        return self._D

    @D.setter
    def D(self, value: AnnData) -> None:
        if not isinstance(value, AnnData):
            raise ValueError("D must be an AnnData object")
        self._D = value
        self._match_donors()

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
            keys_to_sync = []
            for key in self._var_dims_to_sync:
                if key in _D.obsm:
                    keys_to_sync.append(key)
                    _D.obsm[key] = _D.obsm[key].loc[:, _C.var.index]
            self._var_dims_to_sync = keys_to_sync

        return DonorData(_C, _D, self.donor_key, self._var_dims_to_sync)

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
        sync_var: bool = False,
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
            key_added = f"{slot}_{func}"
            if filter_value is not None:
                key_added += f"_{filter_value}"

        if obs is not None:
            obs = obs if isinstance(obs, list) else [obs]
            _data = adata.obs.groupby(self.donor_key, observed=True)[obs].agg(func)
            dtype = _data.dtypes.iloc[0]
            data = pd.DataFrame(index=self.D.obs_names, columns=obs, dtype=dtype)
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
                if sync_var:
                    self._var_dims_to_sync.append(key_added)

            dtype = aggdata.layers[func].dtype
            data = pd.DataFrame(index=self.D.obs_names, columns=columns, dtype=dtype)
            data.loc[aggdata.obs_names] = aggdata.layers[func]
            self.D.obsm[key_added] = data

    def prep_repr(self) -> str:
        """String representation of DonorData showing side-by-side adata and gdata views."""
        # Split the representations into lines
        D_lines, D_highlight = _anndata_repr(self.D, self.D.n_obs, self.D.n_vars, self._var_dims_to_sync)
        C_lines, C_highlight = _anndata_repr(self.C, self.C.n_obs, self.C.n_vars, [self.donor_key])

        def pad_lists(l1, l2):
            max_lines = max(len(l1), len(l2))
            l1 += [""] * (max_lines - len(l1))
            l2 += [""] * (max_lines - len(l2))
            return l1, l2

        # Ensure both have the same number of lines by padding with empty lines if necessary
        D_lines, C_lines = pad_lists(D_lines, C_lines)
        D_highlight, C_highlight = pad_lists(D_highlight, C_highlight)

        def highlight_lines(lines, highlights):
            _lines = []
            for line, highlight in zip(lines, highlights, strict=True):
                text = Text()
                for idx, (el, hl) in enumerate(zip(line, highlight, strict=True)):
                    _el = f"{el}, " if idx not in [0, len(line) - 1] else el
                    text.append(_el, HIGHLIGHT_COLOR if hl else None)
                _lines.append(text)
            return _lines

        C_lines = highlight_lines(C_lines, C_highlight)
        D_lines = highlight_lines(D_lines, D_highlight)

        # Create table
        table = Table(show_header=True, header_style=HIGHLIGHT_COLOR)
        table.add_column("D (donors)", max_width=50, justify="left")
        table.add_column("C (cells)", max_width=50, justify="left")
        for adata_line, gdata_line in zip(C_lines, D_lines, strict=True):
            table.add_row(gdata_line, adata_line)

        return table

    def __repr__(self) -> str:
        table = self.prep_repr()
        Console().print(table)
        return ""

    def __str__(self) -> str:
        n_donors, n_donor_vars, n_cells, n_cell_vars = self.shape
        return f"{__class__.__name__}({n_donors=}, {n_donor_vars=}, {n_cells=}, {n_cell_vars=})"

    @property
    def shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return *self.D.shape, *self.C.shape


def _anndata_repr(adata, n_obs, n_vars, highlight_keys=None) -> str:
    if adata.isbacked:
        backed_at = f" backed at {str(adata.filename)!r}"
    else:
        backed_at = ""
    lines = [[f"AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"]]
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
