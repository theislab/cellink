from typing import Any

import numpy as np
import torch

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    Dataset = None
    DataLoader = None

from anndata import AnnData
from anndata.utils import asarray
from mudata import MuData

from cellink._core.data_fields import CAnn, DAnn


def _get_array(array: Any, mask: np.ndarray | None = None) -> np.ndarray | None:
    """
    Convert an input array-like object to a NumPy array and optionally apply a mask.

    Parameters
    ----------
    array : Any
        Input array-like object, such as a NumPy array, Dask array, or sparse matrix.
    mask : np.ndarray, optional
        Boolean or integer array used to select a subset of the input. If None, the entire array is returned.

    Returns
    -------
    np.ndarray or None
        The resulting NumPy array after masking, or None if the input array is None.
    """
    if array is None:
        return None

    masked = array[mask] if mask is not None else array

    masked = asarray(masked)
    # if isinstance(masked, da.Array):
    #    masked = masked.compute()
    # if issparse(masked):
    #    return masked.toarray()

    return masked


def _get_layer(
    data: AnnData | MuData, layer_key: str | None = None, mask: np.ndarray | None = None
) -> np.ndarray | None:
    """
    Retrieve feature matrix from AnnData or MuData object.

    Parameters
    ----------
    data : AnnData or MuData
        Data object to extract the features from.
    layer_key : str, optional
        Layer name to extract. If None, uses `X` by default.
    mask : np.ndarray, optional
        Boolean array for selecting specific rows.

    Returns
    -------
    np.ndarray or None
        Feature matrix or None if not found.
    """
    if isinstance(data, MuData):
        data_list = [data.mod[key] for key in data.mod.keys()]
        if layer_key is not None:
            arrays = [
                _get_array(d.layers.get(layer_key), mask) for d in data_list if d.layers.get(layer_key) is not None
            ]
            return np.concatenate(arrays, axis=1) if arrays else None
        else:
            arrays = [_get_array(d.X, mask) for d in data_list if d.X is not None]
            return np.concatenate(arrays, axis=1) if arrays else None

    if layer_key is not None:
        return _get_array(data.layers.get(layer_key), mask)

    return _get_array(data.X, mask)


def _get_obs_field(
    data: AnnData | MuData, key: str | list[str] | None, mask: np.ndarray | None = None, force: bool = False
) -> np.ndarray | None:
    """
    Retrieve observation-level field (labels, batch, covariates) from data.

    Supports both single keys and lists of keys. When a list is provided,
    columns are concatenated horizontally.

    Parameters
    ----------
    data : AnnData or MuData
        Data object to extract observation fields from.
    key : str, List[str], or None
        Key or list of keys for fields in `.obs`. If a list, columns are concatenated.
    mask : np.ndarray, optional
        Boolean array to select subset of observations.
    force : bool, default=False
        If True, raises an error if the key is missing. Otherwise, returns None.

    Returns
    -------
    np.ndarray or None
        Array of values for the specified key(s), optionally masked.
        If multiple keys provided, returns concatenated array along axis 1.
    """
    if key is None:
        if force:
            raise ValueError("Field key must be provided when force=True.")
        return None

    if isinstance(key, list):
        if len(key) == 0:
            if force:
                raise ValueError("Empty key list provided.")
            return None

        arrays = []
        for k in key:
            arr = _get_obs_field(data, k, mask=mask, force=force)
            if arr is not None:
                arrays.append(arr)

        if not arrays:
            return None

        return np.concatenate(arrays, axis=0)

    if isinstance(data, MuData):
        for mod in data.mod.values():
            if key in mod.obs.columns:
                result = mod.obs[key].to_numpy() if mask is None else mod.obs[key].to_numpy()[mask]
                return result
        if not force:
            return None
        raise KeyError(f"Field '{key}' not found in MuData")

    if key in data.obs.columns:
        result = data.obs[key].to_numpy() if mask is None else data.obs[key].to_numpy()[mask]
        return result

    if not force:
        return None

    raise KeyError(f"Field '{key}' not found in the data.")


class MILDataset(Dataset):
    """
    Zarr-safe MIL dataset.
    """

    def __init__(
        self,
        dd: AnnData | MuData,
        donor_layer: str | None = None,
        donor_labels_key: str | list[str] | None = None,
        donor_id_key: str | None = None,
        cell_layer: str | None = None,
        cell_labels_key: str | list[str] | None = None,
        cell_donor_key: str | None = None,
        split_donors: list[int] | None = None,
        split_indices: list[int] | None = None,
    ):
        self.dd = dd
        self.donor_layer = donor_layer
        self.cell_layer = cell_layer

        self.donor_labels_key = donor_labels_key or DAnn.DONOR_LABELS_KEY
        self.donor_id_key = donor_id_key or DAnn.DONOR_ID_KEY
        self.cell_labels_key = cell_labels_key or CAnn.CELL_LABELS_KEY
        self.cell_donor_key = cell_donor_key or CAnn.CELL_DONOR_KEY

        donor_ids = _get_obs_field(dd.G, self.donor_id_key, force=True)

        if split_donors is not None:
            self.selected_ids = np.asarray(split_donors)
        elif split_indices is not None:
            self.selected_ids = donor_ids[split_indices]
        else:
            self.selected_ids = donor_ids

        # Donor rows
        donor_id_to_row = {id_: i for i, id_ in enumerate(donor_ids)}
        self.donor_rows = np.asarray([donor_id_to_row[id_] for id_ in self.selected_ids])

        # Cell rows per donor
        cell_donor_ids = _get_obs_field(dd.C, self.cell_donor_key, force=True)
        self.cells_per_donor = {}
        for i, did in enumerate(cell_donor_ids):
            self.cells_per_donor.setdefault(did, []).append(i)

    def __len__(self):
        return len(self.donor_rows)

    def __getitem__(self, idx: int):
        donor_row = self.donor_rows[idx]
        donor_id = self.selected_ids[idx]

        donor_x = self.dd.G.layers[self.donor_layer][donor_row] if self.donor_layer else self.dd.G.X[donor_row]

        cell_rows = self.cells_per_donor.get(donor_id, [])
        cell_x = self.dd.C.layers[self.cell_layer][cell_rows] if self.cell_layer else self.dd.C.X[cell_rows]

        return {
            "donor_x": donor_x,
            "donor_y": _get_obs_field(self.dd.G, self.donor_labels_key)[donor_row],
            "cell_x": cell_x,
            "cell_y": _get_obs_field(self.dd.C, self.cell_labels_key)[cell_rows]
            if self.cell_labels_key is not None
            else None,
        }


def mil_collate_fn(batch):
    out = {}

    out["donor_x"] = torch.stack([torch.from_numpy(b["donor_x"]) for b in batch])
    out["donor_y"] = torch.as_tensor([b["donor_y"] for b in batch], dtype=torch.float32)

    out["cell_x"] = [torch.from_numpy(b["cell_x"]) for b in batch]

    if batch[0]["cell_y"] is not None:
        out["cell_y"] = [torch.from_numpy(b["cell_y"]) for b in batch]

    return out


class GeneticsDataset(Dataset):
    """
    Zarr-safe Genetics dataset.
    - O(1) row access
    - No boolean masking
    - No eager densification
    """

    def __init__(
        self,
        data: AnnData | MuData,
        layer: str | None = None,
        labels_key: str | list[str] | None = None,
        batch_key: str | list[str] | None = None,
        cat_covs_key: str | list[str] | None = None,
        cont_covs_key: str | list[str] | None = None,
        indices_key: str | None = None,
        id_key: str | None = None,
        split_ids: list[int] | None = None,
        split_indices: list[int] | None = None,
    ):
        if split_ids is not None and split_indices is not None:
            raise ValueError("Cannot provide both split_ids and split_indices")

        self.data = data
        self.layer = layer

        self.labels_key = labels_key or DAnn.DONOR_LABELS_KEY
        self.batch_key = batch_key or DAnn.DONOR_BATCH_KEY
        self.cat_covs_key = cat_covs_key or DAnn.DONOR_CAT_COVS_KEY
        self.cont_covs_key = cont_covs_key or DAnn.DONOR_CONT_COVS_KEY
        self.indices_key = indices_key or DAnn.DONOR_INDICES_KEY
        self.id_key = id_key or DAnn.DONOR_ID_KEY

        # All IDs (length = n_obs)
        self.all_ids = _get_obs_field(self.data, self.id_key, force=True)

        if split_ids is not None:
            self.selected_ids = np.asarray(split_ids)
        elif split_indices is not None:
            self.selected_ids = self.all_ids[split_indices]
        else:
            self.selected_ids = self.all_ids

        # PRECOMPUTE ROW INDICES (O(N) ONCE)
        id_to_row = {}
        for i, id_ in enumerate(self.all_ids):
            if id_ not in id_to_row:
                id_to_row[id_] = i

        self.rows = np.asarray([id_to_row[id_] for id_ in self.selected_ids])

    def __len__(self):
        return len(self.rows)

    def _get_x(self, row: int):
        if self.layer is None:
            return self.data.X[row]
        return self.data.layers[self.layer][row]

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        x = self._get_x(row)

        sample = {
            "x": x,  # still NumPy/Zarr-backed
            "y": _get_obs_field(self.data, self.labels_key)[row],
        }

        for key, name in [
            (self.batch_key, "batch"),
            (self.cat_covs_key, "cat_covs"),
            (self.cont_covs_key, "cont_covs"),
            (self.indices_key, "indices"),
        ]:
            if key is not None:
                val = _get_obs_field(self.data, key)
                if val is not None:
                    sample[name] = val[row]

        return sample


def genetics_collate_fn(batch):
    """
    Batched densification happens HERE, once per batch.
    """
    out = {}

    # Features
    x = [torch.from_numpy(b["x"]) for b in batch]
    out["x"] = torch.stack(x)

    # Labels
    out["y"] = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)

    for key in ["batch", "cat_covs", "cont_covs", "indices"]:
        if key in batch[0]:
            out[key] = torch.as_tensor([b[key] for b in batch], dtype=torch.float32)

    return out
