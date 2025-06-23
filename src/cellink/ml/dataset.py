from typing import Any

import numpy as np
import torch

try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    Dataset = None
    DataLoader = None
from anndata import AnnData
from mudata import MuData
import numpy as np
import dask.array as da
from typing import Optional, List, Union, Dict, Any
from .._core import DonorData
from cellink._core.data_fields import CAnn, DAnn, GAnn, VAnn
from anndata.utils import asarray


def get_array(array: Any, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if array is None:
        return None

    masked = array[mask] if mask is not None else array

    masked = asarray(masked)
    #if isinstance(masked, da.Array):
    #    masked = masked.compute()
    #if issparse(masked):
    #    return masked.toarray()

    return masked

class MILDataset(Dataset):
    """
    A PyTorch Dataset for Multiple Instance Learning (MIL) using donor-level and cell-level data
    from AnnData or MuData objects. Supports both pre-split indices and random splitting.
    """

    def __init__(
        self,
        dd: AnnData | MuData,
        donor_layer: str | None = None,
        donor_labels_key: str | None = None,
        donor_batch_key: str | None = None,
        donor_cat_covs_key: str | None = None,
        donor_cont_covs_key: str | None = None,
        donor_indices_key: str | None = None,
        donor_id_key: str | None = None,
        celltype_key: str | None = None,
        cell_layer: str | None = None,
        cell_labels_key: str | None = None,
        cell_batch_key: str | None = None,
        cell_cat_covs_key: str | None = None,
        cell_cont_covs_key: str | None = None,
        cell_indices_key: str | None = None,
        cell_donor_key: str | None = None,
        split_donors: list[int] | None = None,
        split_indices: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        dd :
        donor_labels_key : str, optional
            Key for donor labels.
        donor_batch_key : str, optional
            Key for donor batch.
        donor_cat_covs_key : str, optional
            Key for categorical covariates for donors.
        donor_cont_covs_key : str, optional
            Key for continuous covariates for donors.
        donor_indices_key : str, optional
            Key for donor indices.
        donor_id_key : str, optional
            Key for donor ID.
        celltype_key : str, optional
            Key for cell type.
        cell_labels_key : str, optional
            Key for cell labels.
        cell_batch_key : str, optional
            Key for cell batch.
        cell_cat_covs_key : str, optional
            Key for categorical covariates for cells.
        cell_cont_covs_key : str, optional
            Key for continuous covariates for cells.
        cell_indices_key : str, optional
            Key for cell indices.
        cell_donor_key : str, optional
            Key for cell donor ID.
        split_indices : list of ints, optional
            Predefined indices to use as dataset subset.
        """

        if split_donors is not None and split_indices is not None:
            raise ValueError("Both split_donors and split_indices cannot be provided at the same time.")

        self.donor_labels_key = donor_labels_key or DAnn.DONOR_LABELS_KEY
        self.donor_batch_key = donor_batch_key or DAnn.DONOR_BATCH_KEY
        self.donor_cat_covs_key = donor_cat_covs_key or DAnn.DONOR_CAT_COVS_KEY
        self.donor_cont_covs_key = donor_cont_covs_key or DAnn.DONOR_CONT_COVS_KEY
        self.donor_indices_key = donor_indices_key or DAnn.DONOR_INDICES_KEY
        self.donor_id_key = donor_id_key or DAnn.DONOR_ID_KEY

        self.celltype_key = celltype_key or CAnn.celltype  # TODO MAYBE REMOVE
        self.cell_labels_key = cell_labels_key or CAnn.CELL_LABELS_KEY
        self.cell_batch_key = cell_batch_key or CAnn.CELL_BATCH_KEY
        self.cell_cat_covs_key = cell_cat_covs_key or CAnn.CELL_CAT_COVS_KEY
        self.cell_cont_covs_key = cell_cont_covs_key or CAnn.CELL_CONT_COVS_KEY
        self.cell_indices_key = cell_indices_key or CAnn.CELL_INDICES_KEY
        self.cell_donor_key = cell_donor_key or CAnn.CELL_DONOR_KEY

        self.dd = dd
        self.donor_layer = donor_layer
        self.cell_layer = cell_layer

        self.donor_ids = self._get_obs_field(self.dd.G, self.donor_id_key, force=True)

        if split_donors is not None:
            self.selected_ids = np.array(split_donors)
        elif split_indices is not None:
            self.selected_ids = self.donor_ids[split_indices]
        else:
            self.selected_ids = self.donor_ids

        self._donor_index = {id_: i for i, id_ in enumerate(self.selected_ids)}

    def __len__(self):
        return len(self.selected_ids)

    def __getitem__(self, idx: int):
        donor_id = self.selected_ids[idx]

        donor_mask = self.donor_ids == donor_id
        donor_x = self._get_layer(self.dd.G, mask=donor_mask, layer_key=self.donor_layer, force=False)
        donor_y = self._get_obs_field(self.dd.G, self.donor_labels_key, mask=donor_mask, force=True)
        donor_batch = self._get_obs_field(self.dd.G, self.donor_batch_key, mask=donor_mask, force=False)
        donor_cat_covs = self._get_obs_field(self.dd.G, self.donor_cat_covs_key, mask=donor_mask, force=False)
        donor_cont_covs = self._get_obs_field(self.dd.G, self.donor_cont_covs_key, mask=donor_mask, force=False)
        donor_indices = self._get_obs_field(self.dd.G, self.donor_indices_key, mask=donor_mask, force=False)

        cell_donor_ids = self._get_obs_field(self.dd.C, self.cell_donor_key, force=False)
        cell_mask = cell_donor_ids == donor_id
        cell_x = self._get_layer(self.dd.C, mask=cell_mask, layer_key=self.cell_layer, force=True)
        cell_y = self._get_obs_field(self.dd.C, self.cell_labels_key, mask=cell_mask, force=False)
        cell_batch = self._get_obs_field(self.dd.C, self.cell_batch_key, mask=cell_mask, force=False)
        cell_cat_covs = self._get_obs_field(self.dd.C, self.cell_cat_covs_key, mask=cell_mask, force=False)
        cell_cont_covs = self._get_obs_field(self.dd.C, self.cell_cont_covs_key, mask=cell_mask, force=False)
        cell_indices = self._get_obs_field(self.dd.C, self.cell_indices_key, mask=cell_mask, force=False)

        sample = {
            "donor_id": donor_id,
            "donor_x": torch.tensor(np.squeeze(donor_x), dtype=torch.float32) if donor_x is not None else None,
            "donor_y": torch.tensor(donor_y, dtype=torch.float32),
            "donor_batch": torch.tensor(donor_batch, dtype=torch.float32) if donor_batch is not None else None,
            "donor_cat_covs": torch.tensor(donor_cat_covs, dtype=torch.float32) if donor_cat_covs is not None else None,
            "donor_cont_covs": torch.tensor(donor_cont_covs, dtype=torch.float32)
            if donor_cont_covs is not None
            else None,
            "donor_indices": torch.tensor(donor_indices, dtype=torch.float32) if donor_indices is not None else None,
            "cell_x": torch.tensor(cell_x, dtype=torch.float32),
            "cell_y": torch.tensor(cell_y, dtype=torch.float32) if cell_y is not None else None,
            "cell_batch": torch.tensor(cell_batch, dtype=torch.float32) if cell_batch is not None else None,
            "cell_cat_covs": torch.tensor(cell_cat_covs, dtype=torch.float32) if cell_cat_covs is not None else None,
            "cell_cont_covs": torch.tensor(cell_cont_covs, dtype=torch.float32) if cell_cont_covs is not None else None,
            "cell_indices": torch.tensor(cell_indices, dtype=torch.float32) if cell_indices is not None else None,
        }

        return sample

    def _get_layer(
        self, data: Union[AnnData, MuData], layer_key: Optional[str], mask: np.ndarray = None, force: bool = False
    ) -> Any:
        """
        Get the layer from the data. If force is False and layer is missing, return None.
        """
        if isinstance(data, MuData):
            data_list = [data.mod[key] for key in data.mod.keys()]
            if layer_key is not None:
                arrays = [
                    get_array(d.layers.get(layer_key), mask)
                    for d in data_list
                    if d.layers.get(layer_key) is not None
                ]
                return np.concatenate(arrays, axis=0) if arrays else None
            else:
                arrays = [
                    get_array(d.X, mask)
                    for d in data_list
                    if d.X is not None
                ]
                return np.concatenate(arrays, axis=1) if arrays else None

        if layer_key is not None:
            return get_array(data.layers.get(layer_key), mask)

        return get_array(data.X, mask)

    def _get_obs_field(
        self, data: AnnData | MuData, key: str | None, mask: np.ndarray = None, force: bool = False
    ) -> Any:
        """
        Fetch the obs field. If the field is missing and force is False, return None.
        """
        if key is None:
            raise ValueError("Field key must be provided.")
        if isinstance(data, MuData):
            for mod in data.mod.values():
                if key in mod.obs.columns:
                    return mod.obs[key].to_numpy() if mask is None else mod.obs[key].to_numpy()[mask]
            if not force:
                return None
        if key in data.obs.columns:
            return data.obs[key].to_numpy() if mask is None else data.obs[key].to_numpy()[mask]
        if not force:
            return None
        raise KeyError(f"Field '{key}' not found in the data.")


def mil_collate_fn(batch):
    """
    Custom collate function for MIL Dataset.
    Combines donor and cell data in the correct format for batch processing.
    """

    stack_fields = [
      "donor_x", 
      "donor_y", 
      "donor_batch", 
      "donor_cat_covs", 
      "donor_cont_covs", 
      "donor_indices",
    ]

    list_fields = [
      "cell_x", 
      "cell_y",
      "cell_batch", 
      "cell_cat_covs", 
      "cell_cont_covs", 
      "cell_indices"
    ]

    collected = {key: [] for key in stack_fields + list_fields}

    for item in batch:
        for key in collected:
            if key in item and item[key] is not None:
                collected[key].append(item[key])

    sample = {}
    for key, values in collected.items():
        if not values:
            continue
        sample[key] = torch.stack(values) if key in stack_fields else values

    return sample
