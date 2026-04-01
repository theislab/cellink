from collections.abc import Iterator
from typing import Any, Optional, Union

import numpy as np
import torch
from anndata import AnnData
from anndata.utils import asarray
from mudata import MuData

from .._core import DonorData
from cellink._core.data_fields import CAnn, DAnn

try:
    from torch.utils.data import Dataset, IterableDataset
except ImportError:
    Dataset = None
    IterableDataset = object

try:
    from annbatch import Loader
    from annbatch.abc import Sampler as AnnBatchSampler
except ImportError:
    Loader = None
    AnnBatchSampler = object


def get_array(array: Any, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
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
    #if isinstance(masked, da.Array):
    #    masked = masked.compute()
    #if issparse(masked):
    #    return masked.toarray()

    return masked


def _to_tensor(array: Any, *, squeeze: bool = False) -> torch.Tensor | None:
    if array is None:
        return None
    array = np.squeeze(array) if squeeze else array
    return torch.tensor(array, dtype=torch.float32)


def _to_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.float()
        return x.to_dense() if x.layout != torch.strided else x
    tensor = torch.tensor(asarray(x), dtype=torch.float32)
    return tensor.to_dense() if tensor.layout != torch.strided else tensor


def _get_contiguous_bag_slices(ids: np.ndarray, selected_ids: np.ndarray) -> list[tuple[str, slice]]:
    if ids.ndim != 1:
        raise ValueError("Cell donor ids must be a one-dimensional array.")
    if ids.size == 0:
        return []

    unique_ids, starts = np.unique(ids, return_index=True)
    order = np.argsort(starts)
    unique_ids = unique_ids[order]
    starts = starts[order]
    stops = np.append(starts[1:], ids.size)

    bag_slices = []
    selected_set = set(selected_ids.tolist())
    for donor_id, start, stop in zip(unique_ids, starts, stops, strict=True):
        donor_slice = ids[start:stop]
        if np.any(donor_slice != donor_id):
            raise ValueError(
                "Cells for each donor must be contiguous for annbatch MIL loading. "
                "Construct `DonorData` first or sort `dd.C.obs[cell_donor_key]` by donor."
            )
        if donor_id in selected_set:
            bag_slices.append((donor_id, slice(int(start), int(stop))))
    return bag_slices


def _extract_obs_array(obs, key: str | None) -> np.ndarray | None:
    if obs is None or key is None:
        return None
    if key not in obs.columns:
        return None
    return obs[key].to_numpy()

class MILDataset(Dataset):
    """
    PyTorch Dataset for Multiple Instance Learning (MIL) using donor- and cell-level data
    from AnnData or MuData objects.

    This dataset supports both pre-split donor indices and random splitting. Each
    item returned by the dataset contains the donor-level and corresponding cell-level
    features and labels, along with optional batch information and covariates.
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
        include_donor_x: bool = True,
        split_donors: list[int] | None = None,
        split_indices: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        dd : DonorData object
            Donor-level and cell-level data object.
        donor_layer : str, optional
            Layer name in donor data for features.
        donor_labels_key : str, optional
            Key for donor labels in `dd`.
        donor_batch_key : str, optional
            Key for donor batch labels.
        donor_cat_covs_key : str, optional
            Key for donor categorical covariates.
        donor_cont_covs_key : str, optional
            Key for donor continuous covariates.
        donor_indices_key : str, optional
            Key for donor-specific indices.
        donor_id_key : str, optional
            Key for donor IDs.
        celltype_key : str, optional
            Key for cell type.
        cell_layer : str, optional
            Layer name in cell data for features.
        cell_labels_key : str, optional
            Key for cell labels.
        cell_batch_key : str, optional
            Key for cell batch labels.
        cell_cat_covs_key : str, optional
            Key for cell categorical covariates.
        cell_cont_covs_key : str, optional
            Key for cell continuous covariates.
        cell_indices_key : str, optional
            Key for cell indices.
        cell_donor_key : str, optional
            Key linking cells to donors.
        split_donors : list[int], optional
            List of donor IDs to include in the dataset.
        split_indices : list[int], optional
            List of indices of donors to include (alternative to `split_donors`).

        Raises
        ------
        ValueError
            If both `split_donors` and `split_indices` are provided.
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
        self.include_donor_x = include_donor_x

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
        donor_x = (
            self._get_layer(self.dd.G, mask=donor_mask, layer_key=self.donor_layer, force=False)
            if self.include_donor_x
            else None
        )
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
            "donor_x": _to_tensor(donor_x, squeeze=True),
            "donor_y": _to_tensor(donor_y),
            "donor_batch": _to_tensor(donor_batch),
            "donor_cat_covs": _to_tensor(donor_cat_covs),
            "donor_cont_covs": _to_tensor(donor_cont_covs),
            "donor_indices": _to_tensor(donor_indices),
            "cell_x": _to_tensor(cell_x),
            "cell_y": _to_tensor(cell_y),
            "cell_batch": _to_tensor(cell_batch),
            "cell_cat_covs": _to_tensor(cell_cat_covs),
            "cell_cont_covs": _to_tensor(cell_cont_covs),
            "cell_indices": _to_tensor(cell_indices),
        }

        return sample

    def _get_layer(
        self, data: Union[AnnData, MuData], layer_key: Optional[str], mask: np.ndarray = None, force: bool = False
    ) -> Any:
        """
        Retrieve feature matrix from donor or cell data.

        Parameters
        ----------
        data : AnnData or MuData
            Data object to extract the features from.
        layer_key : str, optional
            Layer name to extract. If None, uses `X` by default.
        mask : np.ndarray, optional
            Boolean array for selecting specific rows.
        force : bool, default=False
            If True, raises an error when the layer is missing. Otherwise, returns None.

        Returns
        -------
        np.ndarray or None
            Feature matrix or None if not found and `force=False`.
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
        Retrieve observation-level field (labels, batch, covariates) from data.

        Parameters
        ----------
        data : AnnData or MuData
            Data object to extract observation fields from.
        key : str
            Key for the field in `.obs`.
        mask : np.ndarray, optional
            Boolean array to select subset of observations.
        force : bool, default=False
            If True, raises an error if the key is missing. Otherwise, returns None.

        Returns
        -------
        np.ndarray or None
            Array of values for the specified key, optionally masked.
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


class DonorBagSampler(AnnBatchSampler):
    """AnnBatch sampler that yields one donor bag per split."""

    def __init__(
        self,
        bag_slices: list[slice],
        *,
        shuffle: bool = False,
        preload_nbags: int = 1,
        rng: np.random.Generator | None = None,
    ):
        if preload_nbags < 1:
            raise ValueError("preload_nbags must be at least 1.")
        self._bag_slices = bag_slices
        self._shuffle = shuffle
        self._preload_nbags = preload_nbags
        self._rng = rng or np.random.default_rng()

    @property
    def batch_size(self) -> None:
        return None

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def mask(self) -> slice:
        stop = self._bag_slices[-1].stop if self._bag_slices else 0
        return slice(0, stop)

    def n_iters(self, n_obs: int) -> int:
        return len(self._bag_slices)

    def validate(self, n_obs: int) -> None:
        if not self._bag_slices:
            raise ValueError("No donor bags were found for annbatch MIL loading.")
        if self._bag_slices[-1].stop > n_obs:
            raise ValueError(
                f"The last donor bag ends at {self._bag_slices[-1].stop}, but the loader only has {n_obs} cells."
            )

    def _sample(self, n_obs: int) -> Iterator[dict[str, list]]:
        order = np.arange(len(self._bag_slices))
        if self._shuffle:
            self._rng.shuffle(order)

        for bag_indices in np.array_split(order, max(1, int(np.ceil(len(order) / self._preload_nbags)))):
            chunks = [self._bag_slices[int(i)] for i in bag_indices]
            start = 0
            splits = []
            for chunk in chunks:
                size = chunk.stop - chunk.start
                splits.append(np.arange(start, start + size))
                start += size
            yield {"chunks": chunks, "splits": splits}


class AnnBatchMILDataset(IterableDataset):
    """MIL iterable dataset backed by annbatch for out-of-core cell loading."""

    def __init__(
        self,
        dd: DonorData,
        donor_layer: str | None = None,
        donor_labels_key: str | None = None,
        donor_batch_key: str | None = None,
        donor_cat_covs_key: str | None = None,
        donor_cont_covs_key: str | None = None,
        donor_indices_key: str | None = None,
        donor_id_key: str | None = None,
        cell_layer: str | None = None,
        cell_labels_key: str | None = None,
        cell_batch_key: str | None = None,
        cell_cat_covs_key: str | None = None,
        cell_cont_covs_key: str | None = None,
        cell_indices_key: str | None = None,
        cell_donor_key: str | None = None,
        include_donor_x: bool = True,
        split_donors: list[int] | None = None,
        split_indices: list[int] | None = None,
        shuffle: bool = False,
        preload_nbags: int = 1,
        preload_to_gpu: bool = False,
        to_torch: bool = True,
        rng: np.random.Generator | None = None,
    ):
        if Loader is None:
            raise ImportError(
                "AnnBatchMILDataset requires `annbatch`. Install it on Python 3.12+ with "
                "`pip install \"annbatch[torch,zarrs]\"`."
            )
        if split_donors is not None and split_indices is not None:
            raise ValueError("Both split_donors and split_indices cannot be provided at the same time.")

        self.dd = dd
        self.donor_layer = donor_layer
        self.include_donor_x = include_donor_x

        self.donor_labels_key = donor_labels_key or DAnn.DONOR_LABELS_KEY
        self.donor_batch_key = donor_batch_key or DAnn.DONOR_BATCH_KEY
        self.donor_cat_covs_key = donor_cat_covs_key or DAnn.DONOR_CAT_COVS_KEY
        self.donor_cont_covs_key = donor_cont_covs_key or DAnn.DONOR_CONT_COVS_KEY
        self.donor_indices_key = donor_indices_key or DAnn.DONOR_INDICES_KEY
        self.donor_id_key = donor_id_key or DAnn.DONOR_ID_KEY

        self.cell_labels_key = cell_labels_key or CAnn.CELL_LABELS_KEY
        self.cell_batch_key = cell_batch_key or CAnn.CELL_BATCH_KEY
        self.cell_cat_covs_key = cell_cat_covs_key or CAnn.CELL_CAT_COVS_KEY
        self.cell_cont_covs_key = cell_cont_covs_key or CAnn.CELL_CONT_COVS_KEY
        self.cell_indices_key = cell_indices_key or CAnn.CELL_INDICES_KEY
        self.cell_donor_key = cell_donor_key or CAnn.CELL_DONOR_KEY

        self.donor_ids = MILDataset._get_obs_field(self, self.dd.G, self.donor_id_key, force=True)
        if split_donors is not None:
            self.selected_ids = np.array(split_donors)
        elif split_indices is not None:
            self.selected_ids = self.donor_ids[split_indices]
        else:
            self.selected_ids = self.donor_ids

        self._selected_set = set(self.selected_ids.tolist())
        self._donor_lookup = {donor_id: i for i, donor_id in enumerate(self.donor_ids)}
        self._bag_specs = _get_contiguous_bag_slices(
            self.dd.C.obs[self.cell_donor_key].to_numpy(),
            self.selected_ids,
        )
        self._bag_lookup = {donor_id: bag_slice for donor_id, bag_slice in self._bag_specs}
        bag_slices = [bag_slice for _, bag_slice in self._bag_specs]
        self._sampler = DonorBagSampler(bag_slices, shuffle=shuffle, preload_nbags=preload_nbags, rng=rng)
        self._loader = Loader(
            batch_sampler=self._sampler,
            preload_to_gpu=preload_to_gpu,
            to_torch=to_torch,
        )
        self._loader.add_dataset(
            self._get_cell_dataset(cell_layer),
            obs=self._get_cell_obs(),
            var=self.dd.C.var,
        )

    def __len__(self):
        return len(self._bag_specs)

    def __iter__(self):
        for batch in self._loader:
            obs = batch["obs"]
            if obs is None or obs.empty:
                continue

            donor_id = obs[self.cell_donor_key].iloc[0]
            donor_mask = self.donor_ids == donor_id
            donor_x = (
                MILDataset._get_layer(self, self.dd.G, mask=donor_mask, layer_key=self.donor_layer, force=False)
                if self.include_donor_x
                else None
            )
            donor_y = MILDataset._get_obs_field(self, self.dd.G, self.donor_labels_key, mask=donor_mask, force=True)
            donor_batch = MILDataset._get_obs_field(self, self.dd.G, self.donor_batch_key, mask=donor_mask, force=False)
            donor_cat_covs = MILDataset._get_obs_field(
                self, self.dd.G, self.donor_cat_covs_key, mask=donor_mask, force=False
            )
            donor_cont_covs = MILDataset._get_obs_field(
                self, self.dd.G, self.donor_cont_covs_key, mask=donor_mask, force=False
            )
            donor_indices = MILDataset._get_obs_field(self, self.dd.G, self.donor_indices_key, mask=donor_mask, force=False)

            yield {
                "donor_id": donor_id,
                "donor_x": _to_tensor(donor_x, squeeze=True),
                "donor_y": _to_tensor(donor_y),
                "donor_batch": _to_tensor(donor_batch),
                "donor_cat_covs": _to_tensor(donor_cat_covs),
                "donor_cont_covs": _to_tensor(donor_cont_covs),
                "donor_indices": _to_tensor(donor_indices),
                "cell_x": _to_float_tensor(batch["X"]),
                "cell_y": _to_tensor(_extract_obs_array(obs, self.cell_labels_key)),
                "cell_batch": _to_tensor(_extract_obs_array(obs, self.cell_batch_key)),
                "cell_cat_covs": _to_tensor(_extract_obs_array(obs, self.cell_cat_covs_key)),
                "cell_cont_covs": _to_tensor(_extract_obs_array(obs, self.cell_cont_covs_key)),
                "cell_indices": _to_tensor(_extract_obs_array(obs, self.cell_indices_key)),
            }

    def _get_cell_dataset(self, cell_layer: str | None) -> Any:
        if cell_layer is not None:
            dataset = self.dd.C.layers.get(cell_layer)
        else:
            dataset = self.dd.C.X

        if dataset is None:
            slot = f"layer '{cell_layer}'" if cell_layer is not None else "X"
            raise ValueError(f"Could not find cell data in {slot}.")
        return dataset

    def _get_cell_obs(self):
        obs_keys = [
            self.cell_donor_key,
            self.cell_labels_key,
            self.cell_batch_key,
            self.cell_cat_covs_key,
            self.cell_cont_covs_key,
            self.cell_indices_key,
        ]
        obs_keys = [key for key in obs_keys if key is not None and key in self.dd.C.obs.columns]
        return self.dd.C.obs.loc[:, obs_keys].copy()


def mil_collate_fn(batch):
    """
    Custom collate function for MILDataset to prepare batched input for PyTorch models.

    Donor-level features are stacked into tensors, while cell-level features remain
    as lists of tensors corresponding to each donor. This preserves the MIL structure.

    Parameters
    ----------
    batch : list[dict]
        List of samples returned by MILDataset.__getitem__.

    Returns
    -------
    dict
        Batched sample with stacked donor-level tensors and list-based cell-level tensors.
        Keys include:
        - 'donor_x', 'donor_y', 'donor_batch', 'donor_cat_covs', 'donor_cont_covs', 'donor_indices'
        - 'cell_x', 'cell_y', 'cell_batch', 'cell_cat_covs', 'cell_cont_covs', 'cell_indices'
    """

    stack_fields = [
      "donor_id",
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
        sample[key] = values if key == "donor_id" else (torch.stack(values) if key in stack_fields else values)

    return sample
