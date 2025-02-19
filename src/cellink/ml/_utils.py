from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from lightning import LightningDataModule
import h5py
import pandas as pd
from scipy.sparse import issparse
import dask.array as da

from scvi import REGISTRY_KEYS, settings
from scvi.module.base import (
    BaseModuleClass,
)
from scvi.data import AnnDataManager
from scvi.dataloaders._data_splitting import DataSplitter
from scvi.model._utils import (
    get_max_epochs_heuristic,
    use_distributed_sampler,
)
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import devices_dsp
from scvi.data._anntorchdataset import AnnTorchDataset

from .._core import data_fields

try:
    # anndata >= 0.10
    from anndata.experimental import CSCDataset, CSRDataset
    SparseDataset = (CSRDataset, CSCDataset)
except ImportError:
    from anndata._core.sparse_dataset import SparseDataset

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

class DonorDataTorchDataset(AnnTorchDataset, Dataset):
    """Extension of :class:`~torch.utils.data.Dataset` and :class:`~torch.utils.data.AnnTorchDataset` for :class:`~cellink.DonorData` objects.
    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    getitem_tensors
        Specifies the keys in the data registry (``adata_manager.data_registry``) to return in
        ``__getitem__``. One of the following:
        * ``dict``: Keys correspond to keys in the data registry and values correspond to the
        desired :class:`~np.dtype` of the returned data.
        * ``list``: Elements correspond to keys in the data registry. Continuous data will be
        returned as :class:`~np.float32` and discrete data will be returned as :class:`~np.int64`.
        * ``None``: All registered data will be returned. Continuous data will be returned as
        :class:`~np.float32` and discrete data will be returned as :class:`~np.int64`.
    load_sparse_tensor
        ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
        :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
        GPUs, depending on the sparsity of the data.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
    ):
        super().__init__(adata_manager=adata_manager, getitem_tensors=getitem_tensors, load_sparse_tensor=load_sparse_tensor)

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Fetch data from the :class:`~cellink.DonorData` object.
        Parameters
        ----------
        indexes
            Indexes of the observations to fetch. Can be a single index, a list of indexes, or a
            slice.
        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_obs, ...)``.
        """
        if isinstance(indexes, int):
            indexes = [indexes]  # force batched single observations

        if self.adata_manager.adata.isbacked and isinstance(indexes, (list, np.ndarray)):
            # need to sort indexes for h5py datasets
            indexes = np.sort(indexes)

        donor_indexes = self.data[data_fields.DAnn.DONOR_ID_KEY].squeeze(axis=1)[indexes]
        cell_indexes = np.where(np.isin(self.data[data_fields.CAnn.CELL_DONOR_KEY].squeeze(axis=1), donor_indexes))[0]
        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]

            indexes_donor_cell = indexes if "donor" in key else cell_indexes

            if isinstance(data, (np.ndarray, h5py.Dataset)):
                sliced_data = data[indexes_donor_cell].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = data.iloc[indexes_donor_cell, :].to_numpy().astype(dtype, copy=False)
            elif issparse(data) or isinstance(data, SparseDataset):
                sliced_data = data[indexes_donor_cell].astype(dtype, copy=False)
                if self.load_sparse_tensor:
                    sliced_data = scipy_to_torch_sparse(sliced_data)
                else:
                    sliced_data = sliced_data.toarray()
            elif isinstance(data, da.Array):
                sliced_data = data[indexes_donor_cell].compute()
            elif isinstance(data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                # for minified  anndata, we need this because we can have a string
                # for `data``, which is the value of the MINIFY_TYPE_KEY in adata.uns,
                # used to record the type data minification
                # TODO: Adata manager should have a list of which fields it will load
                continue
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        return data_map


from scvi.model.base import UnsupervisedTrainingMixin
class TrainingMixin(UnsupervisedTrainingMixin):
    _training_plan_cls = TrainingPlan #MILTrainingPlan

    default_train_size = 0.9 
    default_batch_size = 1  

    def train(self, *args, train_size: float | None = None, batch_size: int | None = None, **kwargs):
        if train_size is None:
            train_size = self.default_train_size 
        if batch_size is None:
            batch_size = self.default_batch_size  

        super().train(*args, train_size=train_size, batch_size=batch_size, **kwargs)

class MILTrainingPlan(TrainingPlan):
    """Train vaes with adversarial loss option to encourage latent space mixing.
    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )

        self.automatic_optimization = False

        self.initialize_train_metrics()
        self.initialize_val_metrics()

    def training_step(self, batch, batch_idx):
        """Training step for training."""

        opt = self.optimizers()

        inference_outputs, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        #self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def validation_step(self, batch, batch_idx):
        """Validation step for training."""

        opt = self.optimizers()

        inference_outputs, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss

        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        #self.compute_and_log_metrics(scvi_loss, self.train_metrics, "valid")

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def initialize_train_metrics(self):
        """Initialize train related metrics."""
        self.mse_train = nn.MSELoss()
        self.auroc_train = torchmetrics.AUROC(task="binary")

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        self.mse_val = nn.MSELoss()
        self.auroc_val = torchmetrics.AUROC(task="binary")

    def configure_optimizers(self):
        """Configure optimizers for training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        return config1