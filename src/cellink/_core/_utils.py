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

try:
    # anndata >= 0.10
    from anndata.experimental import CSCDataset, CSRDataset
    SparseDataset = (CSRDataset, CSCDataset)
except ImportError:
    from anndata._core.sparse_dataset import SparseDataset

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

class DonorDataTorchDataset(AnnTorchDataset, Dataset):
    """Extension of :class:`~torch.utils.data.Dataset` for :class:`~anndata.AnnData` objects.

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
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)

        if adata_manager.adata is None:
            raise ValueError("Please run ``register_fields`` on ``adata_manager`` first.")
        self.adata_manager = adata_manager
        self.keys_and_dtypes = getitem_tensors
        self.load_sparse_tensor = load_sparse_tensor

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Fetch data from the :class:`~anndata.AnnData` object.

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

        donor_indexes = self.data['donor_patient'].squeeze(axis=1)[indexes]
        cell_indexes = np.where(np.isin(self.data['cell_patient'].squeeze(axis=1), donor_indexes))[0]
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

class TrainingMixin:
    """General purpose unsupervised train method.""" #NOt really needed except for TrainingPlan

    _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 1,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        datamodule: LightningDataModule | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs to train the model. The actual number of epochs may be
            less if early stopping is enabled. If ``None``, defaults to a heuristic based on
            :func:`~scvi.model.get_max_epochs_heuristic`. Must be passed in if ``datamodule`` is
            passed in, and it does not have an ``n_obs`` attribute.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range ``[0.0, 1.0]``. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        validation_size
            Size of the test set. If ``None``, defaults to ``1 - train_size``. If
            ``train_size + validation_size < 1``, the remaining cells belong to a test set. Passed
            into :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        shuffle_set_split
            Whether to shuffle indices before splitting. If ``False``, the val, train, and test set
            are split in the sequential order of the data according to ``validation_size`` and
            ``train_size`` percentages. Passed into :class:`~scvi.dataloaders.DataSplitter`. Not
            used if ``datamodule`` is passed in.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        batch_size
            Minibatch size to use during training. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        early_stopping
            Perform early stopping. Additional arguments can be passed in through ``**kwargs``.
            See :class:`~scvi.train.Trainer` for further options.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
            Values in this argument can be overwritten by arguments directly passed into this
            method, when appropriate. Not used if ``datamodule`` is passed in.
        plan_kwargs
            Additional keyword arguments passed into :class:`~scvi.train.TrainingPlan`. Values in
            this argument can be overwritten by arguments directly passed into this method, when
            appropriate.
        datamodule
            ``EXPERIMENTAL`` A :class:`~lightning.pytorch.core.LightningDataModule` instance to use
            for training in place of the default :class:`~scvi.dataloaders.DataSplitter`. Can only
            be passed in if the model was not initialized with :class:`~anndata.AnnData`.
        **kwargs
           Additional keyword arguments passed into :class:`~scvi.train.Trainer`.
        """
        if datamodule is not None and not self._module_init_on_train:
            raise ValueError(
                "Cannot pass in `datamodule` if the model was initialized with `adata`."
            )
        elif datamodule is None and self._module_init_on_train:
            raise ValueError(
                "If the model was not initialized with `adata`, a `datamodule` must be passed in."
            )

        if max_epochs is None:
            if datamodule is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            elif hasattr(datamodule, "n_obs"):
                max_epochs = get_max_epochs_heuristic(datamodule.n_obs)
            else:
                raise ValueError(
                    "If `datamodule` does not have `n_obs` attribute, `max_epochs` must be "
                    "passed in."
                )

        if datamodule is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            datamodule = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                shuffle_set_split=shuffle_set_split,
                distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
                load_sparse_tensor=load_sparse_tensor,
                **datasplitter_kwargs,
            )
        elif self.module is None:
            self.module = self._module_cls(
                datamodule.n_vars,
                n_batch=datamodule.n_batch,
                n_labels=getattr(datamodule, "n_labels", 1),
                n_continuous_cov=getattr(datamodule, "n_continuous_cov", 0),
                n_cats_per_cov=getattr(datamodule, "n_cats_per_cov", None),
                **self._module_kwargs,
            )

        plan_kwargs = plan_kwargs or {}
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=datamodule,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

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
        """Training step for adversarial training."""

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
