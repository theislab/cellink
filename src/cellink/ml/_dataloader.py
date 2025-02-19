from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch.nn import functional as F
from anndata import AnnData

from scvi import settings
from .._core import DonorData
from scvi.data import AnnDataManager, AnnDataManagerValidationCheck
from scvi.data.fields import (
    MuDataLayerField, MuDataCategoricalObsField,
    MuDataCategoricalJointObsField, MuDataNumericalJointObsField, MuDataNumericalObsField,
    AnnDataField
)
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data._utils import (
    _check_if_view
)
from scvi.module.base import LossOutput, auto_move_data
from scvi.model.base import BaseModelClass
from scvi.nn import FCLayers
from scvi.train._callbacks import SaveBestState
from scvi.utils._docstrings import devices_dsp, setup_anndata_dsp
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.module.base._base_module import _get_dict_if_none

from ._utils import TrainingMixin, MILTrainingPlan, DonorDataTorchDataset, AnnTorchDataset
from .._core import data_fields


class DonorDataManager(AnnDataManager):
    def __init__(
            self,
            fields: list[AnnDataField] | None = None,
            setup_method_args: dict | None = None,
            validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        super().__init__(fields=fields, setup_method_args=setup_method_args, validation_checks=validation_checks)

    def _validate_anndata_object(self, dd: DonorData):
        """For a given DonorData object, runs general scvi-tools compatibility checks."""
        if self.validation_checks.check_if_view:
            _check_if_view(dd.G, copy_if_view=False)
            _check_if_view(dd.C, copy_if_view=False)

    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool] = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
    ) -> DonorDataTorchDataset:
        """
        Creates a torch dataset from the DonorData object registered with this instance.
        Parameters
        ----------
        indices
            The indices of the observations in the adata to use
        data_and_attributes
            Dictionary with keys representing keys in data registry
            (``adata_manager.data_registry``) and value equal to desired numpy loading type (later
            made into torch tensor) or list of such keys. A list can be used to subset to certain
            keys in the event that more tensors than needed have been registered. If ``None``,
            defaults to all registered data.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.
        Returns
        -------
        :class:`~scvi.data.AnnTorchDataset`
        """
        dataset = DonorDataTorchDataset(
            self,
            getitem_tensors=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )
        if indices is not None:
            # This is a lazy subset, it just remaps indices
            dataset = Subset(dataset, indices)
        return dataset

class DonorDataModel(nn.Module):
    """
    DonorDataModel
    """

    def __init__(
        self,
        n_input_genes: int = 0,
        n_input_snps: int = 0,
        n_batch: int = 0,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        encode_covariates: bool = True,
        n_hidden: int = 0,
        n_output: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()

        self.n_input_genes = n_input_genes
        self.n_input_snps = n_input_snps
        self.n_batch = n_batch
        self.n_continuous_cov = n_continuous_cov
        self.n_cats_per_cov = n_cats_per_cov
        self.encode_covariates = encode_covariates
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn

        n_cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None and n_cats_per_cov != 0 else []
        n_cat_list = n_cat_list if encode_covariates else None

        input_snp = self.n_input_snps
        n_input_encoder_snp = input_snp + n_continuous_cov * encode_covariates
        self.encoder_genotype = FCLayers(
            n_in=n_input_encoder_snp,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            activation_fn=activation_fn,
            use_batch_norm=False,
        )

        dropout_rate = 0.2
        attention_dim = 64
        attention_branches = 1
        input_exp = self.n_input_genes
        n_input_encoder_exp = input_exp + n_continuous_cov * encode_covariates
        #self.encoder_expression = FCLayers(
        #    n_in=n_input_encoder_exp,
        #    n_out=n_hidden,
        #    n_cat_list=n_cat_list,
        #    activation_fn=activation_fn,
        #)
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_input_encoder_exp, n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, attention_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(attention_dim, attention_branches)
        )

        self.prediction_head = FCLayers(
            n_in=n_hidden*2,
            n_out=n_output,
            activation_fn=activation_fn,
            use_batch_norm=False,
        )

    @property
    def device(self):
        device = list({p.device for p in self.parameters()})
        if len(device) > 1:
            raise RuntimeError("Module tensors on multiple devices.")
        return device[0]

    def on_load(self, model):
        """Callback function run in :meth:`~scvi.model.base.BaseModelClass.load`."""

    @auto_move_data
    def forward(
        self,
        tensors,
        get_inference_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss=True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, LossOutput]:
        """Forward pass through the network.
        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        inference_kwargs
            Keyword args for ``inference()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        inference_kwargs = _get_dict_if_none(inference_kwargs)
        loss_kwargs = _get_dict_if_none(loss_kwargs)
        get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)

        if compute_loss:
            losses = self.loss(tensors, inference_outputs, **loss_kwargs)
            return inference_outputs, losses
        else:
            return inference_outputs

    def _get_inference_input(self, tensors, **kwargs):

        donor_x = tensors[data_fields.DAnn.DONOR_X_KEY]
        cell_x = tensors[data_fields.CAnn.CELL_X_KEY]
        donor_labels = tensors[data_fields.DAnn.DONOR_LABELS_KEY]
        donor_batch = tensors[data_fields.DAnn.DONOR_BATCH_KEY]
        donor_cat_covs = tensors.get(data_fields.DAnn.DONOR_CONT_COVS_KEY)
        donor_cont_covs = tensors.get(data_fields.DAnn.DONOR_CONT_COVS_KEY)
        donor_indices = tensors[data_fields.DAnn.DONOR_INDICES_KEY]
        donor_id = tensors[data_fields.DAnn.DONOR_ID_KEY]
        cell_labels = tensors[data_fields.CAnn.CELL_LABELS_KEY]
        cell_batch = tensors[data_fields.CAnn.CELL_BATCH_KEY]
        cell_cat_covs = tensors.get(data_fields.CAnn.CELL_CONT_COVS_KEY)
        cell_cont_covs = tensors.get(data_fields.CAnn.CELL_CONT_COVS_KEY)
        cell_indices = tensors[data_fields.CAnn.CELL_INDICES_KEY]
        cell_donor = tensors[data_fields.CAnn.CELL_DONOR_KEY]

        input_dict = {
            "donor_x": donor_x,
            "cell_x": cell_x,
            "donor_labels": donor_labels,
            "donor_batch": donor_batch,
            "donor_cat_covs": donor_cat_covs,
            "donor_cont_covs": donor_cont_covs,
            "donor_indices": donor_indices,
            "donor_id": donor_id,
            "cell_labels": cell_labels,
            "cell_batch": cell_batch,
            "cell_cat_covs": cell_cat_covs,
            "cell_cont_covs": cell_cont_covs,
            "cell_indices": cell_indices,
            "cell_donor": cell_donor,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        donor_x,
        cell_x,
        donor_labels,
        donor_batch,
        donor_cat_covs,
        donor_cont_covs,
        donor_indices,
        donor_id,
        cell_labels,
        cell_batch,
        cell_cat_covs,
        cell_cont_covs,
        cell_indices,
        cell_donor,
    ) -> dict[str, torch.Tensor]:
        """Run the inference model."""

        if donor_cont_covs is not None and self.encode_covariates:
            donor_x = torch.cat((cell_x, donor_cont_covs), dim=-1)
        if cell_cont_covs is not None and self.encode_covariates:
            cell_x = torch.cat((cell_x, cont_covs), dim=-1)

        if donor_cat_covs is not None and self.encode_covariates:
            donor_cat_covs = torch.split(donor_cat_covs, 1, dim=1)
        else:
            donor_cat_covs = ()
        if cell_cat_covs is not None and self.encode_covariates:
            cell_cat_covs = torch.split(cell_cat_covs, 1, dim=1)
        else:
            cell_cat_covs = ()

        latent_genotype = self.encoder_genotype(donor_x.float())
        #latent_expression = self.encoder_expression(cell_x) #cell_cat_covs
        H = self.feature_extractor(cell_x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)  #
        latent_expression = torch.mm(A, H)

        latent = torch.concat([latent_genotype, latent_expression], axis=1)
        logits = self.prediction_head(latent)

        outputs = {
            "latent_genotype": latent_genotype,
            "latent_expression": latent_expression,
            "logits": logits
        }

        return outputs

    def loss(self, tensors, inference_outputs, kl_weight: float = 1.0):
        """Computes the loss function for the model."""
        # Get the data
        donor_labels = tensors[data_fields.DAnn.DONOR_LABELS_KEY]
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(inference_outputs["logits"], donor_labels)

        return loss

class DonorDataBaseModel(TrainingMixin, BaseModelClass):
    """
    DonorDataBaseModel
    """

    _module_cls = DonorDataModel
    _training_plan_cls = MILTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_input_genes: int = 0,
        n_input_snps: int = 0,
        n_batch: int = 0,
        n_continuous_cov: int = 0,
        n_cats_per_cov: int = 0,
        encode_covariates: bool = True,
        n_hidden: int = 0,
        n_output: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = self._module_cls(
            n_input_genes=n_input_genes,
            n_input_snps=n_input_snps,
            n_batch=n_batch,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            encode_covariates=encode_covariates,
            n_hidden=n_hidden,
            n_output=n_output,
            activation_fn=activation_fn,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"Model with the following params: \n: None."
        )


    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        #train_idx: np.ndarray | list | None = None,
        #validation_idx: np.ndarray | list | None = None,
        #test_idx: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 1,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Trains the model using amortized variational inference.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            ``DEPRECATED`` Save the best model state with respect to the validation loss, or use
            the final state in the training procedure.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping`
            is `True`. If so, val is checked every epoch.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        Notes
        -----
        ``save_best`` is deprecated in v1.2 and will be removed in v1.3. Please use
        ``enable_checkpointing`` instead.
        """
        update_dict = {
            "lr": lr,
            "weight_decay": weight_decay,
            "eps": eps,
            "optimizer": "AdamW",
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

        if save_best:
            warnings.warn(
                "`save_best` is deprecated in v1.2 and will be removed in v1.3. Please use "
                "`enable_checkpointing` instead. See "
                "https://github.com/scverse/scvi-tools/issues/2568 for more details.",
                DeprecationWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor="reconstruction_loss_validation"))

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            #train_idx=train_idx,
            #validation_idx=validation_idx,
            #test_idx=test_idx,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="validation_loss",
            early_stopping_patience=50,
            #**kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        dd: DonorData,
        layer: str | None = None,
        donor_id_key: str | None = None,
        donor_labels_key: str | None = None,
        donor_batch_key: str | None = None,
        donor_categorical_covariate_keys: list[str] | None = None,
        donor_continuous_covariate_keys: list[str] | None = None,
        cell_donor_key: str | None = None,
        cell_labels_key: str | None = None,
        cell_batch_key: str | None = None,
        cell_categorical_covariate_keys: list[str] | None = None,
        cell_continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        """
        assert type(dd.C) == AnnData
        assert type(dd.G) == AnnData 
        setup_method_args = cls._get_setup_method_args(**locals())
        dd.C.obs["_indices"] = np.arange(dd.C.n_obs)
        dd.G.obs["_indices"] = np.arange(dd.G.n_obs)
        dd.mod = {"C": dd.C, "G": dd.G}
        anndata_fields = [
            MuDataLayerField(data_fields.DAnn.DONOR_X_KEY, layer, is_count_data=False, mod_key="G"),
            MuDataLayerField(data_fields.CAnn.CELL_X_KEY, layer, is_count_data=False, mod_key="C"),
            MuDataCategoricalObsField(data_fields.DAnn.DONOR_ID_KEY, donor_id_key, mod_key="G"),
            MuDataCategoricalObsField(data_fields.DAnn.DONOR_LABELS_KEY, donor_labels_key, mod_key="G"),
            MuDataCategoricalObsField(data_fields.DAnn.DONOR_BATCH_KEY, donor_batch_key, mod_key="G"),
            MuDataCategoricalJointObsField(data_fields.DAnn.DONOR_CAT_COVS_KEY, donor_categorical_covariate_keys, mod_key="G"),
            MuDataNumericalJointObsField(data_fields.DAnn.DONOR_CONT_COVS_KEY, donor_continuous_covariate_keys, mod_key="G"),
            MuDataNumericalObsField(data_fields.DAnn.DONOR_INDICES_KEY, "_indices", mod_key="G"),
            MuDataCategoricalObsField(data_fields.CAnn.CELL_DONOR_KEY, cell_donor_key, mod_key="C"),
            MuDataCategoricalObsField(data_fields.CAnn.CELL_LABELS_KEY, cell_labels_key, mod_key="C"),
            MuDataCategoricalObsField(data_fields.CAnn.CELL_BATCH_KEY, cell_batch_key, mod_key="C"),
            MuDataCategoricalJointObsField(data_fields.CAnn.CELL_CAT_COVS_KEY, cell_categorical_covariate_keys,  mod_key="C"),
            MuDataNumericalJointObsField(data_fields.CAnn.CELL_CONT_COVS_KEY, cell_continuous_covariate_keys, mod_key="C"),
            MuDataNumericalObsField(data_fields.CAnn.CELL_INDICES_KEY, "_indices", mod_key="C"),
        ]

        adata_manager = DonorDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(dd, **kwargs)
        cls.register_manager(adata_manager)

