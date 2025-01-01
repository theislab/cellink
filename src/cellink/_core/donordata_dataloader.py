"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Literal, Optional, Union

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import csr_matrix, vstack
from torch.distributions import Normal
from tqdm import tqdm
import pickle

import sys
sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools/src/")

from scvi import REGISTRY_KEYS, settings
from scvi._types import Number
from scvi.data import AnnDataManager, AnnDataManagerValidationCheck
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
    MuDataLayerField,
    MuDataCategoricalObsField,
    MuDataCategoricalJointObsField,
    MuDataNumericalJointObsField,
    MuDataNumericalObsField
)
from scvi.model._utils import (
    _get_batch_code_from_category,
    scatac_raw_counts_properties,
    scrna_raw_counts_properties,
)
from scvi.model.base import (
    ArchesMixinSparseVI,
    BaseModelClass,
    #UnsupervisedTrainingMixin,
    VAEMixin,
)
from ._utils import TrainingMixin
from scvi.model.base._de_core import _de_core
from scvi.module import SPARSEVAE
#from scvi.train import AdversarialTrainingPlan
from ._utils import MILTrainingPlan
from ._utils import DonorDataTorchDataset
from scvi.train._callbacks import SaveBestState
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp
from scvi.dataloaders._data_splitting import validate_data_split




import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld
from torch.nn import functional as F
import os
import sys

from ._utils import AnnTorchDataset

from ._constants import REGISTRY_KEYS
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial
)
from scvi.module._peakvae import Decoder as DecoderPeakVI
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers, RiggingtheLotteryLayers, SemiSparseLinearLayers

###

import sys
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from uuid import uuid4

import numpy as np
import pandas as pd
import rich
from mudata import MuData
from rich import box
from rich.console import Console
from torch.utils.data import Subset

import scvi
from scvi._types import AnnOrMuData
from scvi.utils import attrdict

from . import _constants
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data._utils import (
    _assign_adata_uuid,
    _check_if_view,
    _check_mudata_fully_paired,
    get_anndata_attribute,
)
from scvi.data.fields import AnnDataField
from scvi.module.base._base_module import _get_dict_if_none
"""
###

from __future__ import annotations

import warnings
import os
import sys
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

sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools/src/")

import scvi
from scvi import settings
from scvi._types import AnnOrMuData
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
from . import _constants
from ._constants import REGISTRY_KEYS


class DonorDataManager(AnnDataManager):
    def __init__(
            self,
            fields: list[AnnDataField] | None = None,
            setup_method_args: dict | None = None,
            validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        super().__init__()

        self.id = str(uuid4())
        self.adata = None
        self.fields = fields or []
        self.validation_checks = validation_checks or AnnDataManagerValidationCheck()
        self._registry = {
            _constants._SCVI_VERSION_KEY: scvi.__version__,
            _constants._MODEL_NAME_KEY: None,
            _constants._SETUP_ARGS_KEY: None,
            _constants._FIELD_REGISTRIES_KEY: defaultdict(dict),
        }
        if setup_method_args is not None:
            self._registry.update(setup_method_args)

    def _validate_anndata_object(self, dd: AnnOrMuData):
        """For a given AnnData object, runs general scvi-tools compatibility checks."""
        if self.validation_checks.check_if_view:
            _check_if_view(dd.gdata, copy_if_view=False)
            _check_if_view(dd.adata, copy_if_view=False)

    def register_fields(
        self,
        adata: AnnOrMuData,
        source_registry: dict | None = None,
        **transfer_kwargs,
    ):
        """Registers each field associated with this instance with the AnnData object.

        Either registers or transfers the setup from `source_setup_dict` if passed in.
        Sets ``self.adata``.

        Parameters
        ----------
        adata
            AnnData object to be registered.
        source_registry
            Registry created after registering an AnnData using an
            :class:`~scvi.data.AnnDataManager` object.
        transfer_kwargs
            Additional keywords which modify transfer behavior. Only applicable if
            ``source_registry`` is set.
        """
        if self.adata is not None:
            raise AssertionError("Existing AnnData object registered with this Manager instance.")

        if source_registry is None and transfer_kwargs:
            raise TypeError(
                f"register_fields() got unexpected keyword arguments {transfer_kwargs} passed "
                "without a source_registry."
            )

        self._validate_anndata_object(adata)

        for field in self.fields:
            self._add_field(
                field=field,
                adata=adata,
                source_registry=source_registry,
                **transfer_kwargs,
            )

        # Save arguments for register_fields.
        self._source_registry = deepcopy(source_registry)
        self._transfer_kwargs = deepcopy(transfer_kwargs)

        self.adata = adata
        self._assign_uuid()
        self._assign_most_recent_manager_uuid()

    def _add_field(
        self,
        field: AnnDataField,
        adata: AnnOrMuData,
        source_registry: dict | None = None,
        **transfer_kwargs,
    ):
        """Internal function for adding a field with optional transferring."""
        field_registries = self._registry[_constants._FIELD_REGISTRIES_KEY]
        field_registries[field.registry_key] = {
            _constants._DATA_REGISTRY_KEY: field.get_data_registry(),
            _constants._STATE_REGISTRY_KEY: {},
        }
        field_registry = field_registries[field.registry_key]

        # A field can be empty if the model has optional fields (e.g. extra covariates).
        # If empty, we skip registering the field.
        if not field.is_empty:
            # Transfer case: Source registry is used for validation and/or setup.
            if source_registry is not None:
                field_registry[_constants._STATE_REGISTRY_KEY] = field.transfer_field(
                    source_registry[_constants._FIELD_REGISTRIES_KEY][field.registry_key][
                        _constants._STATE_REGISTRY_KEY
                    ],
                    adata,
                    **transfer_kwargs,
                )
            else:
                field_registry[_constants._STATE_REGISTRY_KEY] = field.register_field(adata)
        # Compute and set summary stats for the given field.
        state_registry = field_registry[_constants._STATE_REGISTRY_KEY]
        field_registry[_constants._SUMMARY_STATS_KEY] = field.get_summary_stats(state_registry)


    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool] = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
    ) -> DonorDataTorchDataset:
        """
        Creates a torch dataset from the AnnData object registered with this instance.

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
    """Model

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.MULTIVI.setup_anndata`.

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

    def _get_inference_input(self, tensors):

        donor_x = tensors[REGISTRY_KEYS.DONOR_X_KEY]
        cell_x = tensors[REGISTRY_KEYS.CELL_X_KEY]
        donor_labels = tensors[REGISTRY_KEYS.DONOR_LABELS_KEY]
        donor_batch = tensors[REGISTRY_KEYS.DONOR_BATCH_KEY]
        donor_cat_covs = tensors.get(REGISTRY_KEYS.DONOR_CONT_COVS_KEY)
        donor_cont_covs = tensors.get(REGISTRY_KEYS.DONOR_CONT_COVS_KEY)
        donor_indices = tensors[REGISTRY_KEYS.DONOR_INDICES_KEY]
        donor_patient = tensors[REGISTRY_KEYS.DONOR_PATIENT_KEY]
        cell_labels = tensors[REGISTRY_KEYS.CELL_LABELS_KEY]
        cell_batch = tensors[REGISTRY_KEYS.CELL_BATCH_KEY]
        cell_cat_covs = tensors.get(REGISTRY_KEYS.CELL_CONT_COVS_KEY)
        cell_cont_covs = tensors.get(REGISTRY_KEYS.CELL_CONT_COVS_KEY)
        cell_indices = tensors[REGISTRY_KEYS.CELL_INDICES_KEY]
        cell_patient = tensors[REGISTRY_KEYS.CELL_PATIENT_KEY]

        input_dict = {
            "donor_x": donor_x,
            "cell_x": cell_x,
            "donor_labels": donor_labels,
            "donor_batch": donor_batch,
            "donor_cat_covs": donor_cat_covs,
            "donor_cont_covs": donor_cont_covs,
            "donor_indices": donor_indices,
            "donor_patient": donor_patient,
            "cell_labels": cell_labels,
            "cell_batch": cell_batch,
            "cell_cat_covs": cell_cat_covs,
            "cell_cont_covs": cell_cont_covs,
            "cell_indices": cell_indices,
            "cell_patient": cell_patient,
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
        donor_patient,
        cell_labels,
        cell_batch,
        cell_cat_covs,
        cell_cont_covs,
        cell_indices,
        cell_patient,
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
        donor_labels = tensors[REGISTRY_KEYS.DONOR_LABELS_KEY]
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(inference_outputs["logits"], donor_labels)

        return loss

class DonorDataBaseModel(TrainingMixin, BaseModelClass):
    """

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.MULTIVI.setup_anndata`.


    Examples
    --------
    >>>

    Notes
    -----
    *
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
        %(param_accelerator)s
        %(param_devices)s
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
        adversarial_mixing
            Whether to use adversarial training to penalize the model for umbalanced mixing of
            modalities.
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
            early_stopping_monitor="reconstruction_loss_validation",
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
        donor_patient_key: str | None = None,
        donor_labels_key: str | None = None,
        donor_batch_key: str | None = None,
        donor_categorical_covariate_keys: list[str] | None = None,
        donor_continuous_covariate_keys: list[str] | None = None,
        cell_patient_key: str | None = None,
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
        setup_method_args = cls._get_setup_method_args(**locals())
        dd.adata.obs["_indices"] = np.arange(dd.adata.n_obs)
        dd.gdata.obs["_indices"] = np.arange(dd.gdata.n_obs)
        anndata_fields = [
            MuDataLayerField(REGISTRY_KEYS.DONOR_X_KEY, layer, is_count_data=False, mod_key="gdata"),
            MuDataLayerField(REGISTRY_KEYS.CELL_X_KEY, layer, is_count_data=False, mod_key="adata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.DONOR_PATIENT_KEY, donor_patient_key, mod_key="gdata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.DONOR_LABELS_KEY, donor_labels_key, mod_key="gdata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.DONOR_BATCH_KEY, donor_batch_key, mod_key="gdata"),
            MuDataCategoricalJointObsField(REGISTRY_KEYS.DONOR_CAT_COVS_KEY, donor_categorical_covariate_keys, mod_key="gdata"),
            MuDataNumericalJointObsField(REGISTRY_KEYS.DONOR_CONT_COVS_KEY, donor_continuous_covariate_keys, mod_key="gdata"),
            MuDataNumericalObsField(REGISTRY_KEYS.DONOR_INDICES_KEY, "_indices", mod_key="gdata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.CELL_PATIENT_KEY, cell_patient_key, mod_key="adata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.CELL_LABELS_KEY, cell_labels_key, mod_key="adata"),
            MuDataCategoricalObsField(REGISTRY_KEYS.CELL_BATCH_KEY, cell_batch_key, mod_key="adata"),
            MuDataCategoricalJointObsField(REGISTRY_KEYS.CELL_CAT_COVS_KEY, cell_categorical_covariate_keys,  mod_key="adata"),
            MuDataNumericalJointObsField(REGISTRY_KEYS.CELL_CONT_COVS_KEY, cell_continuous_covariate_keys, mod_key="adata"),
            MuDataNumericalObsField(REGISTRY_KEYS.CELL_INDICES_KEY, "_indices", mod_key="adata"),
        ]

        adata_manager = DonorDataManager(fields=anndata_fields, setup_method_args=setup_method_args) #
        adata_manager.register_fields(dd, **kwargs)
        cls.register_manager(adata_manager)

    def _check_adata_modality_weights(self, adata):
        """Checks if adata is None and weights are per cell.

        :param adata: anndata object
        :return:
        """
        if (adata is not None) and (self.module.modality_weights == "cell"):
            raise RuntimeError("Held out data not permitted when using per cell weights")
