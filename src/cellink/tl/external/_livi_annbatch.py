from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from ._livi import LIVIRunner, get_livi_runner

if TYPE_CHECKING:
    from annbatch import DatasetCollection

logger = logging.getLogger(__name__)


def build_annbatch_collection(
    c_group,
    collection_path: str,
    *,
    seed: int = 42,
    **add_adatas_kwargs,
) -> DatasetCollection:
    """Build (if not already built) an annbatch `DatasetCollection` from a "C" group.

    Parameters
    ----------
    c_group
        The cell-level AnnData store to stream from -- a `zarr.Group` /
        `h5py.Group` (e.g. the "C" group of a DonorData `.dd.zarr`/`.dd.h5`
        cache: ``zarr.open(path, "r")["C"]``) or a path to a standalone
        AnnData file.
    collection_path
        Target ``.zarr`` path for the shuffled collection. If a collection
        already exists there, it is reused as-is (this function is a no-op).
    seed
        RNG seed for the initial shuffle (only used the first time).
    **add_adatas_kwargs
        Forwarded to ``DatasetCollection.add_adatas`` (e.g. ``dataset_size``,
        ``shard_size``, ``groupby``).

    Returns
    -------
    The (possibly newly built) `DatasetCollection`.
    """
    import anndata as ad

    from annbatch import DatasetCollection

    # annbatch's write_sharded requires zarr format 3 (anndata defaults to 2).
    ad.settings.zarr_write_format = 3

    collection = DatasetCollection(collection_path)
    if collection.is_empty:
        collection.add_adatas([c_group], rng=np.random.default_rng(seed), **add_adatas_kwargs)
    return collection


def read_g_from_dd_store(path: str):
    """Read only the "G" (genotype) group out of a `.dd.h5`/`.dd.zarr`, skipping "C"."""
    from anndata.io import read_elem

    path_str = str(path)
    if path_str.endswith(".zarr"):
        import zarr

        f = zarr.open(path_str, mode="r")
        if f["G"].attrs.get("encoding-type") != "anndata":
            raise ValueError(f"expected G to be stored as AnnData in {path}")
        return read_elem(f["G"])

    import h5py

    with h5py.File(path_str, "r") as f:
        if f["G"].attrs.get("encoding-type") != "anndata":
            raise ValueError(f"expected G to be stored as AnnData in {path}")
        return read_elem(f["G"])


def _resolve_indices(spec, names: Iterable[str], what: str) -> np.ndarray:
    """Int (first N) | sequence of names | boolean mask -> integer positions into `names`."""
    names = [str(n) for n in names]
    n = len(names)
    if isinstance(spec, int | np.integer):
        if spec > n:
            raise ValueError(f"requested {spec} {what} but only {n} available")
        return np.arange(int(spec))
    arr = np.asarray(spec)
    if arr.dtype == bool:
        if arr.shape[0] != n:
            raise ValueError(f"{what} mask length {arr.shape[0]} != {n} available")
        return np.flatnonzero(arr)
    name_to_pos = {nm: i for i, nm in enumerate(names)}  # first occurrence wins
    wanted = [str(s) for s in arr.tolist()]
    miss = [s for s in wanted if s not in name_to_pos]
    if miss:
        raise ValueError(f"{len(miss)} {what} not found, e.g. {miss[:3]}")
    return np.array([name_to_pos[s] for s in wanted], dtype=int)


def _categories(series: pd.Series) -> list:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(pd.unique(series))


def _code_tensor(column: pd.Series, dtype: pd.CategoricalDtype, what: str, device):
    import torch

    codes = column.astype(dtype).cat.codes.to_numpy().copy()  # .copy() ensures writable (cat.codes is read-only)
    if (codes < 0).any():
        raise ValueError(f"{what} value outside global categories")
    return torch.as_tensor(codes, dtype=torch.long, device=device)


class CisGenotype:
    """Donor-row-aligned cis-eQTL genotype + known-cis mask.

    Attributes
    ----------
    gt_array : torch.Tensor
        (n_donors x n_cis_snps) float32 presence, row-aligned to `donor_categories`.
    known_cis : torch.Tensor
        (n_cis_snps x n_genes) int64 mask over `gene_names`.
    """

    def __init__(
        self,
        donor_categories: Iterable[str],
        gene_names: Iterable[str],
        gdata,
        *,
        known_cis_eqtls: pd.DataFrame | None = None,
        cis_snps: int | Iterable[str] | np.ndarray = 2996,
        target_genes: int | Iterable[str] | np.ndarray = 2000,
        cis_genes_per_snp: int = 5,
        seed: int = 42,
        device: str = "cpu",
    ):
        import torch

        gene_names = [str(g) for g in gene_names]
        donor_categories = list(donor_categories)
        dev = torch.device(device)

        if known_cis_eqtls is not None:
            snp_idx = _resolve_indices(list(known_cis_eqtls.index), gdata.var_names, "cis SNPs")
            gene_pos = {g: i for i, g in enumerate(gene_names)}
            missing = [g for g in known_cis_eqtls.columns if g not in gene_pos]
            if missing:
                raise ValueError(f"{len(missing)} known_cis_eqtls genes not in gene_names, e.g. {missing[:3]}")
            known = np.zeros((len(snp_idx), len(gene_names)), dtype=np.int64)
            known[:, [gene_pos[g] for g in known_cis_eqtls.columns]] = known_cis_eqtls.to_numpy()
            self.known_cis = torch.from_numpy(known).to(dev)
        else:
            snp_idx = _resolve_indices(cis_snps, gdata.var_names, "cis SNPs")
            gene_idx = _resolve_indices(target_genes, gene_names, "target genes")
            self.known_cis = self._synthetic_known(len(snp_idx), len(gene_names), gene_idx, cis_genes_per_snp, seed).to(
                dev
            )

        self.gt_array = self._presence(gdata, snp_idx, donor_categories).to(dev)

    @property
    def n_cis_snps(self) -> int:
        return self.gt_array.shape[1]

    @staticmethod
    def _presence(gdata, snp_idx, donor_categories):
        import torch
        from anndata.utils import asarray

        raw = asarray(gdata[:, snp_idx].X)
        n_missing = int(np.isnan(raw).sum())
        if n_missing:
            logger.warning(
                "%d missing genotype calls (%.2f%%) among the selected cis SNPs; treating them as "
                "allele-absent (0) for this binary presence feature -- NaN > 0 is always False in "
                "numpy, so this was previously happening silently.",
                n_missing,
                100 * n_missing / raw.size,
            )
        gt_block = (raw > 0).astype(np.float32)  # NaN -> False -> 0, now logged above
        pos = {d: i for i, d in enumerate(gdata.obs_names)}
        missing = [c for c in donor_categories if c not in pos]
        if missing:
            raise ValueError(f"{len(missing)} expression donors absent from genotype, e.g. {missing[:3]}")
        rows = [pos[c] for c in donor_categories]
        return torch.from_numpy(np.ascontiguousarray(gt_block[rows]))

    @staticmethod
    def _synthetic_known(n_snps, n_genes, gene_idx, cis_genes_per_snp, seed):
        import torch

        rng = np.random.default_rng(seed)
        known = np.zeros((n_snps, n_genes), dtype=np.int64)
        for i in range(n_snps):
            known[i, rng.choice(gene_idx, size=cis_genes_per_snp, replace=False)] = 1
        return torch.from_numpy(known)


class LIVICisBatchAdapter:
    """Adapts an `annbatch.Loader` (cell expression) into LIVI's batch dict.

    Parameters
    ----------
    loader
        An `annbatch.Loader` built with ``return_index=True``.
    donor_codes, covar_codes
        CPU `torch.Tensor` lookup tables in `collection.obs()` row order
        (``donor_codes``: (n_obs,) int64; ``covar_codes``: ``{key: (n_obs,) int64}``).
    cis : CisGenotype
        Provides ``gt_array`` (genotype, row-aligned to donor codes) and
        ``known_cis``.
    covariate_keys
        Order in which covariate tensors are yielded (must match the model's
        ``covariates_dims`` order).
    donor_key, donor_dtype
        Used only for the one-time alignment self-check against the batch's
        own ``obs[donor_key]``.
    """

    def __init__(self, loader, donor_codes, covar_codes, cis: CisGenotype, covariate_keys, donor_key, donor_dtype):
        self._loader = loader
        self._donor_codes = donor_codes
        self._covar_codes = covar_codes
        self._cis = cis
        self._covariate_keys = covariate_keys
        self._donor_key = donor_key
        self._donor_dtype = donor_dtype
        self._checked = False

    def _check_index_alignment(self, batch, pos):
        obs = batch.get("obs")
        if obs is None:
            return
        expected = obs[self._donor_key].astype(self._donor_dtype).cat.codes.to_numpy()
        got = self._donor_codes[pos].cpu().numpy()
        if not np.array_equal(expected, got):
            raise RuntimeError(
                "batch['index'] does not align with collection.obs() row order; "
                "positional code gather would mislabel cells."
            )

    def __iter__(self):
        import torch

        dev = self._cis.gt_array.device
        for batch in self._loader:
            x = batch["X"]
            if x.layout != torch.strided:
                x = x.to_dense()
            x = x.to(dev)
            pos = torch.as_tensor(batch["index"], dtype=torch.long)
            y = self._donor_codes[pos].to(dev)
            if not self._checked:
                self._check_index_alignment(batch, pos)
                self._checked = True
            gt_cells = self._cis.gt_array[y]
            covariates = [self._covar_codes[k][pos].to(dev) for k in self._covariate_keys]
            yield {
                "x": x,
                "y": y,
                "size_factor": x.sum(dim=1, keepdim=True),
                "covariates": covariates,
                "GT_cells": gt_cells,
                "known_cis": self._cis.known_cis,
            }

    def __len__(self):
        return len(self._loader)


def load_x_with_index(group, donor_key: str):
    """Default ``load_adata`` for the collection: only X + the donor obs column."""
    import anndata as ad

    return ad.AnnData(
        X=ad.io.sparse_dataset(group["X"]),
        obs=ad.io.read_elem(group["obs"])[[donor_key]],
    )


class AnnbatchLIVIDataModule(pl.LightningDataModule):
    """`pytorch_lightning.LightningDataModule` streaming from a `DatasetCollection`."""

    def __init__(
        self,
        collection_path: str,
        donor_codes,
        covar_codes,
        cis: CisGenotype,
        covariate_keys: list[str],
        donor_key: str,
        donor_dtype,
        *,
        batch_size: int = 256,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        preload_to_gpu: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.collection_path = collection_path
        self.donor_codes = donor_codes
        self.covar_codes = covar_codes
        self.cis = cis
        self.covariate_keys = covariate_keys
        self.donor_key = donor_key
        self.donor_dtype = donor_dtype
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.preload_nchunks = preload_nchunks
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.preload_to_gpu = preload_to_gpu
        self.seed = seed
        self.collection = None

    def setup(self, stage=None):
        from annbatch import DatasetCollection

        self.collection = DatasetCollection(self.collection_path)

    def train_dataloader(self):
        from functools import partial

        from annbatch import Loader

        loader = Loader(
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            preload_nchunks=self.preload_nchunks,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            to_torch=True,
            preload_to_gpu=self.preload_to_gpu,
            rng=np.random.default_rng(self.seed),
            return_index=True,
        ).use_collection(self.collection, load_adata=partial(load_x_with_index, donor_key=self.donor_key))
        return LIVICisBatchAdapter(
            loader,
            self.donor_codes,
            self.covar_codes,
            self.cis,
            self.covariate_keys,
            self.donor_key,
            self.donor_dtype,
        )


def train_livi_annbatch(
    c_group,
    gdata,
    output_dir: str,
    collection_path: str,
    *,
    donor_key: str = "donor_id",
    covariates_keys: list[str] | None = None,
    known_cis_eqtls: pd.DataFrame | None = None,
    cis_snps: int | Iterable[str] | np.ndarray = 2996,
    target_genes: int | Iterable[str] | np.ndarray = 2000,
    cis_genes_per_snp: int = 5,
    z_dim: int = 15,
    n_dxc_factors: int = 100,
    n_persistent_factors: int = 5,
    encoder_hidden_dims: list[int] | None = None,
    learning_rate: float = 8e-4,
    warmup_epochs_vae: int = 30,
    warmup_epochs_G: int = 0,
    max_epochs: int = 200,
    min_epochs: int = 50,
    batch_size: int = 256,
    chunk_size: int = 512,
    preload_nchunks: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    preload_to_gpu: bool = True,
    seed: int = 42,
    l1_weight: float = 1e-3,
    A_weight: float = 1e-3,
    batch_norm_decoder: bool = False,
    genetics_seed: int | None = None,
    cell_state_cis: bool = True,
    log_every_n_steps: int = 10,
    enable_progress_bar: bool = True,
    enable_checkpointing: bool = False,
    enable_logger: bool = False,
    gradient_clip_val: float | None = None,
    accumulate_grad_batches: int = 1,
    limit_train_batches: int | float | None = None,
    deterministic: bool = False,
    callbacks: list | None = None,
    run: bool = True,
    runner: LIVIRunner | None = None,
) -> pl.Trainer | None:
    """Train a LIVI cis-eQTL model with `annbatch`-streamed cell expression.

    Counterpart to :func:`train_livi` for cell counts too large to load into
    RAM: expression is streamed from a pre-shuffled, on-disk `annbatch`
    `DatasetCollection` (built from `c_group` if `collection_path` doesn't
    exist yet) instead of being held fully in memory. Genotype (donors x
    SNPs) is still loaded into memory and looked up per cell, exactly like
    `train_livi`'s `eqtl_genotypes` -- it's orders of magnitude smaller than
    cell expression.

    Parameters
    ----------
    c_group
        Cell-level AnnData store to stream from -- e.g. the "C" group of a
        DonorData `.dd.zarr`/`.dd.h5` cache (``zarr.open(path, "r")["C"]``).
        Ignored if `collection_path` already has a built collection.
    gdata : AnnData
        Genotype (donors x SNPs) AnnData, e.g. from :func:`read_g_from_dd_store`.
    output_dir
        Unused for checkpointing currently (annbatch path doesn't checkpoint
        by default -- see Notes); kept for signature parity with `train_livi`.
    collection_path
        Path to the (possibly to-be-built) annbatch `DatasetCollection`.
    donor_key
        Column in the collection's `obs` carrying donor/individual IDs.
    covariates_keys
        Categorical covariate columns in the collection's `obs`.
    known_cis_eqtls
        Real SNPs x genes 0/1 mapping (e.g. from genomic-distance windows or a
        curated association list). If `None`, a synthetic random mapping is
        generated from `cis_snps`/`target_genes`/`cis_genes_per_snp` -- fine
        for smoke testing, not biologically meaningful.
    cis_snps, target_genes, cis_genes_per_snp
        See `CisGenotype`; only used when `known_cis_eqtls` is `None`.
    z_dim, n_dxc_factors, n_persistent_factors, encoder_hidden_dims,
    learning_rate, warmup_epochs_vae, warmup_epochs_G, l1_weight, A_weight,
    batch_norm_decoder, genetics_seed, cell_state_cis
        Model hyperparameters, same meaning as in `train_livi`.
    max_epochs, min_epochs, batch_size, chunk_size, preload_nchunks, shuffle,
    drop_last, preload_to_gpu, log_every_n_steps, enable_progress_bar,
    gradient_clip_val, accumulate_grad_batches, limit_train_batches, deterministic
        Training / dataloader configuration; same meaning as the matching
        `pytorch_lightning.Trainer` arguments (`limit_train_batches` is handy
        for benchmarking a fixed number of batches/epoch instead of a full pass).
    enable_checkpointing, enable_logger
        Both default to `False` (unlike a plain `pytorch_lightning.Trainer`,
        which defaults both to `True` and silently adds its own default
        `ModelCheckpoint` + `TensorBoardLogger`). Pass `callbacks=[ModelCheckpoint(...)]`
        and `enable_logger=True` explicitly if you want checkpointing/logging.
    callbacks
        Extra `pytorch_lightning.Callback` instances to attach to the `Trainer`
        (e.g. a throughput-logging callback, a `ModelCheckpoint`).
    seed
        Global random seed (model init, collection shuffle, loader shuffle).
    run
        If `False`, log the resolved configuration and return without training.
    runner
        `LIVIRunner` instance; uses the global runner when `None`.

    Returns
    -------
    pytorch_lightning.Trainer or None
        The fitted `Trainer` (e.g. to inspect `trainer.current_epoch` or a
        `ModelCheckpoint` callback you passed in), or `None` when `run=False`.

    Notes
    -----
    No `ModelCheckpoint` is added by default (unlike `train_livi`) -- pass one
    via `callbacks` if you need checkpointing.
    """
    if runner is None:
        runner = get_livi_runner()
    if encoder_hidden_dims is None:
        encoder_hidden_dims = [512, 256, 64]

    device = runner.resolve_device()
    LIVI = runner.get_livi_class()

    collection = build_annbatch_collection(c_group, collection_path, seed=seed)

    # Gene names come from the collection itself.
    first_var = None
    for group in collection:
        first_var = group["var"]
        break
    import anndata as ad

    gene_names = list(ad.io.read_elem(first_var).index)
    n_genes = len(gene_names)

    obs = collection.obs(columns=[donor_key] + (covariates_keys or []))
    donor_categories = _categories(obs[donor_key])
    donor_dtype = pd.CategoricalDtype(categories=donor_categories, ordered=False)
    n_donors = len(donor_categories)

    covar_dtypes = {
        k: pd.CategoricalDtype(categories=_categories(obs[k]), ordered=False) for k in (covariates_keys or [])
    }
    covariates_dims = [len(covar_dtypes[k].categories) for k in (covariates_keys or [])]

    if not run:
        logger.info(
            "LIVI annbatch training config (run=False):\n"
            "  x_dim=%d, y_dim=%d, z_dim=%d\n"
            "  n_dxc_factors=%d, n_persistent_factors=%d\n"
            "  covariates_keys=%s, covariates_dims=%s",
            n_genes,
            n_donors,
            z_dim,
            n_dxc_factors,
            n_persistent_factors,
            covariates_keys,
            covariates_dims,
        )
        return None

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    code_dev = "cpu"
    donor_codes = _code_tensor(obs[donor_key], donor_dtype, donor_key, code_dev)
    covar_codes = {k: _code_tensor(obs[k], covar_dtypes[k], k, code_dev) for k in (covariates_keys or [])}

    cis = CisGenotype(
        donor_categories,
        gene_names,
        gdata,
        known_cis_eqtls=known_cis_eqtls,
        cis_snps=cis_snps,
        target_genes=target_genes,
        cis_genes_per_snp=cis_genes_per_snp,
        seed=seed,
        device=device,
    )

    datamodule = AnnbatchLIVIDataModule(
        collection_path,
        donor_codes,
        covar_codes,
        cis,
        covariates_keys or [],
        donor_key,
        donor_dtype,
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=shuffle,
        drop_last=drop_last,
        preload_to_gpu=preload_to_gpu,
        seed=seed,
    )

    model = LIVI(
        x_dim=n_genes,
        z_dim=z_dim,
        y_dim=n_donors,
        n_DxC_factors=n_dxc_factors,
        n_persistent_factors=n_persistent_factors,
        n_cis_snps=cis.n_cis_snps,
        cell_state_cis=cell_state_cis,
        encoder_hidden_dims=encoder_hidden_dims,
        learning_rate=learning_rate,
        warmup_epochs_vae=warmup_epochs_vae,
        warmup_epochs_G=warmup_epochs_G,
        covariates_dims=covariates_dims,
        l1_weight=l1_weight,
        A_weight=A_weight,
        batch_norm_decoder=batch_norm_decoder,
        genetics_seed=genetics_seed,
        device=device,
    )

    os.makedirs(output_dir, exist_ok=True)
    from pytorch_lightning.callbacks import ModelCheckpoint

    if callbacks and any(isinstance(c, ModelCheckpoint) for c in callbacks):
        enable_checkpointing = True  # PL rejects enable_checkpointing=False + a ModelCheckpoint callback
    trainer_kwargs: dict = {
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "accelerator": "gpu" if device == "cuda" else "cpu",
        "devices": 1,
        "default_root_dir": output_dir,
        "callbacks": list(callbacks) if callbacks else None,
        "enable_checkpointing": enable_checkpointing,
        "logger": enable_logger,
        "log_every_n_steps": log_every_n_steps,
        "enable_progress_bar": enable_progress_bar,
        "accumulate_grad_batches": accumulate_grad_batches,
        "deterministic": deterministic,
    }
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, datamodule=datamodule)
    logger.info("LIVI annbatch training complete.")
    return trainer
