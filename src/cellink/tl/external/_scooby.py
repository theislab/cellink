from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


KNOWN_SCOOBY_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "johahi/neurips-scooby": {"cell_emb_dim": 14, "n_tracks": 3, "modality": "multiome", "use_transform_borzoi_emb": True},
    "lauradmartens/onek1k-scooby": {"cell_emb_dim": 10, "n_tracks": 2, "modality": "rna", "use_transform_borzoi_emb": True},
    "lauradmartens/epicardioids-scooby": {
        "cell_emb_dim": 50, "n_tracks": 3, "modality": "multiome", "use_transform_borzoi_emb": True,
    },
}

UCSC_TO_ENSEMBL_CHR_MAP: dict[str, str] = {f"chr{c}": str(c) for c in [*range(1, 23), "X"]}


class ScoobyRunner:
    """Manages device resolution and LoRA configuration for Scooby.

    Parameters
    ----------
    device : str
        Compute device: ``"auto"`` (detect GPU), ``"cuda"``, or ``"cpu"``.
    lora_config : "peft.LoraConfig", optional
        Forwarded to ``scooby.utils.utils.get_lora``. ``None`` (default)
        uses scooby's own default LoRA target-module pattern (the Borzoi
        trunk's conv layers + attention query/value projections).
    """

    def __init__(self, device: str = "auto", lora_config: Any | None = None):
        self.device = device
        self.lora_config = lora_config

    def resolve_device(self) -> str:
        """Return the resolved device string (``"cuda"`` or ``"cpu"``)."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def get_scooby_class(self):
        """Import and return the :class:`scooby.modeling.Scooby` class."""
        try:
            from scooby.modeling import Scooby
        except ImportError as e:
            raise ImportError(
                "scooby is required for this function. Install with:\n\n"
                "    pip install cellink[scooby]"
            ) from e
        return Scooby

    def get_lora_fn(self):
        """Import and return ``scooby.utils.utils.get_lora``."""
        try:
            from scooby.utils.utils import get_lora
        except ImportError as e:
            raise ImportError(
                "scooby is required for this function. Install with:\n\n"
                "    pip install cellink[scooby]"
            ) from e
        return get_lora


_scooby_runner: ScoobyRunner | None = None


def configure_scooby_runner(device: str = "auto", lora_config: Any | None = None) -> ScoobyRunner:
    """Configure the global Scooby runner.

    Must be called once before using any other Scooby function in thismodule.

    Parameters
    ----------
    device : str
        Compute device: ``"auto"`` (auto-detect GPU), ``"cuda"``, or
        ``"cpu"``.
    lora_config : "peft.LoraConfig", optional
        Forwarded to every ``get_lora`` call made through this runner.

    Returns
    -------
    ScoobyRunner

    Examples
    --------
    >>> import cellink as cl
    >>> cl.tl.external.configure_scooby_runner(device="cuda")
    """
    global _scooby_runner
    _scooby_runner = ScoobyRunner(device=device, lora_config=lora_config)
    return _scooby_runner


def get_scooby_runner() -> ScoobyRunner:
    """Return the global :class:`ScoobyRunner`, raising if not configured."""
    if _scooby_runner is None:
        raise RuntimeError("Scooby runner not configured. Call `cellink.tl.external.configure_scooby_runner()` first.")
    return _scooby_runner


def build_scooby_embedding(
    adata,
    n_comps: int = 16,
    *,
    use_hvg: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a per-cell PCA embedding in the shape Scooby's on-the-fly datasets expect.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell x gene count matrix (raw counts expected if ``normalize=True``).
    n_comps : int
        Embedding dimensionality (``cell_emb_dim`` in Scooby's model config).
    use_hvg : bool
        Restrict PCA to highly-variable genes first.
    n_top_genes : int
        Number of HVGs to select when ``use_hvg=True``.
    normalize : bool
        Run ``sc.pp.normalize_total`` + ``sc.pp.log1p`` before PCA. Set
        ``False`` if ``adata`` is already normalized.

    Returns
    -------
    pandas.DataFrame
        Columns: ``embedding`` (one array per row, same row count and order
        as ``adata``
    """
    import scanpy as sc

    total_counts = np.ravel(np.asarray(adata.X.sum(axis=1)))
    nonzero_mask = total_counts > 0
    if not nonzero_mask.all():
        logger.warning(
            "build_scooby_embedding: %d/%d cells have zero total counts; assigning them a zero embedding "
            "vector instead of dropping them, to keep row count/order aligned with `adata`.",
            (~nonzero_mask).sum(),
            adata.n_obs,
        )

    a = adata[nonzero_mask].copy() if not nonzero_mask.all() else adata.copy()
    if normalize:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    if use_hvg:
        sc.pp.highly_variable_genes(a, n_top_genes=n_top_genes)
        a = a[:, a.var["highly_variable"]].copy()
    sc.tl.pca(a, n_comps=n_comps)
    nonzero_emb = a.obsm["X_pca"].astype(np.float32)

    emb = np.zeros((adata.n_obs, n_comps), dtype=np.float32)
    emb[nonzero_mask] = nonzero_emb
    return pd.DataFrame({"embedding": list(emb)}, index=adata.obs_names)


def build_scooby_embedding_scpoli(
    adata,
    condition_key: str,
    cell_type_key: str,
    *,
    latent_dim: int = 16,
    recon_loss: str = "nb",
    n_epochs: int = 50,
    pretraining_epochs: int = 40,
    early_stopping: bool = True,
    use_hvg: bool = True,
    n_top_genes: int = 3000,
    checkpoint_dir: str | None = None,
) -> pd.DataFrame:
    """Build a per-cell embedding via **scPoli** (``scarches.models.scpoli``),
    matching the real recipe the released OneK1K Scooby checkpoint's embedding
    was trained with. Requires raw (unnormalized) counts in ``adata.X``.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell x gene raw-count matrix.
    condition_key : str
        ``adata.obs`` column identifying the batch/sample to condition on
        (``condition_keys`` in scPoli's own API e.g. donor or sample ID;
        the reference recipe uses ``'sample'``).
    cell_type_key : str
        ``adata.obs`` column with cell-type labels (``cell_type_keys`` in
        scPoli's own API; the reference recipe uses ``'cell_label'``).
    latent_dim : int
        Embedding dimensionality (``cell_emb_dim`` in Scooby's model config).
    recon_loss : str
        scPoli reconstruction loss; ``'nb'`` (negative binomial) matches the
        reference recipe.
    n_epochs, pretraining_epochs : int
        scPoli training schedule; defaults match the reference recipe
        (``n_epochs=50``, ``pretraining_epochs=40``).
    early_stopping : bool
        Use the reference recipe's early-stopping config
        (``val_prototype_loss``, patience 20, LR-reduce patience 13).
    use_hvg : bool
        Restrict to highly-variable genes first, matching the reference
        recipe (``sc.pp.highly_variable_genes(..., flavor='seurat_v3',
        batch_key=condition_key)``.
    n_top_genes : int
        Number of HVGs to select when ``use_hvg=True``; the reference
        recipe uses 3000.
    checkpoint_dir : str, optional
        If given: skip training and load a previously-trained model from
        this directory if it already exists (via ``scPoli.load``, real
        scPoli training on a full atlas-scale cohort can run for hours, and
        without this, any failure after training..

    Returns
    -------
    pandas.DataFrame
        Same contract as :func:`build_scooby_embedding`: a single
        ``'embedding'`` column, one array per row, row-aligned to
        ``adata.obs_names``.
    """
    import scanpy as sc
    import anndata as _ad

    if not hasattr(_ad, "_cellink_read_shim_applied"):
        _ad.read = _ad.read_h5ad  # scarches 0.6.x calls the removed anndata.read()
        _ad._cellink_read_shim_applied = True
    from scarches.models.scpoli import scPoli

    import gc

    import scipy.sparse

    a = adata
    if use_hvg:
        assert scipy.sparse.issparse(a.X), (
            f"expected a.X to stay sparse into the HVG step, got {type(a.X)}, "
            "a dense copy here at full-cohort scale (7M+ cells) would itself "
            "exhaust available memory long before any real peak."
        )
        sc.pp.filter_genes(a, min_cells=10)
        gc.collect()
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top_genes, batch_key=condition_key)
        gc.collect()
        a.X = a.X.tocsc()
        a = a[:, a.var["highly_variable"]].copy()
        gc.collect()
        adata.X = scipy.sparse.csr_matrix((adata.n_obs, adata.n_vars), dtype=adata.X.dtype)
        gc.collect()

    if scipy.sparse.issparse(a.X):
        a.X = np.asarray(a.X.todense(), dtype=np.float32)
        gc.collect()

    partial_marker = Path(checkpoint_dir, "_partial_epoch") if checkpoint_dir is not None else None
    is_partial_checkpoint = partial_marker is not None and partial_marker.exists()

    if checkpoint_dir is not None and Path(checkpoint_dir).exists() and not is_partial_checkpoint:
        import torch

        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        logger.info(
            "loading previously-trained scPoli model from %s (skipping training, map_location=%s)",
            checkpoint_dir, map_location,
        )
        model = scPoli.load(checkpoint_dir, adata=a, map_location=map_location)
    else:
        completed_epochs = 0
        if is_partial_checkpoint:
            import torch

            completed_epochs = int(partial_marker.read_text())
            map_location = None if torch.cuda.is_available() else torch.device("cpu")
            logger.info(
                "Scooby scPoli embedding: resuming from a PARTIAL checkpoint at %s (%d/%d epochs "
                "already done), warm-starting from these weights with a fresh optimizer for the "
                "remaining epochs (not an exact optimizer/scheduler resume like the RNA training "
                "loop's, since scPoliTrainer has no state_dict for that, but far better than losing "
                "the whole run)", checkpoint_dir, completed_epochs, n_epochs,
            )
            model = scPoli.load(checkpoint_dir, adata=a, map_location=map_location)
        else:
            model = scPoli(
                adata=a,
                condition_keys=condition_key,
                cell_type_keys=cell_type_key,
                recon_loss=recon_loss,
                latent_dim=latent_dim,
            )

        remaining_epochs = max(1, n_epochs - completed_epochs)
        remaining_pretraining_epochs = max(0, pretraining_epochs - completed_epochs)
        early_stopping_kwargs = (
            {
                "early_stopping_metric": "val_prototype_loss",
                "mode": "min",
                "threshold": 0,
                "patience": 20,
                "reduce_lr": True,
                "lr_patience": 13,
                "lr_factor": 0.1,
            }
            if early_stopping
            else None
        )
        train_kwargs: dict[str, Any] = {"n_epochs": remaining_epochs, "pretraining_epochs": remaining_pretraining_epochs}
        if early_stopping_kwargs is not None:
            train_kwargs["early_stopping_kwargs"] = early_stopping_kwargs

        if checkpoint_dir is not None:
            from scarches.trainers.scpoli.trainer import scPoliTrainer

            _original_on_epoch_end = scPoliTrainer.on_epoch_end
            _original_on_iteration = scPoliTrainer.on_iteration
            _checkpoint_every_n_epochs = 5
            _checkpoint_every_n_iters = 2000

            def _on_epoch_end_with_checkpoint(trainer_self):
                _original_on_epoch_end(trainer_self)
                if (trainer_self.epoch + 1) % _checkpoint_every_n_epochs == 0:
                    real_epoch = completed_epochs + trainer_self.epoch + 1
                    logger.info(
                        "Scooby scPoli embedding: periodic checkpoint at epoch %d/%d -> %s",
                        real_epoch, n_epochs, checkpoint_dir,
                    )
                    model.save(checkpoint_dir, overwrite=True)
                    Path(checkpoint_dir, "_partial_epoch").write_text(str(real_epoch))

            def _on_iteration_with_checkpoint(trainer_self, batch_data):
                _original_on_iteration(trainer_self, batch_data)
                if trainer_self.iter > 0 and trainer_self.iter % _checkpoint_every_n_iters == 0:
                    logger.info(
                        "Scooby scPoli embedding: mid-epoch checkpoint at epoch %d (iter %d/%d) -> %s "
                        "(warm-start snapshot, not a completed-epoch boundary)",
                        completed_epochs + trainer_self.epoch, trainer_self.iter, trainer_self.iters_per_epoch, checkpoint_dir,
                    )
                    model.save(checkpoint_dir, overwrite=True)
                    Path(checkpoint_dir, "_partial_epoch").write_text(str(completed_epochs + trainer_self.epoch))

            scPoliTrainer.on_epoch_end = _on_epoch_end_with_checkpoint
            scPoliTrainer.on_iteration = _on_iteration_with_checkpoint
            try:
                model.train(**train_kwargs)
            finally:
                scPoliTrainer.on_epoch_end = _original_on_epoch_end
                scPoliTrainer.on_iteration = _original_on_iteration
        else:
            model.train(**train_kwargs)

        if checkpoint_dir is not None:
            logger.info("Training done. saving model to %s.", checkpoint_dir)
            model.save(checkpoint_dir, overwrite=True)
            if Path(checkpoint_dir, "_partial_epoch").exists():
                Path(checkpoint_dir, "_partial_epoch").unlink()

    if scipy.sparse.issparse(a.X):
        a.X = np.asarray(a.X.todense())

    emb = model.get_latent(a, mean=True).astype(np.float32)
    return pd.DataFrame({"embedding": list(emb)}, index=adata.obs_names)


def build_scooby_dataset(
    rna_plus,
    rna_minus,
    embedding: pd.DataFrame,
    genome_intervals,
    *,
    neighbors=None,
    cell_sample_size: int = 64,
    cell_weights=None,
    clip_soft: float = 5,
    get_targets: bool = True,
    random_cells: bool = True,
    cells_to_run=None,
    custom_read_length: int = 90,
):
    """Build an RNA-only on-the-fly training/eval dataset.

    Thin wrapper around ``scooby.data.onTheFlyDataset``. ``rna_plus``/
    ``rna_minus`` should be backed handles from ``scooby.utils.utils.read_backed``
    (e.g. ``read_backed(h5py.File(path), "fragment_single")``) so the
    dataset streams cells rather than loading the full matrix.
    """
    try:
        from scooby.data import onTheFlyDataset
    except ImportError as e:
        raise ImportError(
            "scooby is required for this function. Install with:\n\n"
            "    pip install cellink[scooby]"
        ) from e
    return onTheFlyDataset(
        rna_plus,
        rna_minus,
        neighbors=neighbors,
        embedding=embedding,
        ds=genome_intervals,
        cell_sample_size=cell_sample_size,
        cell_weights=cell_weights,
        clip_soft=clip_soft,
        get_targets=get_targets,
        random_cells=random_cells,
        cells_to_run=cells_to_run,
        custom_read_length=custom_read_length,
    )


def build_scooby_multiome_dataset(
    adatas: dict[str, Any],
    embedding: pd.DataFrame,
    genome_intervals,
    *,
    neighbors=None,
    cell_sample_size: int = 64,
    cell_weights=None,
    clip_soft: float = 5,
    normalize_atac: bool = False,
    get_targets: bool = True,
    random_cells: bool = True,
    cells_to_run=None,
    custom_read_length: int = 90,
):
    """Build an RNA+ATAC multiome on-the-fly training/eval dataset.

    Thin wrapper around ``scooby.data.onTheFlyMultiomeDataset``.

    Parameters
    ----------
    adatas : dict
        Backed handles keyed by modality name containing ``"rna"``/``"atac"``
        (scooby dispatches on this substring internally), e.g.
        ``{"rna_plus": ..., "rna_minus": ..., "atac": ...}``, each from
        ``scooby.utils.utils.read_backed`` (RNA: ``obsm["fragment_single"]``;
        ATAC: ``obsm["insertion"]``).
    normalize_atac : bool
        Scale ATAC coverage (x0.05) for training stability, matching
        scooby's own reference multiome training script.
    """
    try:
        from scooby.data import onTheFlyMultiomeDataset
    except ImportError as e:
        raise ImportError(
            "scooby is required for this function. Install with:\n\n"
            "    pip install cellink[scooby]"
        ) from e
    return onTheFlyMultiomeDataset(
        adatas,
        embedding=embedding,
        ds=genome_intervals,
        neighbors=neighbors,
        cell_sample_size=cell_sample_size,
        cell_weights=cell_weights,
        clip_soft=clip_soft,
        normalize_atac=normalize_atac,
        get_targets=get_targets,
        random_cells=random_cells,
        cells_to_run=cells_to_run,
        custom_read_length=custom_read_length,
    )


def _make_genome_interval_datasets(
    sequences_path, genome_path, *, test_fold, val_fold, context_length, shift_augs, rc_aug, chr_bed_to_fasta_map=None
):
    """Shared train/val ``GenomeIntervalDataset`` builder for both training
    entrypoints below."""
    import polars as pl
    from enformer_pytorch.data import GenomeIntervalDataset

    chr_bed_to_fasta_map = chr_bed_to_fasta_map or {}

    def filter_train(df):
        return df.filter((pl.col("column_4") != f"fold{test_fold}") & (pl.col("column_4") != f"fold{val_fold}"))

    def filter_val(df):
        return df.filter(pl.col("column_4") == f"fold{val_fold}")

    train_ds = GenomeIntervalDataset(
        bed_file=sequences_path,
        fasta_file=genome_path,
        filter_df_fn=filter_train,
        return_seq_indices=False,
        shift_augs=shift_augs,
        rc_aug=rc_aug,
        return_augs=True,
        context_length=context_length,
        chr_bed_to_fasta_map=chr_bed_to_fasta_map,
    )
    val_ds = GenomeIntervalDataset(
        bed_file=sequences_path,
        fasta_file=genome_path,
        filter_df_fn=filter_val,
        return_seq_indices=False,
        shift_augs=(0, 0),
        rc_aug=False,
        return_augs=True,
        context_length=context_length,
        chr_bed_to_fasta_map=chr_bed_to_fasta_map,
    )
    return train_ds, val_ds


_PROGRESS_FILENAME = "_cellink_progress.json"


def _write_training_progress(checkpoint_dir: str, *, completed_epochs: int, num_epochs: int) -> None:
    """Persist the true number of fully-completed epochs alongside a checkpoint.

    This is the single source of truth for resume, not the epoch number
    embedded in the checkpoint directory's own name (``scooby_epoch_{epoch}_
    {step}_...``): that name-based scheme can drift from the real count
    across a long chain of resumes, so every checkpoint carries its own
    verified count directly, immune to directory-name drift, a mid-chain
    kill/resubmit, or any future bug in the naming scheme.
    """
    with open(Path(checkpoint_dir) / _PROGRESS_FILENAME, "w") as f:
        json.dump({"completed_epochs": completed_epochs, "num_epochs": num_epochs}, f)


def _read_training_progress(checkpoint_dir: str) -> int | None:
    """Read back the true completed-epoch count written by ``_write_training_progress``.

    Returns ``None`` (not 0) when absent, so callers can distinguish an
    old-style checkpoint that predates this tracking, which should fall back
    to a best-effort directory-name parse, from a run that has genuinely
    completed zero epochs.
    """
    path = Path(checkpoint_dir) / _PROGRESS_FILENAME
    if not path.is_file():
        return None
    with open(path) as f:
        return int(json.load(f)["completed_epochs"])


def _run_scooby_training_loop(
    *,
    scooby_model,
    training_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    wd: float,
    warmup_steps: int,
    clip_global_norm: float,
    eval_every_n: int,
    total_weight: float,
    output_dir: str,
    run_name: str,
    fix_rev_comp_fn,
    mode: str,
    save_every_n_steps: int = 1000,
    log_with: str | None = None,
    mixed_precision: str | None = "bf16",
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: str | None = None,
):
    """Shared Accelerate-based training loop for both RNA-only and multiome scooby fine-tuning.

    ``resume_from_checkpoint``: path to a directory previously written by
    ``accelerator.save_state`` (e.g. one of this same function's own
    ``{output_dir}/scooby_epoch_{epoch}_{i}_{run_name}`` checkpoints). Loaded
    via ``accelerator.load_state`` after ``accelerator.prepare``, the correct
    order since model/optimizer/scheduler must already be wrapped before
    their states can be restored into the wrapped objects. Model weights,
    optimizer state, scheduler state, and dataloader sampler position are all
    restored, a genuine continuation rather than a fresh run seeded with old
    weights.
    """
    import torch
    import torch.nn as nn
    import tqdm
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    from scooby.utils.utils import add_weight_decay, evaluate, poisson_multinomial_torch

    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(
        log_with=log_with, kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False,
        mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device

    # In real optimizer-update terms (matching warmup_steps' own units, and
    # the reference's "warmed up over the first 1,000 steps" of real steps,
    # not raw dataloader batches), divided by gradient_accumulation_steps so
    # the LR schedule decays over the intended number of real updates rather
    # than scaling down with the micro-batch count.
    num_steps = (45_000 * num_epochs) // (training_loader.batch_size * gradient_accumulation_steps)
    parameters = add_weight_decay(scooby_model, lr=lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(parameters)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-7, total_iters=warmup_steps)
    train_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(num_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [warmup_steps])

    scooby_model = nn.SyncBatchNorm.convert_sync_batchnorm(scooby_model)
    scooby_model, optimizer, scheduler, training_loader, val_loader = accelerator.prepare(
        scooby_model, optimizer, scheduler, training_loader, val_loader
    )

    starting_epoch = 0
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        logger.info("Scooby training: resumed from %s", resume_from_checkpoint)
        progress = _read_training_progress(resume_from_checkpoint)
        if progress is not None:
            starting_epoch = progress
            logger.info(
                "Scooby training: resuming at true epoch %d/%d (read from this checkpoint's own "
                "%s, verified, not inferred)", starting_epoch, num_epochs, _PROGRESS_FILENAME,
            )
        else:
            m = re.search(r"scooby_epoch_(\d+)_\d+_", Path(resume_from_checkpoint).name)
            if m:
                starting_epoch = int(m.group(1))
                logger.warning(
                    "Scooby training: %s not found in %r (an old-style checkpoint predating real "
                    "progress-tracking), falling back to parsing epoch %d from the directory name. "
                    "This inferred value is only as trustworthy as the name itself, which can drift "
                    "from the true epoch count across chained resumes. Every checkpoint saved from "
                    "this run onward carries its own verified count and will not need this fallback.",
                    _PROGRESS_FILENAME, Path(resume_from_checkpoint).name, starting_epoch,
                )
            else:
                logger.warning(
                    "Scooby training: could not parse an epoch number from checkpoint name %r either, "
                    "restarting the epoch loop at 0 (real optimizer/scheduler state is still correctly "
                    "restored regardless; only the redundant-raw-pass bound above is lost for this resume)",
                    Path(resume_from_checkpoint).name,
                )
    if log_with:
        accelerator.init_trackers("scooby", init_kwargs={"wandb": {"name": run_name}})
    loss_fn = poisson_multinomial_torch

    completed_epochs = starting_epoch
    while completed_epochs < num_epochs:
        epoch = completed_epochs
        for i, (inputs, rc_augs, targets, cell_emb_idx) in enumerate(tqdm.tqdm(training_loader)):
            with accelerator.accumulate(scooby_model):
                inputs = inputs.permute(0, 2, 1).to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                for rc_aug_idx in rc_augs.nonzero():
                    rc_aug_idx = rc_aug_idx[0]
                    flipped = torch.flip(targets[rc_aug_idx].unsqueeze(0), (1, -3))
                    targets[rc_aug_idx] = fix_rev_comp_fn(flipped)[0]

                with accelerator.autocast():
                    outputs = scooby_model(inputs, cell_emb_idx)
                    loss = loss_fn(outputs, targets, total_weight=total_weight)
                    if log_with:
                        accelerator.log({"loss": loss})
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(scooby_model.parameters(), clip_global_norm)

                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    if log_with:
                        accelerator.log({"learning_rate": scheduler.get_last_lr()[0]})
                optimizer.zero_grad()
            if i % eval_every_n == 0:
                evaluate(accelerator, scooby_model, val_loader, mode=mode, stop_idx=0)
                scooby_model.train()
            if (i % save_every_n_steps == 0 and epoch != 0) or (i % (2 * save_every_n_steps) == 0 and epoch == 0 and i != 0):
                ckpt_dir = f"{output_dir}/scooby_epoch_{epoch}_{i}_{run_name}"
                accelerator.save_state(output_dir=ckpt_dir)

                _write_training_progress(ckpt_dir, completed_epochs=completed_epochs, num_epochs=num_epochs)
        completed_epochs += 1
        logger.info("Scooby training: completed epoch %d/%d (true, persisted count)", completed_epochs, num_epochs)

    final_dir = f"{output_dir}/scooby_final_{run_name}"
    accelerator.save_state(output_dir=final_dir)
    _write_training_progress(final_dir, completed_epochs=completed_epochs, num_epochs=num_epochs)
    if log_with:
        accelerator.end_training()
    logger.info("Scooby training complete. Final state: %s (true epoch count: %d/%d)", final_dir, completed_epochs, num_epochs)
    return final_dir


def train_scooby(
    rna_plus_path: str,
    rna_minus_path: str,
    embedding_path: str,
    *,
    output_dir: str,
    run_name: str,
    sequences_path: str,
    genome_path: str,
    neighbors_path: str | None = None,
    pretrained_model: str = "johahi/borzoi-replicate-0",
    cell_emb_dim: int = 16,
    num_tracks: int = 2,
    context_length: int = 524_288,
    batch_size: int = 1,
    lr: float = 1e-4,
    wd: float = 1e-6,
    clip_global_norm: float = 1.0,
    warmup_steps: int = 1000,
    num_epochs: int = 2,
    eval_every_n: int = 2000,
    total_weight: float = 0.2,
    test_fold: int = 7,
    val_fold: int = 4,
    shift_augs: tuple[int, int] = (-3, 3),
    rc_aug: bool = True,
    chr_bed_to_fasta_map: dict[str, str] | None = None,
    cell_sample_size: int = 64,
    val_cell_sample_size: int = 32,
    clip_soft: float = 5,
    num_workers: int = 8,
    log_with: str | None = None,
    runner: ScoobyRunner | None = None,
    use_transform_borzoi_emb: bool = False,
    mixed_precision: str | None = "bf16",
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: str | None = None,
) -> str:
    """RNA-only Scooby fine-tuning: cell embedding + DNA sequence -> per-cell
    RNA coverage. No genotype required (see module docstring).

    ``use_transform_borzoi_emb`` defaults to False: this flag is only correct
    as True when resuming a training run whose backbone has already
    converged without it (the reference's own `scooby_reproducibility`
    resume script uses True for exactly that reason). 

    Parameters
    ----------
    rna_plus_path, rna_minus_path : str
        Paths to strand-split RNA h5ads with ``obsm["fragment_single"]``.
    embedding_path : str
        Path to a per-cell embedding parquet (see ``build_scooby_embedding``).
    neighbors_path : str, optional
        Path to a (possibly empty/no-op) neighbors ``.npz``. Required by
        scooby's dataset API even when unused.
    pretrained_model : str
        HuggingFace Hub model id or local path passed to
        ``Scooby.from_pretrained``.
    cell_emb_dim, num_tracks : int
        Must match ``embedding_path``'s dimensionality and the modality
        (RNA-only = 2 tracks: plus/minus strand).

    Returns
    -------
    str
        Path to the final saved Accelerate state directory.
    """
    import h5py
    import scipy.sparse

    from scooby.utils.utils import fix_rev_comp_rna, read_backed

    from torch.utils.data import DataLoader

    runner = runner or get_scooby_runner()
    Scooby = runner.get_scooby_class()
    get_lora = runner.get_lora_fn()

    adatas = {
        "rna_plus": read_backed(h5py.File(rna_plus_path), "fragment_single"),
        "rna_minus": read_backed(h5py.File(rna_minus_path), "fragment_single"),
    }
    embedding = pd.read_parquet(embedding_path)
    neighbors = scipy.sparse.load_npz(neighbors_path) if neighbors_path else None

    scooby_model = Scooby.from_pretrained(
        pretrained_model,
        cell_emb_dim=cell_emb_dim,
        embedding_dim=1920,
        n_tracks=num_tracks,
        return_center_bins_only=True,
        disable_cache=True,
        use_transform_borzoi_emb=use_transform_borzoi_emb,
    )
    scooby_model = get_lora(scooby_model, train=True, lora_config=runner.lora_config)

    train_ds, val_ds = _make_genome_interval_datasets(
        sequences_path, genome_path, test_fold=test_fold, val_fold=val_fold,
        context_length=context_length, shift_augs=shift_augs, rc_aug=rc_aug,
        chr_bed_to_fasta_map=chr_bed_to_fasta_map,
    )
    otf_dataset = build_scooby_dataset(
        adatas["rna_plus"], adatas["rna_minus"], embedding, train_ds,
        neighbors=neighbors, cell_sample_size=cell_sample_size, clip_soft=clip_soft,
    )
    val_dataset = build_scooby_dataset(
        adatas["rna_plus"], adatas["rna_minus"], embedding, val_ds,
        neighbors=neighbors, cell_sample_size=val_cell_sample_size, clip_soft=clip_soft,
    )
    training_loader = DataLoader(otf_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return _run_scooby_training_loop(
        scooby_model=scooby_model, training_loader=training_loader, val_loader=val_loader,
        num_epochs=num_epochs, lr=lr, wd=wd, warmup_steps=warmup_steps, clip_global_norm=clip_global_norm,
        eval_every_n=eval_every_n, total_weight=total_weight, output_dir=output_dir, run_name=run_name,
        fix_rev_comp_fn=fix_rev_comp_rna, mode="rna", log_with=log_with,
        mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def train_scooby_multiome(
    rna_plus_path: str,
    rna_minus_path: str,
    atac_path: str,
    embedding_path: str,
    *,
    output_dir: str,
    run_name: str,
    sequences_path: str,
    genome_path: str,
    neighbors_path: str | None = None,
    pretrained_model: str = "johahi/borzoi-replicate-0",
    cell_emb_dim: int = 16,
    num_tracks: int = 3,
    normalize_atac: bool = True,
    context_length: int = 524_288,
    batch_size: int = 1,
    lr: float = 1e-4,
    wd: float = 1e-6,
    clip_global_norm: float = 1.0,
    warmup_steps: int = 1000,
    num_epochs: int = 2,
    eval_every_n: int = 2000,
    total_weight: float = 0.2,
    test_fold: int = 7,
    val_fold: int = 4,
    shift_augs: tuple[int, int] = (-3, 3),
    rc_aug: bool = True,
    chr_bed_to_fasta_map: dict[str, str] | None = None,
    cell_sample_size: int = 64,
    val_cell_sample_size: int = 32,
    clip_soft: float = 5,
    num_workers: int = 8,
    log_with: str | None = None,
    runner: ScoobyRunner | None = None,
    use_transform_borzoi_emb: bool = False,
    mixed_precision: str | None = "bf16",
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: str | None = None,
) -> str:
    """RNA+ATAC multiome Scooby fine-tuning: cell embedding + DNA sequence ->
    per-cell RNA coverage AND ATAC accessibility, jointly.

    Parameters
    ----------
    atac_path : str
        Path to an ATAC h5ad with ``obsm["insertion"]``.
    normalize_atac : bool
        Scale ATAC coverage (x0.05) for training stability,
        reference default for multiome training.
    Other parameters : see :func:`train_scooby` (``use_transform_borzoi_emb``
        defaults to False here for the same cold-start reason).

    Returns
    -------
    str
        Path to the final saved Accelerate state directory.
    """
    import h5py
    import scipy.sparse

    from scooby.utils.utils import fix_rev_comp_multiome, read_backed

    from torch.utils.data import DataLoader

    runner = runner or get_scooby_runner()
    Scooby = runner.get_scooby_class()
    get_lora = runner.get_lora_fn()

    adatas = {
        "rna_plus": read_backed(h5py.File(rna_plus_path), "fragment_single"),
        "rna_minus": read_backed(h5py.File(rna_minus_path), "fragment_single"),
        "atac": read_backed(h5py.File(atac_path), "insertion"),
    }
    embedding = pd.read_parquet(embedding_path)
    neighbors = scipy.sparse.load_npz(neighbors_path) if neighbors_path else None

    scooby_model = Scooby.from_pretrained(
        pretrained_model,
        cell_emb_dim=cell_emb_dim,
        embedding_dim=1920,
        n_tracks=num_tracks,
        return_center_bins_only=True,
        disable_cache=True,
        use_transform_borzoi_emb=use_transform_borzoi_emb,
    )
    scooby_model = get_lora(scooby_model, train=True, lora_config=runner.lora_config)

    train_ds, val_ds = _make_genome_interval_datasets(
        sequences_path, genome_path, test_fold=test_fold, val_fold=val_fold,
        context_length=context_length, shift_augs=shift_augs, rc_aug=rc_aug,
        chr_bed_to_fasta_map=chr_bed_to_fasta_map,
    )
    otf_dataset = build_scooby_multiome_dataset(
        adatas, embedding, train_ds, neighbors=neighbors,
        cell_sample_size=cell_sample_size, clip_soft=clip_soft, normalize_atac=normalize_atac,
    )
    val_dataset = build_scooby_multiome_dataset(
        adatas, embedding, val_ds, neighbors=neighbors,
        cell_sample_size=val_cell_sample_size, clip_soft=clip_soft, normalize_atac=normalize_atac,
    )
    training_loader = DataLoader(otf_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return _run_scooby_training_loop(
        scooby_model=scooby_model, training_loader=training_loader, val_loader=val_loader,
        num_epochs=num_epochs, lr=lr, wd=wd, warmup_steps=warmup_steps, clip_global_norm=clip_global_norm,
        eval_every_n=eval_every_n, total_weight=total_weight, output_dir=output_dir, run_name=run_name,
        fix_rev_comp_fn=fix_rev_comp_multiome, mode="multiome", log_with=log_with,
        mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def load_scooby_checkpoint(
    model_path_or_name: str,
    *,
    cell_emb_dim: int | None = None,
    n_tracks: int | None = None,
    embedding_dim: int = 1920,
    use_transform_borzoi_emb: bool | None = None,
    runner: ScoobyRunner | None = None,
):
    """Load a trained Scooby checkpoint via ``Scooby.from_pretrained``.

    If ``model_path_or_name`` matches a known released checkpoint (see
    ``KNOWN_SCOOBY_CHECKPOINTS``), ``cell_emb_dim``/``n_tracks``/
    ``use_transform_borzoi_emb`` default from that registry; otherwise they
    must be supplied explicitly.

    Returns
    -------
    scooby.modeling.Scooby
        On the runner's resolved device, in eval mode.
    """
    runner = runner or get_scooby_runner()
    Scooby = runner.get_scooby_class()

    known = KNOWN_SCOOBY_CHECKPOINTS.get(model_path_or_name)
    if cell_emb_dim is None:
        if known is None:
            raise ValueError(
                f"'{model_path_or_name}' is not a known released checkpoint "
                f"({sorted(KNOWN_SCOOBY_CHECKPOINTS)}), pass cell_emb_dim explicitly."
            )
        cell_emb_dim = known["cell_emb_dim"]
    if n_tracks is None:
        if known is None:
            raise ValueError(
                f"'{model_path_or_name}' is not a known released checkpoint "
                f"({sorted(KNOWN_SCOOBY_CHECKPOINTS)}), pass n_tracks explicitly."
            )
        n_tracks = known["n_tracks"]
    if use_transform_borzoi_emb is None:
        if known is None:
            raise ValueError(
                f"'{model_path_or_name}' is not a known released checkpoint "
                f"({sorted(KNOWN_SCOOBY_CHECKPOINTS)}), pass use_transform_borzoi_emb explicitly."
            )
        use_transform_borzoi_emb = known["use_transform_borzoi_emb"]

    model = Scooby.from_pretrained(
        model_path_or_name,
        cell_emb_dim=cell_emb_dim,
        embedding_dim=embedding_dim,
        n_tracks=n_tracks,
        return_center_bins_only=True,
        use_transform_borzoi_emb=use_transform_borzoi_emb,
    )
    model = model.to(runner.resolve_device()).eval()
    return model


def convert_scooby_lora_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    *,
    pretrained_model: str = "johahi/borzoi-replicate-0",
    cell_emb_dim: int,
    n_tracks: int,
    embedding_dim: int = 1920,
    use_transform_borzoi_emb: bool = True,
    overwrite: bool = False,
    runner: ScoobyRunner | None = None,
) -> str:
    """Convert an ``accelerate.save_state()``-saved LoRA fine-tune checkpoint
    (as ``train_scooby``/``train_scooby_multiome`` write via
    ``accelerator.save_state(...)`` into a clean, directly
    ``from_pretrained``-loadable checkpoint directory (with a real
    ``config.json``).

    Parameters
    ----------
    checkpoint_dir : str
        An ``accelerate``-saved checkpoint directory (contains
        ``model.safetensors``; optimizer/scheduler/RNG state files are
        ignored).
    output_dir : str
        Where to write the clean, from_pretrained-loadable checkpoint.
    pretrained_model : str
        Must match what training actually started from (the ``pretrained_model``
        value in the training config).
    cell_emb_dim, n_tracks, use_transform_borzoi_emb
        Must match the training config.

    Returns
    -------
    str
        ``output_dir``, for chaining into ``load_scooby_checkpoint``.
    """
    out_dir = Path(output_dir)
    if out_dir.exists() and not overwrite:
        logger.info("%s exists; pass overwrite=True to redo. Skipping.", out_dir)
        return str(out_dir)

    try:
        import safetensors.torch
    except ImportError as e:
        raise ImportError("safetensors is required for this function. Install with: pip install safetensors") from e

    runner = runner or get_scooby_runner()
    Scooby = runner.get_scooby_class()
    get_lora = runner.get_lora_fn()

    ckpt_path = Path(checkpoint_dir) / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")

    logger.info("building base model from %s (cell_emb_dim=%d, n_tracks=%d)", pretrained_model, cell_emb_dim, n_tracks)
    model = Scooby.from_pretrained(
        pretrained_model,
        cell_emb_dim=cell_emb_dim,
        embedding_dim=embedding_dim,
        n_tracks=n_tracks,
        return_center_bins_only=True,
        disable_cache=True,
        use_transform_borzoi_emb=use_transform_borzoi_emb,
    )
    model = get_lora(model, train=False, lora_config=runner.lora_config)

    logger.info("loading fine-tuned weights from %s", ckpt_path)
    safetensors.torch.load_model(model, str(ckpt_path))

    logger.info("merging LoRA deltas into the base weights (merge_and_unload)")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    logger.info("DONE -> %s (now directly loadable via Scooby.from_pretrained / ScoobyWrapper)", out_dir)
    return str(out_dir)


def predict_scooby_profile(
    model,
    sequence: np.ndarray,
    cell_embeddings: np.ndarray,
    *,
    aggregate: Literal["pseudobulk", "none"] = "pseudobulk",
    undo_squashed_scale: bool = True,
    track_indices=None,
):
    """Predict Scooby coverage for a (one-hot-encoded) sequence, given a set
    of per-cell embeddings, the ref/alt-agnostic single-forward-pass
    primitive that variant-effect scoring diffs against two sequences.

    Parameters
    ----------
    model : scooby.modeling.Scooby
        A loaded checkpoint (see ``load_scooby_checkpoint``).
    sequence : np.ndarray
        One-hot-encoded sequence, shape ``(seq_len, 4)`` or ``(bs, seq_len, 4)``.
    cell_embeddings : np.ndarray
        Shape ``(n_cells, cell_emb_dim)``.
    aggregate : {"pseudobulk", "none"}
        ``"pseudobulk"`` sums per-cell linear-scale predictions across
        cells (matching ``scooby.utils.utils.get_pseudobulk_profile_pred``);
        ``"none"`` returns the full per-cell profile.
    undo_squashed_scale : bool
        Invert Borzoi's soft-clip+power-law training-target transform back
        to linear scale (recommended for any downstream diffing/comparison).

    Returns
    -------
    numpy.ndarray
        Predicted profile, shape ``(seq_len_bins, n_tracks)`` if
        ``aggregate="pseudobulk"`` else ``(n_cells, seq_len_bins, n_tracks)``.
    """
    import torch

    device = next(model.parameters()).device
    seq_t = torch.as_tensor(sequence, dtype=torch.float32, device=device)
    if seq_t.ndim == 2:
        seq_t = seq_t.unsqueeze(0)

    seq_t = seq_t.permute(0, 2, 1)
    cell_emb_t = torch.as_tensor(cell_embeddings, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        conv_weights, conv_biases = model.forward_cell_embs_only(cell_emb_t)
        out = model.forward_sequence_w_convs(seq_t, conv_weights, conv_biases)
        if undo_squashed_scale:
            from scooby.utils.utils import undo_squashed_scale as _undo

            out = _undo(out)

    out_np = out.detach().cpu().numpy()

    n_cells = cell_embeddings.shape[0] if np.asarray(cell_embeddings).ndim == 2 else 1
    n_tracks = out_np.shape[-1] // n_cells
    out_np = out_np[0].reshape(out_np.shape[1], n_cells, n_tracks)  # (seq_len_bins, n_cells, n_tracks)
    if track_indices is not None:
        out_np = out_np[..., track_indices]
    if aggregate == "pseudobulk":
        out_np = out_np.sum(axis=1)  # (seq_len_bins, n_tracks)
    else:
        out_np = out_np.transpose(1, 0, 2)  # (n_cells, seq_len_bins, n_tracks)
    return out_np


def score_variant_effects_scooby(
    snp,
    chromosome_sequence: str,
    model_path_or_name: str,
    cell_embeddings: np.ndarray,
    *,
    bin_indices=None,
    pseudocount: float = 1.0,
    aggregate: Literal["pseudobulk", "none"] = "pseudobulk",
    runner: ScoobyRunner | None = None,
    **checkpoint_kwargs,
):
    """Score a single variant's effect on predicted Scooby coverage via
    ref/alt sequence diffing.


    Parameters
    ----------
    snp : embpy.tl.genomics.SNPContext
        The variant to score (chromosome, position, ref/alt alleles).
    chromosome_sequence : str
        Full chromosome sequence the variant sits in (or a large enough
        window around it).
    model_path_or_name : str
        Passed to ``load_scooby_checkpoint``.
    cell_embeddings : np.ndarray
        Per-cell embeddings to pseudobulk over (or per-cell if
        ``aggregate="none"``).
    bin_indices : array-like, optional
        Restrict the scoring statistic to these model-output bins (e.g. a
        gene's exon bins from ``embpy.tl.genomics.genomic_to_bin_indices``).
        ``None`` scores over all bins.
    pseudocount : float
        Added before the log2-fold-change computation.

    Returns
    -------
    embpy.tl.genomics.snp_utils.VariantEffectResult
    """
    try:
        from embpy.models.scooby_models import ScoobyWrapper
        from embpy.tl.genomics import SNPEmbedder
    except ImportError as e:
        raise ImportError(
            "embpy is required for `score_variant_effects_scooby`. Variant-effect scoring is not part of "
            "scooby itself, it's implemented in the separate embpy package. Install with:\n\n"
            "    pip install cellink[embpy]"
        ) from e

    runner = runner or get_scooby_runner()
    wrapper = ScoobyWrapper(model_path_or_name, device=runner.resolve_device(), **checkpoint_kwargs)
    wrapper.load(runner.resolve_device())

    embedder = SNPEmbedder(wrapper)
    return embedder.predict_variant_effect(
        snp,
        chromosome_sequence,
        bin_indices=bin_indices,
        pseudocount=pseudocount,
        aggregate=aggregate,
        cell_embeddings=cell_embeddings,
    )


def load_scooby_wrapper_and_embedder(
    model_path_or_name: str,
    *,
    pooling_strategy: str | None = None,
    runner: ScoobyRunner | None = None,
    **checkpoint_kwargs,
):
    """Load a Scooby checkpoint once into a ``(wrapper, embedder)`` pair for
    repeated batched scoring via ``score_variant_effects_scooby_batched``.

    Call this once per checkpoint, then pass the same ``wrapper``/``embedder``
    into multiple ``score_variant_effects_scooby_batched`` calls (e.g. one
    per credible set, with each call's own ``cell_embeddings``) instead of
    reloading the checkpoint each time.

    Parameters
    ----------
    model_path_or_name : str
        Passed to ``load_scooby_checkpoint``.
    pooling_strategy : str, optional
        Forwarded to ``SNPEmbedder`` if given; ``None`` uses embpy's own
        default.

    Returns
    -------
    (embpy.models.scooby_models.ScoobyWrapper, embpy.tl.genomics.SNPEmbedder)
    """
    try:
        from embpy.models.scooby_models import ScoobyWrapper
        from embpy.tl.genomics import SNPEmbedder
    except ImportError as e:
        raise ImportError(
            "embpy is required for Scooby variant-effect scoring. Install with:\n\n"
            "    pip install cellink[embpy]"
        ) from e

    runner = runner or get_scooby_runner()
    wrapper = ScoobyWrapper(model_path_or_name, device=runner.resolve_device(), **checkpoint_kwargs)
    wrapper.load(runner.resolve_device())
    embedder = SNPEmbedder(wrapper) if pooling_strategy is None else SNPEmbedder(wrapper, pooling_strategy=pooling_strategy)
    return wrapper, embedder


def score_variant_effects_scooby_batched(
    variants: list[tuple[Any, str]],
    model_path_or_name: str,
    cell_embeddings: np.ndarray,
    *,
    bin_indices: list | None = None,
    pseudocount: float = 1.0,
    aggregate: Literal["pseudobulk", "none"] = "pseudobulk",
    pooling_strategy: str | None = None,
    runner: ScoobyRunner | None = None,
    wrapper: Any | None = None,
    embedder: Any | None = None,
    **checkpoint_kwargs,
):
    """Score many variants against ONE loaded Scooby checkpoint.

    ``score_variant_effects_scooby`` reconstructs and reloads the checkpoint
    from disk on every call, which is the right shape for a single
    exploratory lookup but does not scale to scoring a batch of candidates
    (e.g. every member of a fine-mapped credible set, or a control-SNP set)
    in one job. This function scores a whole batch against one load.

    By default it loads the checkpoint itself (like
    ``score_variant_effects_scooby``, just once for the whole ``variants``
    list instead of once per variant). For a run that needs to score many
    separate batches with the same checkpoint but different
    ``cell_embeddings`` per batch (e.g. one call per credible set), load
    once via ``load_scooby_wrapper_and_embedder`` and pass the result in as
    ``wrapper``/``embedder`` to skip the reload on every call.

    Parameters
    ----------
    variants : list of (snp, chromosome_sequence)
        Each element is one variant to score: an
        ``embpy.tl.genomics.SNPContext`` and the reference sequence window
        it should be scored against (same meaning as
        ``score_variant_effects_scooby``'s ``snp``/``chromosome_sequence``).
    model_path_or_name : str
        Passed to ``load_scooby_checkpoint`` if ``wrapper``/``embedder``
        aren't given; ignored otherwise.
    cell_embeddings : np.ndarray
        Per-cell embeddings to pseudobulk over (or per-cell if
        ``aggregate="none"``), shared across every variant in this batch.
    bin_indices : list, optional
        Per-variant bin restriction, same length and order as ``variants``
        (e.g. each variant's own gene's exon bins). ``None`` scores every
        variant over all bins; an individual entry may also be ``None``.
    pseudocount : float
        Added before the log2-fold-change computation.
    pooling_strategy : str, optional
        Forwarded to ``SNPEmbedder`` if given and ``wrapper``/``embedder``
        aren't; ``None`` uses embpy's own default.
    wrapper, embedder : optional
        A ``(ScoobyWrapper, SNPEmbedder)`` pair already loaded via
        ``load_scooby_wrapper_and_embedder``. If given, both must be given,
        and ``model_path_or_name``/``pooling_strategy``/``runner``/
        ``checkpoint_kwargs`` are ignored (the checkpoint is not reloaded).

    Returns
    -------
    list of embpy.tl.genomics.snp_utils.VariantEffectResult
        One result per element of ``variants``, in the same order.
    """
    if bin_indices is not None and len(bin_indices) != len(variants):
        raise ValueError(f"bin_indices has {len(bin_indices)} entries, expected one per variant ({len(variants)}).")
    bin_indices_per_variant = bin_indices if bin_indices is not None else [None] * len(variants)

    if (wrapper is None) != (embedder is None):
        raise ValueError("wrapper and embedder must be given together, or not at all.")
    if wrapper is None:
        _wrapper, embedder = load_scooby_wrapper_and_embedder(
            model_path_or_name, pooling_strategy=pooling_strategy, runner=runner, **checkpoint_kwargs
        )

    return [
        embedder.predict_variant_effect(
            snp,
            chromosome_sequence,
            bin_indices=variant_bin_indices,
            pseudocount=pseudocount,
            aggregate=aggregate,
            cell_embeddings=cell_embeddings,
        )
        for (snp, chromosome_sequence), variant_bin_indices in zip(variants, bin_indices_per_variant)
    ]


def resolve_snp_and_exon_bins(
    *,
    chrom: str,
    pos: int,
    a0: str,
    a1: str,
    window: str,
    snp_offset: int,
    window_start: int,
    exon_intervals,
    bin_size: int,
    profile_offset_bp: int,
    num_bins: int,
    context_window: int,
    strand: str = "+",
    variant_id: str | None = None,
):
    """Orient a variant's alleles against the real reference sequence and compute its
    exon-overlapping, profile-clipped bin indices, or report why it must be skipped.

    This applies three correctness fixes:

    1. **ref/alt allele-order mismatch.** ``a0``/``a1`` (e.g. from a pgen/PLINK file) are
       not guaranteed to be in forward-strand reference/alternate order. Silently trusting
       the stored order can score a no-op "effect" (ref vs ref). This checks the real
       reference base in ``window`` at the variant's position and swaps if it matches
       ``a1`` instead of ``a0``.
    2. **Unclipped bin indices.** The embpy ``genomic_to_bin_indices`` helper does not
       clip its upper bound to the model's actual (cropped) profile length on its own;
       passing ``num_bins`` (the model's real bin count) here avoids a downstream
       out-of-bounds crash for variants whose exons sit far from the scored window.
    3. **Silent no-op scoring.** If a variant's gene has no exon bins inside the predicted
       (cropped) profile region at all, there is nothing to score; this reports that
       explicitly as a skip rather than scoring an empty statistic.

    Parameters
    ----------
    chrom, pos : str, int
        The variant's chromosome and 1-based genomic position.
    a0, a1 : str
        The variant's two alleles, in whatever order the caller's genotype source
        stores them (order not assumed to be reference-first).
    window : str
        The reference sequence window already fetched around this variant (e.g. via
        an embpy ``SequenceProvider``).
    snp_offset : int
        The variant's 1-based offset of ``pos`` within ``window`` (the same convention
        returned by ``SequenceProvider.get_window``).
    window_start : int
        The genomic start coordinate of ``window`` (``pos - snp_offset``).
    exon_intervals
        The scored gene's exon intervals, as expected by embpy's
        ``genomic_to_bin_indices``.
    bin_size, profile_offset_bp, num_bins, context_window : int
        Passed through to embpy (``wrapper.BIN_SIZE``, ``wrapper.profile_offset_bp``,
        ``wrapper.model.crop.target_length``, ``wrapper.SEQUENCE_LENGTH`` at the
        original call sites).
    variant_id : str, optional
        Used only in the skip message when returning ``None``.

    Returns
    -------
    tuple[SNPContext, numpy.ndarray] | None
        The correctly-oriented ``SNPContext`` and clipped exon bin indices, or ``None``
        if this variant cannot be scored (the reason is logged).
    """
    try:
        from embpy.tl.genomics import SNPContext, genomic_to_bin_indices
    except ImportError as e:
        raise ImportError(
            "embpy is required for `resolve_snp_and_exon_bins`. Install with:\n\n    pip install cellink[embpy]"
        ) from e

    label = variant_id or f"{chrom}:{pos}"
    actual_ref = window[snp_offset - 1 : snp_offset].upper()
    a0, a1 = a0.upper(), a1.upper()
    if actual_ref == a0:
        ref_allele, alt_allele = a0, a1
    elif actual_ref == a1:
        ref_allele, alt_allele = a1, a0
    else:
        logger.warning(f"{label}: neither a0={a0} nor a1={a1} matches the true reference base ({actual_ref}), skipping.")
        return None

    snp = SNPContext(
        chrom=chrom,
        position=snp_offset,
        ref_allele=ref_allele,
        alt_alleles=[alt_allele],
        context_window=context_window,
        strand=strand,
        variant_id=variant_id,
    )

    bin_indices = genomic_to_bin_indices(
        exon_intervals,
        window_start=window_start,
        bin_size=bin_size,
        profile_offset_bp=profile_offset_bp,
        num_bins=num_bins,
    )
    if len(bin_indices) == 0:
        logger.warning(f"{label}: no exon bins fall inside the predicted (cropped) profile region, skipping")
        return None

    return snp, bin_indices
