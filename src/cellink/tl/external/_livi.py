from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



class LIVIRunner:
    """Manages access to the LIVI package.

    Parameters
    ----------
    livi_root : str or Path
        Path to the LIVI repository root (must contain ``src/`` and
        ``configs/`` directories).
    execution_mode : {"python_api", "subprocess"}
        ``"python_api"`` (default) imports LIVI directly into the current
        process — no Hydra overhead, returns Python objects.
        ``"subprocess"`` runs LIVI's ``src/train.py`` as a child process via
        the Hydra CLI, which is better suited for isolated HPC job submission.
    python_executable : str
        Python binary used in ``"subprocess"`` mode.
    device : str
        Compute device: ``"auto"`` (detect GPU), ``"cuda"``, or ``"cpu"``.
    """

    def __init__(
        self,
        livi_root: Union[str, Path],
        execution_mode: Literal["python_api", "subprocess"] = "python_api",
        python_executable: str = "python",
        device: str = "auto",
    ):
        self.livi_root = Path(livi_root).resolve()
        if not self.livi_root.exists():
            raise ValueError(f"LIVI root not found: {self.livi_root}")
        if execution_mode not in ("python_api", "subprocess"):
            raise ValueError("execution_mode must be 'python_api' or 'subprocess'")
        self.execution_mode = execution_mode
        self.python_executable = python_executable
        self.device = device

    def _ensure_on_path(self) -> None:
        """Prepend livi_root to sys.path if not already present."""
        root_str = str(self.livi_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def get_livi_class(self):
        """Import and return the :class:`LIVI` model class."""
        self._ensure_on_path()
        from src.models.livi import LIVI  
        return LIVI

    def get_datamodule_classes(self):
        """Import and return ``(LIVIDataModule, LIVIDataset)``."""
        self._ensure_on_path()
        from src.data_modules.livi_data import LIVIDataModule, LIVIDataset 
        return LIVIDataModule, LIVIDataset

    def get_association_testing_fn(self):
        """Import and return ``run_LIVI_genetic_association_testing``."""
        self._ensure_on_path()
        from src.analysis.livi_testing import run_LIVI_genetic_association_testing  
        return run_LIVI_genetic_association_testing

    def resolve_device(self) -> str:
        """Return the resolved device string (``"cuda"`` or ``"cpu"``)."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


_livi_runner: Optional[LIVIRunner] = None


def configure_livi_runner(
    livi_root: Union[str, Path],
    execution_mode: Literal["python_api", "subprocess"] = "python_api",
    python_executable: str = "python",
    device: str = "auto",
) -> LIVIRunner:
    """Configure the global LIVI runner.

    Must be called once before using any other LIVI functions.

    Parameters
    ----------
    livi_root : str or Path
        Path to the LIVI repository root (contains ``src/`` and ``configs/``).
    execution_mode : {"python_api", "subprocess"}
        How to run LIVI training.  ``"python_api"`` (recommended) imports
        LIVI classes directly; results are Python objects.  ``"subprocess"``
        runs LIVI's Hydra CLI, writing configs to ``configs/`` inside the
        LIVI repository.
    python_executable : str
        Python binary used in subprocess mode (default: ``"python"``).
    device : str
        Compute device: ``"auto"`` (auto-detect GPU), ``"cuda"``,
        or ``"cpu"``.

    Returns
    -------
    LIVIRunner

    Examples
    --------
    >>> import cellink as cl
    >>> cl.tl.external.configure_livi_runner("/lustre/projects/LIVI")
    """
    global _livi_runner
    _livi_runner = LIVIRunner(
        livi_root=livi_root,
        execution_mode=execution_mode,
        python_executable=python_executable,
        device=device,
    )
    return _livi_runner


def get_livi_runner() -> LIVIRunner:
    """Return the global :class:`LIVIRunner`, raising if not configured."""
    if _livi_runner is None:
        raise RuntimeError(
            "LIVI runner not configured. "
            "Call `cellink.tl.external.configure_livi_runner(livi_root=...)` first."
        )
    return _livi_runner


def _resolve_adata(
    adata_or_dd,
    individual_col: Optional[str],
) -> Tuple:
    """Extract (cell_adata, individual_col, donor_gdata_or_None) from input.

    Accepts :class:`~anndata.AnnData` or :class:`~cellink.DonorData`.
    When DonorData is passed, ``dd.C`` is the cell AnnData, ``dd.donor_id``
    is the individual column, and ``dd.G`` is returned as the third element
    (donor-level AnnData carrying genotypes, kinship, genotype PCs, etc.).
    """
    from anndata import AnnData

    try:
        from cellink._core.donordata import DonorData

        if isinstance(adata_or_dd, DonorData):
            adata = adata_or_dd.C
            if individual_col is None:
                individual_col = adata_or_dd.donor_id
            return adata, individual_col, adata_or_dd.G
    except ImportError:
        pass

    if isinstance(adata_or_dd, AnnData):
        if individual_col is None:
            raise ValueError(
                "`individual_col` must be specified when passing an AnnData directly."
            )
        return adata_or_dd, individual_col, None

    raise TypeError(
        f"Expected AnnData or DonorData, got {type(adata_or_dd).__name__}"
    )


def _gdata_to_genotype_df(gdata) -> pd.DataFrame:
    """Convert ``dd.G`` (donors × SNPs AnnData) to a dense DataFrame."""
    from anndata.utils import asarray
    return pd.DataFrame(
        asarray(gdata.X),
        index=gdata.obs_names,
        columns=gdata.var_names,
    )


def _infer_covariates_dims(adata, covariates_keys: List[str]) -> List[int]:
    return [int(adata.obs[k].nunique()) for k in covariates_keys]


def _adata_path_or_save(adata, output_dir: str, filename: str = "_livi_input.h5ad") -> str:
    """Return an on-disk path for *adata*, saving first if it is not backed."""
    if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
        return str(adata.file.filename)
    dest = os.path.join(output_dir, filename)
    logger.info("Saving input adata to %s for subprocess LIVI training", dest)
    adata.write_h5ad(dest)
    return dest


def _df_or_path_to_path(
    obj: Union[str, "pd.DataFrame", None],
    dest_path: str,
    sep: str = "\t",
) -> Optional[str]:
    """If *obj* is a DataFrame write it to *dest_path* and return the path."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    obj.to_csv(dest_path, sep=sep)
    return dest_path


def _train_livi_python_api(
    *,
    adata,
    individual_col: str,
    output_dir: str,
    n_genes: int,
    n_donors: int,
    z_dim: int,
    n_dxc_factors: int,
    n_persistent_factors: int,
    n_cis_snps: int,
    encoder_hidden_dims: List[int],
    learning_rate: float,
    use_size_factor: bool,
    size_factor_key: Optional[str],
    layer_key: Optional[str],
    covariates_keys: Optional[List[str]],
    covariates_dims: Optional[List[int]],
    known_cis_eqtls,
    eqtl_genotypes,
    warmup_epochs_vae: int,
    warmup_epochs_G: int,
    max_epochs: int,
    min_epochs: int,
    batch_size: int,
    seed: int,
    l1_weight: float,
    A_weight: float,
    batch_norm_decoder: bool,
    genetics_seed: Optional[int],
    cell_state_cis: bool,
    num_workers: int,
    strict: bool,
    drop_last: bool,
    shuffle: bool,
    pin_memory: bool,
    log_every_n_steps: int,
    enable_progress_bar: bool,
    gradient_clip_val: Optional[float],
    accumulate_grad_batches: int,
    device: str,
    runner: LIVIRunner,
    enable_checkpointing: bool = True,
    enable_logger: bool = True,
    limit_train_batches: Optional[Union[int, float]] = None,
    deterministic: bool = False,
    callbacks: Optional[List] = None,
) -> Optional[str]:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
    except ImportError as exc:
        raise ImportError(
            "pytorch_lightning is required for LIVI training. "
            "Install it with: pip install pytorch-lightning"
        ) from exc

    import torch

    LIVI = runner.get_livi_class()
    LIVIDataModule, _ = runner.get_datamodule_classes()

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    ckpt_dir = os.path.join(output_dir, "checkpoints")

    if callbacks and any(isinstance(c, ModelCheckpoint) for c in callbacks):
        enable_checkpointing = True  # PL rejects enable_checkpointing=False + a ModelCheckpoint callback

    checkpoint_cb = (
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:04d}_{hp_metric:.5f}",
            save_top_k=1,
            monitor="train/livi_loss",
            mode="min",
            save_last=True,
        )
        if enable_checkpointing
        else None
    )

    datamodule = LIVIDataModule(
        adata=adata,
        y_key=individual_col,
        use_size_factor=use_size_factor,
        size_factor_key=size_factor_key,
        layer_key=layer_key,
        covariates_keys=covariates_keys,
        known_cis_eqtls=known_cis_eqtls,
        eqtl_genotypes=eqtl_genotypes,
        strict=strict,
        backed_mode=False,
        data_split=[1.0],
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory and (device != "cpu"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = LIVI(
        x_dim=n_genes,
        z_dim=z_dim,
        y_dim=n_donors,
        n_DxC_factors=n_dxc_factors,
        n_persistent_factors=n_persistent_factors,
        n_cis_snps=n_cis_snps,
        encoder_hidden_dims=encoder_hidden_dims,
        learning_rate=learning_rate,
        cell_state_cis=cell_state_cis,
        warmup_epochs_vae=warmup_epochs_vae,
        warmup_epochs_G=warmup_epochs_G,
        covariates_dims=covariates_dims,
        l1_weight=l1_weight,
        A_weight=A_weight,
        batch_norm_decoder=batch_norm_decoder,
        genetics_seed=genetics_seed,
        device=device,
    )

    trainer_kwargs: dict = dict(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[*([checkpoint_cb] if checkpoint_cb is not None else []), *(callbacks or [])],
        default_root_dir=output_dir,
        enable_checkpointing=enable_checkpointing,
        logger=enable_logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=deterministic,
    )
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, datamodule=datamodule)

    if checkpoint_cb is None:
        logger.info("LIVI training complete (enable_checkpointing=False, no checkpoint saved).")
        return None

    best_ckpt = checkpoint_cb.best_model_path
    if not best_ckpt:
        best_ckpt = os.path.join(ckpt_dir, "last.ckpt")

    logger.info("LIVI training complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


def _write_yaml(path: str, data: dict) -> None:
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)


def _train_livi_subprocess(
    *,
    adata,
    individual_col: str,
    output_dir: str,
    n_genes: int,
    n_donors: int,
    z_dim: int,
    n_dxc_factors: int,
    n_persistent_factors: int,
    n_cis_snps: int,
    encoder_hidden_dims: List[int],
    learning_rate: float,
    use_size_factor: bool,
    size_factor_key: Optional[str],
    layer_key: Optional[str],
    covariates_keys: Optional[List[str]],
    covariates_dims: Optional[List[int]],
    known_cis_eqtls,
    eqtl_genotypes,
    warmup_epochs_vae: int,
    warmup_epochs_G: int,
    max_epochs: int,
    min_epochs: int,
    batch_size: int,
    seed: int,
    l1_weight: float,
    A_weight: float,
    batch_norm_decoder: bool,
    genetics_seed: Optional[int],
    cell_state_cis: bool,
    num_workers: int,
    strict: bool,
    drop_last: bool,
    shuffle: bool,
    pin_memory: bool,
    device: str,
    runner: LIVIRunner,
) -> str:
    import yaml

    config_root = runner.livi_root / "configs"

    adata_path = _adata_path_or_save(adata, output_dir)
    cis_path = _df_or_path_to_path(
        known_cis_eqtls, os.path.join(output_dir, "_livi_known_cis_eqtls.tsv")
    )
    gt_path = _df_or_path_to_path(
        eqtl_genotypes, os.path.join(output_dir, "_livi_eqtl_genotypes.tsv")
    )

    model_cfg: dict = {
        "_target_": "src.models.livi.LIVI",
        "x_dim": n_genes,
        "y_dim": n_donors,
        "z_dim": z_dim,
        "n_DxC_factors": n_dxc_factors,
        "n_persistent_factors": n_persistent_factors,
        "n_cis_snps": n_cis_snps,
        "encoder_hidden_dims": encoder_hidden_dims,
        "learning_rate": learning_rate,
        "cell_state_cis": cell_state_cis,
        "warmup_epochs_vae": warmup_epochs_vae,
        "warmup_epochs_G": warmup_epochs_G,
        "l1_weight": l1_weight,
        "A_weight": A_weight,
        "batch_norm_decoder": batch_norm_decoder,
        "device": device,
    }
    if covariates_dims is not None:
        model_cfg["covariates_dims"] = covariates_dims
    if genetics_seed is not None:
        model_cfg["genetics_seed"] = genetics_seed

    dm_cfg: dict = {
        "_target_": "src.data_modules.livi_data.LIVIDataModule",
        "adata": adata_path,
        "y_key": individual_col,
        "use_size_factor": use_size_factor,
        "size_factor_key": size_factor_key,
        "layer_key": layer_key,
        "covariates_keys": covariates_keys,
        "known_cis_eqtls": cis_path,
        "eqtl_genotypes": gt_path,
        "data_split": [1.0],
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "seed": seed,
        "strict": strict,
        "num_workers": num_workers,
        "pin_memory": pin_memory and (device != "cpu"),
        "backed_mode": True,
    }

    _write_yaml(str(config_root / "model" / "cellink_livi.yaml"), model_cfg)
    _write_yaml(str(config_root / "datamodule" / "cellink_livi.yaml"), dm_cfg)

    # Trainer overrides
    accelerator = "gpu" if device == "cuda" else "cpu"

    cmd = [
        runner.python_executable,
        str(runner.livi_root / "src" / "train.py"),
        "model=cellink_livi",
        "datamodule=cellink_livi",
        f"trainer.max_epochs={max_epochs}",
        f"trainer.min_epochs={min_epochs}",
        f"trainer.accelerator={accelerator}",
        "trainer.devices=1",
        f"seed={seed}",
        f"hydra.run.dir={output_dir}",
    ]

    logger.info("Running LIVI subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(runner.livi_root), check=True)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        epoch_ckpts = sorted(f for f in os.listdir(ckpt_dir) if "epoch" in f)
        if epoch_ckpts:
            return os.path.join(ckpt_dir, epoch_ckpts[-1])
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            return last

    return output_dir


def train_livi(
    adata_or_dd,
    output_dir: str,
    *,
    individual_col: Optional[str] = None,
    z_dim: int = 15,
    n_dxc_factors: int = 100,
    n_persistent_factors: int = 5,
    n_cis_snps: int = 0,
    encoder_hidden_dims: Optional[List[int]] = None,
    learning_rate: float = 8e-4,
    use_size_factor: bool = True,
    size_factor_key: Optional[str] = None,
    layer_key: Optional[str] = None,
    covariates_keys: Optional[List[str]] = None,
    covariates_dims: Optional[List[int]] = None,
    known_cis_eqtls: Optional[Union[str, "pd.DataFrame"]] = None,
    eqtl_genotypes: Optional[Union[str, "pd.DataFrame"]] = None,
    warmup_epochs_vae: int = 30,
    warmup_epochs_G: int = 0,
    max_epochs: int = 200,
    min_epochs: int = 50,
    batch_size: int = 512,
    seed: int = 42,
    l1_weight: float = 1e-3,
    A_weight: float = 1e-3,
    batch_norm_decoder: bool = False,
    genetics_seed: Optional[int] = None,
    cell_state_cis: bool = True,
    num_workers: int = 0,
    strict: bool = False,
    drop_last: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
    log_every_n_steps: int = 1,
    enable_progress_bar: bool = True,
    enable_checkpointing: bool = True,
    enable_logger: bool = True,
    gradient_clip_val: Optional[float] = None,
    accumulate_grad_batches: int = 1,
    limit_train_batches: Optional[Union[int, float]] = None,
    deterministic: bool = False,
    callbacks: Optional[List] = None,
    run: bool = True,
    runner: Optional[LIVIRunner] = None,
) -> Optional[str]:
    """Train a LIVI model on single-cell RNA-seq data.

    LIVI decomposes cell × gene expression into:

    * **Cell-state factors** (z): shared across donors, learned by the VAE encoder.
    * **DxC factors** (D): donor × cell-state interaction effects, stored in a
      learned donor embedding.
    * **Persistent factors** (V): cell-state-independent donor effects.

    Parameters
    ----------
    adata_or_dd : AnnData or DonorData
        Input data.  When :class:`~cellink.DonorData` is passed, ``dd.C``
        (cell-level AnnData) is used and ``individual_col`` defaults to
        ``dd.donor_id``.
    output_dir : str
        Directory where checkpoints and logs are written.  Created if absent.
    individual_col : str, optional
        Column in ``adata.obs`` carrying donor / individual IDs.  Required
        when passing raw AnnData; auto-inferred from DonorData.
    z_dim : int
        Dimensionality of the cell-state latent space.
    n_dxc_factors : int
        Number of donor × cell-state interaction (DxC) factors.
    n_persistent_factors : int
        Number of persistent (cell-state-independent) donor factors.
    n_cis_snps : int
        Number of known cis-eQTL SNPs included during training.  Set to 0
        when not using genotype data.
    encoder_hidden_dims : list of int, optional
        Hidden-layer widths of the encoder MLP.  Defaults to
        ``[512, 256, 64]``.
    learning_rate : float
        Adam optimizer learning rate.
    use_size_factor : bool
        Use per-cell library-size factors for count normalisation.
    size_factor_key : str, optional
        Key in ``adata.obs`` with pre-computed size factors.  If *None* and
        ``use_size_factor=True``, size factors are the total counts per cell.
    layer_key : str, optional
        Key in ``adata.layers`` containing raw integer counts.  If *None*,
        ``adata.X`` is used.
    covariates_keys : list of str, optional
        Categorical covariate columns in ``adata.obs`` (e.g.
        ``["pool", "sex"]``).  These are corrected for as nuisance effects.
    covariates_dims : list of int, optional
        Number of unique categories per covariate (same order as
        ``covariates_keys``).  If *None*, inferred from data.
    known_cis_eqtls : str or pd.DataFrame, optional
        TSV path or DataFrame (SNPs × genes, binary 0/1) indicating known
        cis-eQTL associations.  ``n_cis_snps`` must equal the number of
        rows.
    eqtl_genotypes : str or pd.DataFrame, optional
        TSV path or DataFrame (individuals × SNPs) with genotype dosages
        (0/1/2).  Required when ``known_cis_eqtls`` is provided.
    warmup_epochs_vae : int
        Epochs to train only the VAE before activating donor-level effects.
    warmup_epochs_G : int
        Additional epochs to train only persistent donor effects (V) after
        VAE warm-up, before enabling DxC effects.
    max_epochs : int
        Maximum training epochs.
    min_epochs : int
        Minimum training epochs.
    batch_size : int
        Cells per training batch.
    seed : int
        Global random seed.
    l1_weight : float
        L1 penalty weight on the DxC decoder weights.
    A_weight : float
        Penalty weight on the assignment matrix *A*.
    batch_norm_decoder : bool
        Apply batch normalisation in the combined decoder.
    genetics_seed : int, optional
        Separate seed for initialising genetic model parameters.
    cell_state_cis : bool
        If *True*, learn a cell-state-specific cis-eQTL correction per SNP.
        If *False*, learn a single cell-level correction.
    num_workers : int
        DataLoader worker processes.
    strict : bool
        Raise on non-integer values in input data (LIVI expects raw counts).
    drop_last : bool
        Drop the last incomplete batch each epoch.
    shuffle : bool
        Shuffle cells before each epoch.
    pin_memory : bool
        Pin DataLoader host memory for faster GPU transfer.
    log_every_n_steps : int
        Logging interval (steps).
    enable_progress_bar : bool
        Show the PyTorch Lightning training progress bar.
    gradient_clip_val : float, optional
        Maximum gradient norm for clipping.
    accumulate_grad_batches : int
        Gradient accumulation steps.
    limit_train_batches : int or float, optional
        Cap on batches per epoch (handy for benchmarking a fixed number of
        batches instead of a full pass); forwarded to
        ``pytorch_lightning.Trainer``.
    enable_checkpointing : bool
        If *False*, skip the ``ModelCheckpoint`` callback entirely (and the
        return value becomes *None*) -- useful for short benchmark/smoke runs
        where you don't want checkpoint I/O.
    enable_logger : bool
        Passed through to ``pytorch_lightning.Trainer`` as ``logger=``; set
        *False* to skip the default ``TensorBoardLogger``.
    deterministic : bool
        Passed through to ``pytorch_lightning.Trainer``; set *True* for
        reproducible (but slower) training.
    callbacks : list, optional
        Extra ``pytorch_lightning.Callback`` instances to attach to the
        ``Trainer`` (e.g. a throughput-logging callback), added alongside the
        ``ModelCheckpoint`` this function always sets up.
    run : bool
        If *False*, log the resolved configuration and return without
        training (dry-run / debug).
    runner : LIVIRunner, optional
        Runner instance.  Uses the global runner when *None*.

    Returns
    -------
    str or None
        Path to the best model checkpoint, or *None* when ``run=False``.

    Examples
    --------
    Basic training from DonorData:

    >>> import cellink as cl
    >>> cl.tl.external.configure_livi_runner("/lustre/projects/LIVI")
    >>> ckpt = cl.tl.external.train_livi(
    ...     dd,
    ...     output_dir="livi_run",
    ...     z_dim=15,
    ...     n_dxc_factors=100,
    ...     covariates_keys=["pool", "sex"],
    ...     max_epochs=300,
    ...     warmup_epochs_vae=60,
    ... )

    With cis-eQTL correction:

    >>> ckpt = cl.tl.external.train_livi(
    ...     dd,
    ...     output_dir="livi_cis_run",
    ...     n_cis_snps=3000,
    ...     known_cis_eqtls="known_cis.tsv",
    ...     eqtl_genotypes="genotypes.tsv",
    ...     layer_key="counts",
    ... )
    """
    if runner is None:
        runner = get_livi_runner()

    if encoder_hidden_dims is None:
        encoder_hidden_dims = [512, 256, 64]

    adata, individual_col, gdata = _resolve_adata(adata_or_dd, individual_col)

    # Auto-extract eqtl_genotypes from dd.G when training with cis-eQTL correction
    if eqtl_genotypes is None and gdata is not None and n_cis_snps > 0:
        eqtl_genotypes = _gdata_to_genotype_df(gdata)
        if n_cis_snps != gdata.n_vars:
            raise ValueError(
                f"n_cis_snps={n_cis_snps} does not match dd.G.n_vars={gdata.n_vars}. "
                "Either set n_cis_snps to match the number of SNPs in dd.G, "
                "or pass eqtl_genotypes explicitly."
            )

    n_genes = adata.shape[1]
    n_donors = int(adata.obs[individual_col].nunique())

    if covariates_dims is None and covariates_keys is not None:
        covariates_dims = _infer_covariates_dims(adata, covariates_keys)

    if not run:
        logger.info(
            "LIVI training config (run=False):\n"
            "  x_dim=%d, y_dim=%d, z_dim=%d\n"
            "  n_dxc_factors=%d, n_persistent_factors=%d, n_cis_snps=%d\n"
            "  encoder_hidden_dims=%s, learning_rate=%g\n"
            "  covariates_keys=%s, covariates_dims=%s",
            n_genes, n_donors, z_dim,
            n_dxc_factors, n_persistent_factors, n_cis_snps,
            encoder_hidden_dims, learning_rate,
            covariates_keys, covariates_dims,
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    device = runner.resolve_device()
    logger.info("Using device: %s", device)

    shared_kwargs = dict(
        adata=adata,
        individual_col=individual_col,
        output_dir=output_dir,
        n_genes=n_genes,
        n_donors=n_donors,
        z_dim=z_dim,
        n_dxc_factors=n_dxc_factors,
        n_persistent_factors=n_persistent_factors,
        n_cis_snps=n_cis_snps,
        encoder_hidden_dims=encoder_hidden_dims,
        learning_rate=learning_rate,
        use_size_factor=use_size_factor,
        size_factor_key=size_factor_key,
        layer_key=layer_key,
        covariates_keys=covariates_keys,
        covariates_dims=covariates_dims,
        known_cis_eqtls=known_cis_eqtls,
        eqtl_genotypes=eqtl_genotypes,
        warmup_epochs_vae=warmup_epochs_vae,
        warmup_epochs_G=warmup_epochs_G,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        batch_size=batch_size,
        seed=seed,
        l1_weight=l1_weight,
        A_weight=A_weight,
        batch_norm_decoder=batch_norm_decoder,
        genetics_seed=genetics_seed,
        cell_state_cis=cell_state_cis,
        num_workers=num_workers,
        strict=strict,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        device=device,
        runner=runner,
    )

    if runner.execution_mode == "python_api":
        return _train_livi_python_api(
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=enable_progress_bar,
            enable_checkpointing=enable_checkpointing,
            enable_logger=enable_logger,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_train_batches,
            deterministic=deterministic,
            callbacks=callbacks,
            **shared_kwargs,
        )
    else:
        if callbacks is not None or limit_train_batches is not None:
            raise ValueError(
                "`callbacks` and `limit_train_batches` are only supported with "
                "execution_mode='python_api' (subprocess mode goes through LIVI's "
                "Hydra CLI, which can't take Python callback objects)."
            )
        return _train_livi_subprocess(**shared_kwargs)


def infer_livi(
    adata_or_dd,
    checkpoint_path: str,
    *,
    individual_col: Optional[str] = None,
    layer_key: Optional[str] = None,
    batch_size: int = 50_000,
    variance_threshold: Optional[float] = None,
    device: Optional[str] = None,
    runner: Optional[LIVIRunner] = None,
) -> Dict[str, pd.DataFrame]:
    """Extract latent factors from a trained LIVI model.

    Runs batch-wise inference to obtain cell-state latent factors for every
    cell and donor-level embeddings for every unique individual.

    Parameters
    ----------
    adata_or_dd : AnnData or DonorData
        Single-cell data to perform inference on.  Uses ``dd.C`` when
        DonorData is provided.  The data should contain the same donors (in
        the same factorisation order) as the training data so that donor
        embedding indices align with the trained model.
    checkpoint_path : str
        Path to a LIVI ``.ckpt`` checkpoint file.
    individual_col : str, optional
        Column in ``adata.obs`` with donor IDs.  Auto-inferred from
        DonorData.
    layer_key : str, optional
        Layer key for raw counts.  If *None*, uses ``adata.X``.
    batch_size : int
        Cells per inference batch (default 50 000).
    variance_threshold : float, optional
        When set, only D factors with cross-donor variance ≥ this threshold
        are retained in the returned ``"D_embedding"`` DataFrame.
    device : str, optional
        Override device for inference (``"cpu"`` or ``"cuda"``).  Defaults
        to the runner's device.
    runner : LIVIRunner, optional
        Runner instance.  Uses the global runner when *None*.

    Returns
    -------
    dict
        Dictionary with the following DataFrames (keys absent when the
        corresponding model component was not trained):

        ``"cell_state_latent"``
            (n_cells × z_dim) — cell-state latent factors per cell.
        ``"cell_state_decoder"``
            (n_genes × z_dim) — gene loadings of the cell-state decoder.
        ``"D_embedding"``
            (n_donors × n_DxC) — donor × cell-state interaction embeddings.
        ``"DxC_decoder"``
            (n_genes × n_DxC) — gene loadings of the DxC decoder.
        ``"V_embedding"``
            (n_donors × n_persistent) — persistent donor factor embeddings.
        ``"V_decoder"``
            (n_genes × n_persistent) — gene loadings of the V decoder.
        ``"assignment_matrix"``
            (z_dim × n_DxC) — assignment matrix *A* mapping cell-state
            factors to DxC factors.

    Examples
    --------
    >>> results = cl.tl.external.infer_livi(dd, "livi_out/checkpoints/best.ckpt")
    >>> cell_state = results["cell_state_latent"]  # cells × z_dim
    >>> D_embed = results["D_embedding"]           # donors × DxC
    """
    import torch

    if runner is None:
        runner = get_livi_runner()

    if device is None:
        device = runner.resolve_device()

    adata, individual_col, _gdata = _resolve_adata(adata_or_dd, individual_col)

    LIVI = runner.get_livi_class()
    _, LIVIDataset = runner.get_datamodule_classes()

    logger.info("Loading LIVI checkpoint: %s", checkpoint_path)
    model = LIVI.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        initialise_training_mode=False,
    )
    model.eval()

    # Factorize donors — order must match training factorisation
    _, y_index = pd.factorize(
        adata.obs[individual_col], sort=False, use_na_sentinel=False
    )

    dataset = LIVIDataset(
        adata=adata,
        y_key=individual_col,
        use_size_factor=True,
        size_factor_key=None,
        layer_key=layer_key,
        covariates_keys=None,
        known_cis_eqtls=None,
        eqtl_genotypes=None,
        strict=False,
        backed_mode=False,
    )

    n_cells = len(dataset)
    # Build batch index lists
    batch_index_lists: List[List[int]] = []
    start = 0
    while start < n_cells:
        end = min(start + batch_size, n_cells)
        batch_index_lists.append(list(range(start, end)))
        start = end

    z_factor_cols = [f"Cell-state_Factor{f}" for f in range(1, model.z_dim + 1)]
    cell_state_chunks: List[pd.DataFrame] = []
    device_obj = torch.device(device)

    for idx_list in batch_index_lists:
        data = dataset.__getitem__(idx_list)
        with torch.no_grad():
            batch_res = model.predict(
                x=data["x"].to(device_obj),
                y=data["y"].to(device_obj),
            )
        z_np = batch_res["cell-state_latent"].detach().cpu().numpy()
        cell_state_chunks.append(
            pd.DataFrame(z_np, index=adata.obs.index[idx_list], columns=z_factor_cols)
        )

    cell_state_latent = pd.concat(cell_state_chunks, axis=0)

    results: Dict[str, pd.DataFrame] = {"cell_state_latent": cell_state_latent}

    # Cell-state decoder weights — shape (n_genes, z_dim)
    cs_dec_np = model.decoder.mean[0].weight.detach().cpu().numpy()
    results["cell_state_decoder"] = pd.DataFrame(
        cs_dec_np, index=adata.var.index, columns=z_factor_cols
    )

    # Donor embeddings — access weight tensors directly (avoids n_cells overhead)
    n_unique_donors = len(y_index)

    if model.n_DxC_factors != 0:
        D_np = model.D_context.weight.detach().cpu().numpy()[:n_unique_donors]
        dxc_cols = [f"D_Factor{f}" for f in range(1, model.n_DxC_factors + 1)]

        if variance_threshold is not None:
            keep = np.where(np.var(D_np, axis=0) >= variance_threshold)[0]
            D_np = D_np[:, keep]
            dxc_cols = [f"D_Factor{f+1}" for f in keep]

        results["D_embedding"] = pd.DataFrame(D_np, index=y_index, columns=dxc_cols)

        DxC_dec_np = model.decoder.DxC_decoder[0].weight.detach().cpu().numpy()
        results["DxC_decoder"] = pd.DataFrame(
            DxC_dec_np,
            index=adata.var.index,
            columns=[f"D_Factor{f}" for f in range(1, model.n_DxC_factors + 1)],
        )

        A_np = torch.sigmoid(model.A).detach().cpu().numpy()
        results["assignment_matrix"] = pd.DataFrame(
            A_np,
            index=z_factor_cols,
            columns=[f"D_Factor{f}" for f in range(1, model.n_DxC_factors + 1)],
        )

    if model.n_persistent_factors != 0:
        V_np = model.V_persistent.weight.detach().cpu().numpy()[:n_unique_donors]
        v_cols = [f"V_Factor{f}" for f in range(1, model.n_persistent_factors + 1)]
        results["V_embedding"] = pd.DataFrame(V_np, index=y_index, columns=v_cols)

        V_dec_np = model.decoder.persistent_decoder[0].weight.detach().cpu().numpy()
        results["V_decoder"] = pd.DataFrame(
            V_dec_np, index=adata.var.index, columns=v_cols
        )

    return results


def run_livi_association_testing(
    inference_results: Dict[str, pd.DataFrame],
    genotype_matrix: Union[str, "pd.DataFrame", object],
    output_dir: str,
    *,
    method: Literal["LMM", "TensorQTL"] = "LMM",
    kinship: Optional[Union[str, "pd.DataFrame"]] = None,
    genotype_pcs: Optional[Union[str, "pd.DataFrame"]] = None,
    covariates: Optional["pd.DataFrame"] = None,
    fdr_threshold: float = 0.05,
    fdr_method: str = "Benjamini-Hochberg",
    quantile_norm: bool = False,
    variance_threshold: Optional[float] = None,
    variable_factors: Optional[List[int]] = None,
    output_file_prefix: str = "livi",
    runner: Optional[LIVIRunner] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Run genetic association testing on LIVI donor embeddings.

    Performs trans-eQTL-style testing between LIVI's learned donor
    embeddings (D and/or V) and a genotype matrix, using either a linear
    mixed model (LMM / LIMIX) or TensorQTL.

    Parameters
    ----------
    inference_results : dict
        Output of :func:`infer_livi`.  Must contain at least one of
        ``"D_embedding"`` or ``"V_embedding"``.
    genotype_matrix : DonorData, str, or pd.DataFrame
        Genotype data.  When a :class:`~cellink.DonorData` object is passed,
        ``dd.G.X`` is used as the donors × SNPs genotype matrix, and
        ``kinship`` / ``genotype_pcs`` are automatically extracted from
        ``dd.G.uns["kinship"]`` and ``dd.G.obsm["gPCs"]`` respectively
        (unless explicitly overridden).  Alternatively, pass a TSV file path
        or a DataFrame directly.
    output_dir : str
        Directory where association result files are written.
    method : {"LMM", "TensorQTL"}
        Association testing method.  ``"LMM"`` uses LIMIX with a kinship
        matrix for relatedness correction; ``"TensorQTL"`` is faster but
        does not support repeated-measures designs.
    kinship : str or pd.DataFrame, optional
        Donors × donors kinship / GRM matrix.  Required for ``method="LMM"``.
        Auto-extracted from ``dd.G.uns["kinship"]`` when DonorData is passed.
    genotype_pcs : str or pd.DataFrame, optional
        Donors × PCs genotype principal-component matrix used as additional
        covariates.  Auto-extracted from ``dd.G.obsm["gPCs"]`` when DonorData
        is passed.
    covariates : pd.DataFrame, optional
        Extra donor-level covariates (donors × covariates) passed alongside
        genotype PCs during testing.
    fdr_threshold : float
        FDR threshold for significance.
    fdr_method : str
        FDR-controlling procedure (e.g. ``"Benjamini-Hochberg"``).
    quantile_norm : bool
        Quantile-normalise LIVI embeddings before testing.
    variance_threshold : float, optional
        Only test D factors with variance ≥ this threshold.
    variable_factors : list of int, optional
        Explicit (zero-based) factor indices to test.  Overrides
        ``variance_threshold``.
    output_file_prefix : str
        Common prefix for output TSV files.
    runner : LIVIRunner, optional
        Runner instance.  Uses the global runner when *None*.

    Returns
    -------
    pd.DataFrame or tuple
        If only D (or only V) embeddings are present: a single DataFrame
        with association results.  If both are present: a tuple
        ``(DxC_associations, V_associations)``.

    Examples
    --------
    Pass DonorData — genotype matrix, kinship, and gPCs are auto-extracted:

    >>> assoc = cl.tl.external.run_livi_association_testing(
    ...     results,
    ...     genotype_matrix=dd,
    ...     output_dir="livi_assoc",
    ...     method="LMM",
    ... )

    Or supply components explicitly:

    >>> assoc = cl.tl.external.run_livi_association_testing(
    ...     results,
    ...     genotype_matrix="genotypes.tsv",
    ...     output_dir="livi_assoc",
    ...     method="LMM",
    ...     kinship="kinship.tsv",
    ...     genotype_pcs="gpcs.tsv",
    ... )
    """
    if runner is None:
        runner = get_livi_runner()

    run_assoc = runner.get_association_testing_fn()
    os.makedirs(output_dir, exist_ok=True)

    D_context = inference_results.get("D_embedding")
    V_persistent = inference_results.get("V_embedding")

    if D_context is None and V_persistent is None:
        raise ValueError(
            "inference_results contains neither 'D_embedding' nor 'V_embedding'. "
            "Run infer_livi first."
        )

    # Resolve DonorData → extract dd.G components
    try:
        from cellink._core.donordata import DonorData as _DonorData

        if isinstance(genotype_matrix, _DonorData):
            gdata = genotype_matrix.G
            GT = _gdata_to_genotype_df(gdata)
            if kinship is None and "kinship" in gdata.uns:
                kinship = gdata.uns["kinship"]
            if genotype_pcs is None and "gPCs" in gdata.obsm:
                genotype_pcs = gdata.obsm["gPCs"]
            genotype_matrix = GT  # replace for downstream use
    except ImportError:
        pass

    # Load genotype matrix if path
    if isinstance(genotype_matrix, str):
        sep = "\t" if genotype_matrix.endswith(".tsv") else ","
        GT = pd.read_csv(genotype_matrix, index_col=0, sep=sep)
    else:
        GT = genotype_matrix  # already a DataFrame at this point

    # Load kinship if path
    if isinstance(kinship, str):
        sep = "\t" if kinship.endswith(".tsv") else ","
        kinship = pd.read_csv(kinship, index_col=0, sep=sep)
        kinship.index = kinship.index.astype(str)

    # Load genotype PCs if path
    if isinstance(genotype_pcs, str):
        sep = "\t" if genotype_pcs.endswith(".tsv") else ","
        genotype_pcs = pd.read_csv(genotype_pcs, index_col=0, sep=sep)
        genotype_pcs.index = genotype_pcs.index.astype(str)

    # livi_testing always does covariates["intercept"] = 1.0 — it must be a DataFrame.
    # Initialise with the donor index from D_context (or V_persistent) if not supplied.
    if covariates is None:
        ref = D_context if D_context is not None else V_persistent
        covariates = pd.DataFrame(index=ref.index)

    # For LMM, livi_testing does NOT merge genotype_pcs into covariates (only TensorQTL
    # does). Merge them here so they are available as fixed effects.
    if method in ("LMM", "LIMIX") and genotype_pcs is not None:
        covariates = covariates.merge(
            genotype_pcs, how="left", left_index=True, right_index=True
        )

    results = run_assoc(
        D_context=D_context,
        V_persistent=V_persistent,
        GT_matrix=GT,
        variant_info=None,
        Kinship=kinship,
        genotype_pcs=genotype_pcs,
        method=method,
        fdr_method=fdr_method,
        output_dir=output_dir,
        output_file_prefix=output_file_prefix,
        covariates=covariates,
        quantile_norm=quantile_norm,
        variance_threshold=variance_threshold,
        variable_factors=variable_factors,
        fdr_threshold=fdr_threshold,
        return_associations=True,
    )

    return results


def save_livi_results(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str = "livi",
    sep: str = "\t",
) -> Dict[str, str]:
    """Save :func:`infer_livi` results as TSV files.

    Parameters
    ----------
    results : dict
        Output of :func:`infer_livi`.
    output_dir : str
        Destination directory (created if absent).
    prefix : str
        Common filename prefix.
    sep : str
        Column separator for TSV files.

    Returns
    -------
    dict
        Mapping of result key → saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for key, df in results.items():
        if df is None:
            continue
        path = os.path.join(output_dir, f"{prefix}_{key}.tsv")
        df.to_csv(path, sep=sep)
        paths[key] = path
        logger.info("Saved %s → %s", key, path)
    return paths


def load_livi_results(
    output_dir: str,
    prefix: str = "livi",
    sep: str = "\t",
) -> Dict[str, pd.DataFrame]:
    """Load TSV files written by :func:`save_livi_results`.

    Parameters
    ----------
    output_dir : str
        Directory containing the TSV files.
    prefix : str
        Common filename prefix used during saving.
    sep : str
        Column separator.

    Returns
    -------
    dict
        Dictionary mapping result name → DataFrame.
    """
    keys = [
        "cell_state_latent",
        "cell_state_decoder",
        "D_embedding",
        "DxC_decoder",
        "V_embedding",
        "V_decoder",
        "assignment_matrix",
    ]
    results: Dict[str, pd.DataFrame] = {}
    for key in keys:
        path = os.path.join(output_dir, f"{prefix}_{key}.tsv")
        if os.path.exists(path):
            results[key] = pd.read_csv(path, sep=sep, index_col=0)
    return results
