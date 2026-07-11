# %% [markdown]
# # LIVI annbatch — JOINT C+G (cis-eQTL) mode, PAPER config
#
# Companion to `livi_baseline_cis_train.py`. Both run LIVI's onek1k cis-eQTL
# "cell-state" config from the paper (configs/model/LIVIcis_onek1k_10K-HVG-HEX),
# kept identical EXCEPT:
#   * full 36469 genes (not the paper's ~14K HVG subset)  -> heavier
#   * warmup_epochs_vae = 0 (paper 60) so the cis/genotype path is active during
#     a short benchmark instead of sitting in VAE-only pretrain
#   * limit_train_batches caps the run (a full epoch at this scale is hours)
#
# Difference between the two scripts: expression is streamed by annbatch here
# (via `cellink.tl.external.train_livi_annbatch`); the cis genotype (donors x N)
# is held in memory and looked up per cell, exactly like the stock LIVIDataset.
# Covariates (pool_number, sex) are streamed with the expression. So the
# comparison isolates the expression+obs dataloader.
#
# This script is a thin wrapper around `train_livi_annbatch` -- the
# `DatasetCollection`/`CisGenotype`/batch-adapter machinery it used to define
# inline now lives in `cellink.tl.external` (`_livi_annbatch.py`), reusable
# outside this script too.

# %%
import os
import time
import warnings

import zarr
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

import cellink as cl  # noqa: F401
from cellink.tl.external import configure_livi_runner, read_g_from_dd_store, train_livi_annbatch

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
# zarrs' Rust threadpool defaults (threading.max_workers=None) to the node's full
# CPU count, NOT the SLURM/cgroup allocation -- on a shared node this oversubscribes.
# Pin it to what we actually got. See annbatch's zarr-configuration.md.
zarr.config.set({"threading.max_workers": len(os.sched_getaffinity(0))})

LIVI_ROOT = "LIVI"
C_COLLECTION = "dd_C_collection.zarr"
DONOR_KEY = "donor_id"
COVARIATE_KEYS = ["pool_number", "sex"]  # paper covariates

# --- real genotype: G of the cached onek1k DonorData, built once by
# build_onek1k_dd_cache.py; train_livi_annbatch reads the "G"/"C" groups
# straight out of the .dd.zarr cache (no .vcz re-read).
DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"
DD_CACHE_PATH_ZARR = f"{DATA_HOME}/onek1k/onek1k.dd.zarr"

# --- loader knobs ---
BATCH_SIZE = 1024  # paper; drop if OOM (full genes is heavy)
CHUNK_SIZE = 128
PRELOAD_NCHUNKS = 16

# --- cis config (synthetic, deterministic; matches baseline) ---
# N_CIS_SNPS / N_TARGET_GENES each accept either:
#   * an int   -> the first N (SNPs over dd.G var axis, genes over collection's genes)
#   * a list/array of names    -> exactly those
#   * a boolean mask over the axis -> the selected entries
N_CIS_SNPS = 2996  # paper value
N_TARGET_GENES = 2000
CIS_GENES_PER_SNP = 5
CELL_STATE_CIS = False  # paper "cell-state" variant

# --- model / training config: matches baseline & paper ---
Z_DIM = 15
N_DXC_FACTORS = 700
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [5000, 2000, 500, 100]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 0  # paper 60; 0 here so cis path runs in benchmark
WARMUP_EPOCHS_G = 0
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
BATCH_NORM_DECODER = True
GENETICS_SEED = 200
SEED = 42
BENCH_BATCHES = 50  # cap (full epoch ~ hours at this scale)
MAX_EPOCHS = 10

seed_everything(SEED, workers=True)
runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
print(f"device: {DEVICE}")

PRELOAD_TO_GPU = DEVICE != "cpu"  # needs cupy-cuda12x; no GPU here means no cupy

if not os.path.exists(DD_CACHE_PATH_ZARR):
    raise FileNotFoundError(f"{DD_CACHE_PATH_ZARR} not found. Run build_onek1k_dd_cache.py first to build it.")

gdata = read_g_from_dd_store(DD_CACHE_PATH_ZARR)
_dd_zarr = zarr.open(DD_CACHE_PATH_ZARR, mode="r")

# %% [markdown]
# ## Throughput callback + train (capped at BENCH_BATCHES)

# %%
warnings.filterwarnings("ignore", message=r".*no logger configured.*")


class ThroughputCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.perf_counter()
        self.n = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.n += 1

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.perf_counter() - self.t0
        cells = self.n * BATCH_SIZE
        print(
            f"[epoch {trainer.current_epoch}] {self.n} batches, {cells:,} cells in {dt:.1f}s "
            f"-> {cells / dt:,.0f} cells/s, {dt / max(self.n, 1) * 1000:.1f} ms/batch"
        )


t0 = time.perf_counter()
trainer = train_livi_annbatch(
    _dd_zarr["C"],
    gdata,
    output_dir="livi_annbatch_run",
    collection_path=C_COLLECTION,
    donor_key=DONOR_KEY,
    covariates_keys=COVARIATE_KEYS,
    cis_snps=N_CIS_SNPS,
    target_genes=N_TARGET_GENES,
    cis_genes_per_snp=CIS_GENES_PER_SNP,
    z_dim=Z_DIM,
    n_dxc_factors=N_DXC_FACTORS,
    n_persistent_factors=N_PERSISTENT_FACTORS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
    learning_rate=LEARNING_RATE,
    warmup_epochs_vae=WARMUP_EPOCHS_VAE,
    warmup_epochs_G=WARMUP_EPOCHS_G,
    max_epochs=MAX_EPOCHS,
    min_epochs=1,
    batch_size=BATCH_SIZE,
    chunk_size=CHUNK_SIZE,
    preload_nchunks=PRELOAD_NCHUNKS,
    preload_to_gpu=PRELOAD_TO_GPU,
    seed=SEED,
    l1_weight=L1_WEIGHT,
    A_weight=A_WEIGHT,
    batch_norm_decoder=BATCH_NORM_DECODER,
    genetics_seed=GENETICS_SEED,
    cell_state_cis=CELL_STATE_CIS,
    limit_train_batches=BENCH_BATCHES,
    enable_progress_bar=True,
    log_every_n_steps=10,
    callbacks=[ThroughputCallback()],
)
print("\n==== ANNBATCH cis (paper config, full genes) ====")
print(
    f"epochs: {trainer.current_epoch}  batches/epoch cap: {BENCH_BATCHES}  total wall: {time.perf_counter() - t0:.1f}s"
)

# %%
