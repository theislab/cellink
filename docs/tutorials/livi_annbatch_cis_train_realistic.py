# %% [markdown]
# # LIVI annbatch — JOINT C+G (cis-eQTL) mode, REALISTIC paper-scale run
#
# Full-scale counterpart to `livi_annbatch_cis_train.py` (fast smoke test: 4000
# genes, ~5000 SNPs, synthetic random cis assignment, 50 batches/epoch, single
# annbatch shard). Reads the same prebuilt realistic cache as
# `livi_baseline_cis_train_realistic.py` -- see that script's docstring and
# `build_onek1k_dd_cache_realistic.py` for what's realistic vs. approximated
# here (real genomic-distance cis windows instead of the paper's curated,
# unavailable association list; paper's exact warmup/epoch schedule).
#
# At this scale (14212 genes, full donor/cell count) the annbatch
# `DatasetCollection` should split into multiple shards (unlike the smoke test,
# where the filtered data fit in one), which is the actual point of comparing
# this against the baseline in-memory loader.
#
# Thin wrapper around `cellink.tl.external.train_livi_annbatch` -- the
# `DatasetCollection`/`CisGenotype`/batch-adapter machinery lives there, not here.

# %%
import os
import time
import warnings

import pandas as pd
import zarr
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

import cellink as cl  # noqa: F401
from cellink.tl.external import build_annbatch_collection, configure_livi_runner, read_g_from_dd_store, train_livi_annbatch

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
# zarrs' Rust threadpool defaults (threading.max_workers=None) to the node's full
# CPU count, NOT the SLURM/cgroup allocation -- on a shared GPU node (other users'
# jobs on the same physical machine) this oversubscribes badly. Pin it to what we
# actually got. See annbatch's zarr-configuration.md ("zarrs Performance").
zarr.config.set({"threading.max_workers": len(os.sched_getaffinity(0))})

LIVI_ROOT = "LIVI"
DONOR_KEY = "donor_id"
COVARIATE_KEYS = ["pool_number", "sex"]

DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"
# When CELLINK_SSD_HOME is set (e.g. by the SLURM script), data and collection
# are read/written from the SSD-backed filesystem for faster I/O.
SSD_HOME = os.environ.get("CELLINK_SSD_HOME", "")
if SSD_HOME:
    DD_CACHE_PATH_ZARR_REALISTIC = f"{SSD_HOME}/onek1k_realistic.dd.zarr"
    KNOWN_CIS_EQTLS_PATH = f"{SSD_HOME}/onek1k_realistic_known_cis_eqtls.parquet"
    C_COLLECTION = f"{SSD_HOME}/dd_C_collection_realistic.zarr"
    OUTPUT_DIR = f"{SSD_HOME}/livi_annbatch_run_realistic"
else:
    DD_CACHE_PATH_ZARR_REALISTIC = f"{DATA_HOME}/onek1k/onek1k_realistic.dd.zarr"
    KNOWN_CIS_EQTLS_PATH = f"{DATA_HOME}/onek1k/onek1k_realistic_known_cis_eqtls.parquet"
    C_COLLECTION = "dd_C_collection_realistic.zarr"
    OUTPUT_DIR = "livi_annbatch_run_realistic"

# --- loader knobs (SSD-tuned) ---
# Hyperparameters tuned by systematic sweep (livi_annbatch_hparam_sweep.py, job 38490780):
# batch_size=256 gives 2.5× higher cells/s than 1024 on H100 (super-linear memory
# scaling of the DxC backward pass). chunk_size and preload_nchunks make <5%
# difference — kept at annbatch defaults. preload_to_gpu=True always wins (~25%).
BATCH_SIZE = 256
CHUNK_SIZE = 512
PRELOAD_NCHUNKS = 32
COLLECTION_N_OBS_PER_CHUNK = 512   # fixed at collection-build time; matched to CHUNK_SIZE
COLLECTION_SHARD_SIZE = "2GB"      # 2× default (1GB); fewer file handles on SSD

CELL_STATE_CIS = False  # paper "cell-state" variant; set True to match exactly

# --- model / training config: paper values (LIVIcis_onek1k_10K-HVG-HEX.yaml) ---
Z_DIM = 15
N_DXC_FACTORS = 700
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [5000, 2000, 500, 100]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 60
WARMUP_EPOCHS_G = 0
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
BATCH_NORM_DECODER = True
GENETICS_SEED = 200
SEED = 42
MAX_EPOCHS = 600
MIN_EPOCHS = 160

seed_everything(SEED, workers=True)
runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
print(f"device: {DEVICE}")
if DEVICE != "cuda":
    warnings.warn("No GPU detected -- this paper-scale config will be extremely slow on CPU.", stacklevel=2)

PRELOAD_TO_GPU = DEVICE != "cpu"  # needs cupy-cuda12x; no GPU here means no cupy

for path in (DD_CACHE_PATH_ZARR_REALISTIC, KNOWN_CIS_EQTLS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_onek1k_dd_cache_full.py then "
            "build_onek1k_dd_cache_realistic.py first to build it."
        )

# Real cis genotype + real known_cis mask: both come straight from the
# prebuilt realistic cache -- the "G" group of onek1k_realistic.dd.zarr already
# IS the real cis-SNP set, and known_cis_eqtls.parquet is the real gene-SNP
# pairing from that same build step (build_onek1k_dd_cache_realistic.py) -- no
# synthetic rng.choice assignment here.
gdata = read_g_from_dd_store(DD_CACHE_PATH_ZARR_REALISTIC)
known_cis_eqtls = pd.read_parquet(KNOWN_CIS_EQTLS_PATH)
_dd_zarr = zarr.open(DD_CACHE_PATH_ZARR_REALISTIC, mode="r")

# Pre-build the annbatch collection with SSD-optimised chunk layout.
# n_obs_per_chunk and shard_size are written into the zarr store at build time
# and cannot be changed later without a full rebuild -- so we set them here
# rather than letting train_livi_annbatch use its (untuned) defaults.
# is_empty guard inside build_annbatch_collection means this is a no-op if
# C_COLLECTION was already built in a previous run.
print(f"building/verifying annbatch collection at {C_COLLECTION!r} ...")
t_coll = time.perf_counter()
build_annbatch_collection(
    _dd_zarr["C"], C_COLLECTION,
    n_obs_per_chunk=COLLECTION_N_OBS_PER_CHUNK,
    shard_size=COLLECTION_SHARD_SIZE,
    seed=SEED,
)
print(f"collection ready in {time.perf_counter() - t_coll:.1f}s")

# %% [markdown]
# ## Throughput callback + checkpointing + full paper-scale training

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
        print(f"[epoch {trainer.current_epoch}] {self.n} batches, {cells:,} cells in {dt:.1f}s "
              f"-> {cells / dt:,.0f} cells/s, {dt / max(self.n, 1) * 1000:.1f} ms/batch")


t0 = time.perf_counter()
trainer = train_livi_annbatch(
    _dd_zarr["C"], gdata,
    output_dir=OUTPUT_DIR,
    collection_path=C_COLLECTION,
    donor_key=DONOR_KEY,
    covariates_keys=COVARIATE_KEYS,
    known_cis_eqtls=known_cis_eqtls,  # real mapping -> cis_snps/target_genes/cis_genes_per_snp unused
    z_dim=Z_DIM, n_dxc_factors=N_DXC_FACTORS, n_persistent_factors=N_PERSISTENT_FACTORS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS, learning_rate=LEARNING_RATE,
    warmup_epochs_vae=WARMUP_EPOCHS_VAE, warmup_epochs_G=WARMUP_EPOCHS_G,
    max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, batch_size=BATCH_SIZE,
    chunk_size=CHUNK_SIZE, preload_nchunks=PRELOAD_NCHUNKS, preload_to_gpu=PRELOAD_TO_GPU,
    seed=SEED, l1_weight=L1_WEIGHT, A_weight=A_WEIGHT, batch_norm_decoder=BATCH_NORM_DECODER,
    genetics_seed=GENETICS_SEED, cell_state_cis=CELL_STATE_CIS,
    enable_progress_bar=True, log_every_n_steps=10, enable_logger=True,
    enable_checkpointing=True,  # we pass our own ModelCheckpoint below
    callbacks=[
        ThroughputCallback(),
        ModelCheckpoint(dirpath=f"{OUTPUT_DIR}/checkpoints", save_last=True,
                         monitor="train/livi_loss", mode="min", save_top_k=1),
    ],
)
print("\n==== ANNBATCH cis (REALISTIC paper-scale config) ====")
print(f"epochs: {trainer.current_epoch}  total wall: {time.perf_counter() - t0:.1f}s")

# %%
