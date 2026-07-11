# %% [markdown]
# # LIVI baseline — JOINT C+G (cis-eQTL) mode, REALISTIC paper-scale run
#
# Full-scale counterpart to `livi_baseline_cis_train.py` (which is a fast smoke
# test: 4000 genes, ~5000 SNPs, synthetic random cis assignment, 50
# batches/epoch). This script matches the paper config
# (`LIVIcis_onek1k_10K-HVG-HEX.yaml`) as closely as possible given what's
# available in this environment:
#
#   * x_dim = 14212 HVGs (paper's exact value) and the real genomic-distance
#     cis SNP set, both precomputed by `build_onek1k_dd_cache_realistic.py`
#     (not the paper's literal gene list / curated cis association TSV --
#     neither is available here, so the gene/SNP sets will differ from the
#     paper's, even though the *counts* and *method* match its config/intent).
#   * warmup_epochs_vae=60, max_epochs=600, min_epochs=160, no
#     `limit_train_batches` cap -- matches the paper's trainer/model config.
#   * data_split=[1.0] -- the paper itself trains on the full set with no
#     held-out split (see the datamodule config), so this is not a realism gap.
#
# Thin wrapper around `cellink.tl.external.train_livi` -- same as
# `livi_baseline_cis_train.py`, just with checkpointing/logging left at their
# `train_livi` defaults (both `True`) since this is a real training run, not a
# benchmark.
#
# Needs a GPU node and `onek1k_realistic.dd.h5` --
# build it with `build_onek1k_dd_cache_full.py` + `build_onek1k_dd_cache_realistic.py`.

# %%
import os
import time
import warnings

import pandas as pd
from anndata.utils import asarray
from pytorch_lightning.callbacks import Callback

import cellink as cl  # noqa: F401
from cellink.io import read_h5_dd
from cellink.tl.external import configure_livi_runner, train_livi

LIVI_ROOT = "LIVI"
DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"
DD_CACHE_PATH_H5_REALISTIC = f"{DATA_HOME}/onek1k/onek1k_realistic.dd.h5"
KNOWN_CIS_EQTLS_PATH = f"{DATA_HOME}/onek1k/onek1k_realistic_known_cis_eqtls.parquet"
COVARIATE_KEYS = ["pool_number", "sex"]

CELL_STATE_CIS = False  # paper "cell-state" variant; set True to match exactly

# --- model / training config: paper values (LIVIcis_onek1k_10K-HVG-HEX.yaml) ---
Z_DIM = 15
N_DXC_FACTORS = 700
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [5000, 2000, 500, 100]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 60
WARMUP_EPOCHS_G = 0
BATCH_SIZE = 1024
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
BATCH_NORM_DECODER = True
GENETICS_SEED = 200
SEED = 42
NUM_WORKERS = 4  # was 15 in the smoke-test script; at full paper scale (14212 genes,
# 1.25M cells, a 13588x14212 known_cis_eqtls one-hot) each forked
# DataLoader worker ends up touching/copying large Python objects
# (refcount writes break copy-on-write), multiplying memory -- this
# OOM-killed the job at NUM_WORKERS=15 even with 128GB requested.
MAX_EPOCHS = 600
MIN_EPOCHS = 160

runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
print(f"device: {DEVICE}")
if DEVICE != "cuda":
    warnings.warn("No GPU detected -- this paper-scale config will be extremely slow on CPU.", stacklevel=2)

# %% [markdown]
# ## Load the prebuilt realistic cache (genes/SNPs already filtered, see
# ## `build_onek1k_dd_cache_realistic.py`)

# %%
for path in (DD_CACHE_PATH_H5_REALISTIC, KNOWN_CIS_EQTLS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_onek1k_dd_cache_full.py then "
            "build_onek1k_dd_cache_realistic.py first to build it."
        )

t_load = time.perf_counter()
dd = read_h5_dd(DD_CACHE_PATH_H5_REALISTIC)
known_cis_eqtls = pd.read_parquet(KNOWN_CIS_EQTLS_PATH)
load_secs = time.perf_counter() - t_load
N_CIS_SNPS = dd.G.n_vars
eqtl_genotypes = pd.DataFrame(asarray(dd.G.X), index=dd.G.obs_names, columns=dd.G.var_names)
# Real genotype calls have missing entries (NaN) -- e.g. ~1.2% here. LIVI's V/DxC
# path (activated after warmup_epochs_vae) feeds these dosages straight into the
# decoder; left as NaN they propagate into a NaN decoder output and crash
# training right when V/DxC turns on. Mean-impute per SNP (standard eQTL
# practice) instead of leaving them as NaN or silently zeroing them.
n_missing = int(eqtl_genotypes.isna().sum().sum())
if n_missing:
    print(
        f"imputing {n_missing} missing genotype calls ({n_missing / eqtl_genotypes.size:.2%}) "
        "with per-SNP mean dosage"
    )
    eqtl_genotypes = eqtl_genotypes.fillna(eqtl_genotypes.mean())
print(
    f"loaded in {load_secs:.1f}s | cells: {dd.C.n_obs:,}  genes: {dd.C.n_vars:,}  "
    f"donors: {dd.G.n_obs}  cis SNPs: {N_CIS_SNPS:,}"
)

# %% [markdown]
# ## Train via `train_livi`, full paper-scale (checkpointing + logging stay on)

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
checkpoint_path = train_livi(
    dd,
    output_dir="livi_baseline_run_realistic",
    z_dim=Z_DIM,
    n_dxc_factors=N_DXC_FACTORS,
    n_persistent_factors=N_PERSISTENT_FACTORS,
    n_cis_snps=N_CIS_SNPS,
    cell_state_cis=CELL_STATE_CIS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
    learning_rate=LEARNING_RATE,
    covariates_keys=COVARIATE_KEYS,
    known_cis_eqtls=known_cis_eqtls,
    eqtl_genotypes=eqtl_genotypes,
    warmup_epochs_vae=WARMUP_EPOCHS_VAE,
    warmup_epochs_G=WARMUP_EPOCHS_G,
    max_epochs=MAX_EPOCHS,
    min_epochs=MIN_EPOCHS,
    batch_size=BATCH_SIZE,
    seed=SEED,
    l1_weight=L1_WEIGHT,
    A_weight=A_WEIGHT,
    batch_norm_decoder=BATCH_NORM_DECODER,
    genetics_seed=GENETICS_SEED,
    num_workers=NUM_WORKERS,
    deterministic=True,
    enable_progress_bar=True,
    log_every_n_steps=10,
    callbacks=[ThroughputCallback()],
)
fit_secs = time.perf_counter() - t0
print("\n==== BASELINE cis (REALISTIC paper-scale config) ====")
print(f"data load:   {load_secs:.1f}s")
print(f"total wall:  {fit_secs:.1f}s")
print(f"checkpoint:  {checkpoint_path}")

# %%
