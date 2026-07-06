# %% [markdown]
# # LIVI baseline — JOINT C+G (cis-eQTL) mode, PAPER config, default in-memory loader
#
# Companion to `livi_annbatch_cis_train.py`. Same model + same deterministic
# synthetic cis annotation; the ONLY difference is the dataloader — here the stock
# in-memory `LIVIDataModule` (loads the whole AnnData into RAM, torch DataLoader).
#
# Thin wrapper around `cellink.tl.external.train_livi` -- the
# `LIVIDataModule`/`LIVI`/`Trainer` wiring it used to define inline now lives
# in `_livi.py`'s `train_livi`, reusable outside this script too. We still pass
# `callbacks=[ThroughputCallback()]` and `limit_train_batches=BENCH_BATCHES` so
# this stays a clean mirror of the annbatch script.
#
# Deviations from the paper, identical to the annbatch script:
#   * full 36469 genes (paper ~14K HVG)
#   * warmup_epochs_vae = 0 (paper 60) so the cis path runs in a short benchmark
#   * limit_train_batches caps the run

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
from anndata.utils import asarray
from pytorch_lightning.callbacks import Callback

import cellink as cl  # noqa: F401
from cellink.io import read_h5_dd
from cellink.tl.external import configure_livi_runner, train_livi

LIVI_ROOT = "LIVI"
DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"
DD_CACHE_PATH_H5 = f"{DATA_HOME}/onek1k/onek1k.dd.h5"
COVARIATE_KEYS = ["pool_number", "sex"]

# --- cis config: matches annbatch script ---
N_CIS_SNPS = 2996
N_TARGET_GENES = 2000
CIS_GENES_PER_SNP = 5
CELL_STATE_CIS = False

# --- model / training config: matches annbatch script & paper ---
Z_DIM = 15
N_DXC_FACTORS = 700
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [5000, 2000, 500, 100]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 0
WARMUP_EPOCHS_G = 0
BATCH_SIZE = 1024
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
BATCH_NORM_DECODER = True
GENETICS_SEED = 200
SEED = 42
NUM_WORKERS = 15
BENCH_BATCHES = 50
MAX_EPOCHS = 10

runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
print(f"device: {DEVICE}")

# %% [markdown]
# ## Load real OneK1K (no celltype filter)
#
# Reads the cached `DonorData` built once by `build_onek1k_dd_cache.py` (run
# that script first if `DD_CACHE_PATH_H5` doesn't exist yet) instead of
# re-running the full `get_onek1k` ingestion pipeline here.

# %%
if not os.path.exists(DD_CACHE_PATH_H5):
    raise FileNotFoundError(
        f"{DD_CACHE_PATH_H5} not found. Run build_onek1k_dd_cache.py first to build it."
    )
t_load = time.perf_counter()
dd = read_h5_dd(DD_CACHE_PATH_H5)
load_secs = time.perf_counter() - t_load
N_GENES = dd.C.n_vars
N_DONORS = int(dd.C.obs[dd.donor_id].nunique())
print(f"loaded in {load_secs:.1f}s | cells: {dd.C.n_obs:,}  genes: {N_GENES:,}  "
      f"donors: {N_DONORS}  snps(total): {dd.G.n_vars:,}")

# %% [markdown]
# ## Deterministic synthetic cis annotation (same construction as annbatch script)

# %%
rng = np.random.default_rng(SEED)
snp_names = list(dd.G.var_names[:N_CIS_SNPS])
target_genes = list(dd.C.var_names[:N_TARGET_GENES])
eqtl_genotypes = pd.DataFrame(
    asarray(dd.G[:, :N_CIS_SNPS].X), index=dd.G.obs_names, columns=snp_names
)
# Real genotype calls can have missing entries (NaN); left as-is they'd propagate
# into a NaN decoder output once LIVI's V/DxC path activates. Mean-impute per SNP
# (standard eQTL practice) -- see livi_baseline_cis_train_realistic.py, which hits
# this for real at full SNP scale (these first N_CIS_SNPS happen to have none).
n_missing = int(eqtl_genotypes.isna().sum().sum())
if n_missing:
    print(f"imputing {n_missing} missing genotype calls with per-SNP mean dosage")
    eqtl_genotypes = eqtl_genotypes.fillna(eqtl_genotypes.mean())
known_cis_eqtls = pd.DataFrame(0, index=snp_names, columns=target_genes, dtype=int)
for s in snp_names:
    known_cis_eqtls.loc[s, rng.choice(target_genes, size=CIS_GENES_PER_SNP, replace=False)] = 1
print(f"cis set: {N_CIS_SNPS} SNPs | known_cis_eqtls: {known_cis_eqtls.shape}")

# %% [markdown]
# ## Train via `train_livi`, capped at BENCH_BATCHES

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
checkpoint_path = train_livi(
    dd,
    output_dir="livi_baseline_run",
    z_dim=Z_DIM, n_dxc_factors=N_DXC_FACTORS, n_persistent_factors=N_PERSISTENT_FACTORS,
    n_cis_snps=N_CIS_SNPS, cell_state_cis=CELL_STATE_CIS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS, learning_rate=LEARNING_RATE,
    covariates_keys=COVARIATE_KEYS,
    known_cis_eqtls=known_cis_eqtls, eqtl_genotypes=eqtl_genotypes,
    warmup_epochs_vae=WARMUP_EPOCHS_VAE, warmup_epochs_G=WARMUP_EPOCHS_G,
    max_epochs=MAX_EPOCHS, min_epochs=1, batch_size=BATCH_SIZE,
    seed=SEED, l1_weight=L1_WEIGHT, A_weight=A_WEIGHT, batch_norm_decoder=BATCH_NORM_DECODER,
    genetics_seed=GENETICS_SEED, num_workers=NUM_WORKERS,
    limit_train_batches=BENCH_BATCHES, enable_checkpointing=False, enable_logger=False,
    enable_progress_bar=True, log_every_n_steps=10,
    callbacks=[ThroughputCallback()],
)
fit_secs = time.perf_counter() - t0
print("\n==== BASELINE cis (default in-memory dataloader, paper config, full genes) ====")
print(f"data load:   {load_secs:.1f}s")
print(f"batches:     {BENCH_BATCHES}  total wall: {fit_secs:.1f}s")
print(f"checkpoint:  {checkpoint_path}")

# %%
