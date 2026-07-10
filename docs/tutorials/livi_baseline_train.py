# %% [markdown]
# # LIVI baseline (default in-memory dataloader) on real OneK1K
#
# Baseline to compare against `livi_annbatch_model_train.py`. This uses the
# **default** LIVI data path — the high-level `cl.tl.external.train_livi()`
# wrapper, which builds the stock in-memory `LIVIDataModule` (loads the whole
# AnnData into RAM, computes size factors up front, then iterates with a
# torch `DataLoader`).
#
# Everything that affects compute is kept identical to the annbatch script
# (model dims, batch size, epochs, warmup, optimiser) so the *only* difference
# between the two runs is the dataloader. No celltype filtering, no genotype /
# cis-eQTL part.
#
# Setup (run once, separately):
#     git clone https://github.com/theislab/cellink
#     # then in Python:
#     import cellink
#     cellink.resources.get_onek1k(
#         config_path="./cellink/resources/config/onek1k.yaml",
#         data_home="/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data",
#     )

# %%
import time

import cellink as cl  # noqa: F401
from cellink.resources import get_onek1k
from cellink.tl.external import configure_livi_runner, train_livi

# --- paths ---
LIVI_ROOT = "LIVI"
# In-repo config; if you cloned theislab/cellink separately, point this at that
# clone's onek1k.yaml instead (e.g. "./cellink/resources/config/onek1k.yaml").
CONFIG_PATH = "../../src/cellink/resources/config/onek1k.yaml"
DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"

# --- model / training config: MUST match livi_annbatch_model_train.py ---
Z_DIM = 15
N_DXC_FACTORS = 100
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [256, 128, 64]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 1
WARMUP_EPOCHS_G = 0
MAX_EPOCHS = 3
MIN_EPOCHS = 1
BATCH_SIZE = 2048
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
SEED = 42
# Default loader workers. 0 == single-process, matching the annbatch run.
NUM_WORKERS = 15

# %% [markdown]
# ## Configure LIVI runner

# %%
runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
print(f"device: {runner.resolve_device()}")

# %% [markdown]
# ## Load real OneK1K (no celltype filtering)
#
# `get_onek1k` returns a `DonorData`; `dd.C` is the cell-level AnnData with raw
# counts in `dd.C.X` and `donor_id` in `dd.C.obs`, and `dd.donor_id` names the
# individual column — exactly what `train_livi` consumes.

# %%
t_load = time.perf_counter()
dd = get_onek1k(config_path=CONFIG_PATH, data_home=DATA_HOME, verify_checksum=False)
load_secs = time.perf_counter() - t_load

n_cells = dd.C.n_obs
n_genes = dd.C.n_vars
n_donors = int(dd.C.obs[dd.donor_id].nunique())
print(f"loaded in {load_secs:.1f}s | cells: {n_cells:,}  genes: {n_genes:,}  donors: {n_donors}")

# %% [markdown]
# ## Train with the default in-memory dataloader and time it
#
# `train_livi` wraps model construction + the stock `LIVIDataModule` + a
# `pl.Trainer`. We can't inject a per-batch throughput callback through the
# wrapper, so we time the whole `fit` (which, as the default path, legitimately
# *includes* the one-time in-memory load + size-factor computation that annbatch
# avoids by streaming). The live progress bar also shows it/s per epoch.

# %%
t0 = time.perf_counter()
checkpoint_path = train_livi(
    dd,
    output_dir="livi_baseline_run",
    z_dim=Z_DIM,
    n_dxc_factors=N_DXC_FACTORS,
    n_persistent_factors=N_PERSISTENT_FACTORS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
    learning_rate=LEARNING_RATE,
    use_size_factor=True,
    layer_key=None,  # raw counts in dd.C.X
    warmup_epochs_vae=WARMUP_EPOCHS_VAE,
    warmup_epochs_G=WARMUP_EPOCHS_G,
    max_epochs=MAX_EPOCHS,
    min_epochs=MIN_EPOCHS,
    batch_size=BATCH_SIZE,
    l1_weight=L1_WEIGHT,
    A_weight=A_WEIGHT,
    num_workers=NUM_WORKERS,
    seed=SEED,
    enable_progress_bar=True,
)
fit_secs = time.perf_counter() - t0

cell_passes = n_cells * MAX_EPOCHS
print("\n==== BASELINE (default in-memory dataloader) ====")
print(f"best checkpoint:      {checkpoint_path}")
print(f"data load:            {load_secs:.1f}s")
print(f"train_livi fit total: {fit_secs:.1f}s  ({MAX_EPOCHS} epochs)")
print(f"aggregate throughput: {cell_passes / fit_secs:,.0f} cells/s "
      f"({cell_passes:,} cell-passes / {fit_secs:.1f}s)")
print(f"per-epoch wall (avg):  {fit_secs / MAX_EPOCHS:.1f}s")
