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
# Difference between the two scripts: expression is streamed by annbatch here;
# the cis genotype (donors x N) is held in memory and looked up per cell, exactly
# like the stock LIVIDataset. Covariates (pool_number, sex) are streamed with the
# expression. So the comparison isolates the expression+obs dataloader.

# %%
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
import zarr
from annbatch import DatasetCollection, Loader
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback

import cellink as cl  # noqa: F401
from cellink.tl.external import configure_livi_runner

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

LIVI_ROOT = "LIVI"
C_COLLECTION = "dd_C_collection.zarr"
G_ZARR = "dd_G.zarr"
DONOR_KEY = "donor_id"
COVARIATE_KEYS = ["pool_number", "sex"]   # paper covariates

# --- loader knobs ---
BATCH_SIZE = 1024                          # paper; drop if OOM (full genes is heavy)
CHUNK_SIZE = 128
PRELOAD_NCHUNKS = 16
SHUFFLE = True
DROP_LAST = True
PRELOAD_TO_GPU = True                      # needs cupy-cuda12x

# --- cis config (synthetic, deterministic; matches baseline) ---
N_CIS_SNPS = 2996                          # paper value
N_TARGET_GENES = 2000
CIS_GENES_PER_SNP = 5
CELL_STATE_CIS = True                      # paper "cell-state" variant

# --- model / training config: matches baseline & paper ---
Z_DIM = 15
N_DXC_FACTORS = 700
N_PERSISTENT_FACTORS = 5
ENCODER_HIDDEN_DIMS = [5000, 2000, 500, 100]
LEARNING_RATE = 8e-4
WARMUP_EPOCHS_VAE = 0                       # paper 60; 0 here so cis path runs in benchmark
WARMUP_EPOCHS_G = 0
L1_WEIGHT = 1e-3
A_WEIGHT = 1e-3
BATCH_NORM_DECODER = True
GENETICS_SEED = 200
SEED = 42
BENCH_BATCHES = 50                         # cap (full epoch ~ hours at this scale)
MAX_EPOCHS = 1

seed_everything(SEED, workers=True)
runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
LIVI = runner.get_livi_class()
print(f"device: {DEVICE}")


# %% [markdown]
# ## Global categorical mappings (donor + covariates) and gene names

# %%
def read_categories(collection_path, key):
    cats, seen = [], set()
    for group in DatasetCollection(collection_path):
        elem = ad.io.read_elem(group["obs"][key])
        values = list(elem.categories) if hasattr(elem, "categories") else pd.unique(np.asarray(elem)).tolist()
        for v in values:
            if v not in seen:
                seen.add(v)
                cats.append(v)
    return cats


_first_group = list(DatasetCollection(C_COLLECTION))[0]
GENE_NAMES = list(ad.io.read_elem(_first_group["var"]).index)
N_GENES = len(GENE_NAMES)

DONOR_CATEGORIES = read_categories(C_COLLECTION, DONOR_KEY)
DONOR_DTYPE = pd.CategoricalDtype(categories=DONOR_CATEGORIES, ordered=False)
N_DONORS = len(DONOR_CATEGORIES)

COVAR_DTYPES = {k: pd.CategoricalDtype(categories=read_categories(C_COLLECTION, k), ordered=False)
                for k in COVARIATE_KEYS}
COVARIATES_DIMS = [len(COVAR_DTYPES[k].categories) for k in COVARIATE_KEYS]
print(f"n_donors: {N_DONORS}  n_genes: {N_GENES}  covariates_dims: {COVARIATES_DIMS}")


# %% [markdown]
# ## In-memory cis genotype (donors x N) + known_cis mask (N x genes)

# %%
def build_cis_tensors():
    zg = zarr.open(G_ZARR, mode="r")
    g_donors = list(ad.io.read_elem(zg["obs"]).index)
    gt_block = (np.asarray(zg["X"][:, :N_CIS_SNPS]) > 0).astype(np.float32)  # presence
    pos = {d: i for i, d in enumerate(g_donors)}
    missing = [c for c in DONOR_CATEGORIES if c not in pos]
    if missing:
        raise ValueError(f"{len(missing)} expression donors absent from genotype, e.g. {missing[:3]}")
    gt_array = torch.from_numpy(np.ascontiguousarray(gt_block[[pos[c] for c in DONOR_CATEGORIES]]))

    rng = np.random.default_rng(SEED)
    target_genes = GENE_NAMES[:N_TARGET_GENES]
    gene_to_col = {g: i for i, g in enumerate(GENE_NAMES)}
    known = np.zeros((N_CIS_SNPS, N_GENES), dtype=np.int64)
    for i in range(N_CIS_SNPS):
        for h in rng.choice(target_genes, size=CIS_GENES_PER_SNP, replace=False):
            known[i, gene_to_col[h]] = 1
    # Put the constant cis tensors on the GPU once so they are not re-transferred.
    dev = torch.device(DEVICE)
    return gt_array.to(dev), torch.from_numpy(known).to(dev)


GT_ARRAY, KNOWN_CIS = build_cis_tensors()
print(f"GT_array: {tuple(GT_ARRAY.shape)}  known_cis: {tuple(KNOWN_CIS.shape)}")


# %% [markdown]
# ## annbatch -> LIVI cis-batch adapter (with covariates)

# %%
def load_x_donor_covars(group):
    cols = [DONOR_KEY] + COVARIATE_KEYS
    return ad.AnnData(
        X=ad.io.sparse_dataset(group["X"]),
        obs=ad.io.read_elem(group["obs"])[cols],
    )


class LIVICisBatchAdapter:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            x = batch["X"]
            if x.layout != torch.strided:
                x = x.to_dense()
            obs = batch["obs"]
            codes = np.ascontiguousarray(obs[DONOR_KEY].astype(DONOR_DTYPE).cat.codes.to_numpy())
            if (codes < 0).any():
                raise ValueError("donor_id outside global donor categories")
            y_cpu = torch.as_tensor(codes, dtype=torch.long)
            gt_cells = GT_ARRAY[y_cpu.to(GT_ARRAY.device)]            # cells x N
            covariates = [
                torch.as_tensor(
                    np.ascontiguousarray(obs[k].astype(COVAR_DTYPES[k]).cat.codes.to_numpy()),
                    dtype=torch.long,
                ).to(x.device)
                for k in COVARIATE_KEYS
            ]
            yield {
                "x": x,
                "y": y_cpu.to(x.device),
                "size_factor": x.sum(dim=1, keepdim=True),
                "covariates": covariates,
                "GT_cells": gt_cells,
                "known_cis": KNOWN_CIS,
            }

    def __len__(self):
        return len(self._loader)


class AnnbatchLIVICisDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.collection = DatasetCollection(C_COLLECTION)

    def train_dataloader(self):
        loader = Loader(
            batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE, preload_nchunks=PRELOAD_NCHUNKS,
            shuffle=SHUFFLE, drop_last=DROP_LAST, to_torch=True,
            preload_to_gpu=PRELOAD_TO_GPU, rng=np.random.default_rng(SEED),
        ).use_collection(self.collection, load_adata=load_x_donor_covars)
        return LIVICisBatchAdapter(loader)


# %% [markdown]
# ## Throughput callback + train (capped at BENCH_BATCHES)

# %%
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
              f"-> {cells / dt:,.0f} cells/s, {dt / max(self.n,1)*1000:.1f} ms/batch")


model = LIVI(
    x_dim=N_GENES, z_dim=Z_DIM, y_dim=N_DONORS,
    n_DxC_factors=N_DXC_FACTORS, n_persistent_factors=N_PERSISTENT_FACTORS,
    n_cis_snps=N_CIS_SNPS, cell_state_cis=CELL_STATE_CIS,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS, learning_rate=LEARNING_RATE,
    warmup_epochs_vae=WARMUP_EPOCHS_VAE, warmup_epochs_G=WARMUP_EPOCHS_G,
    covariates_dims=COVARIATES_DIMS, l1_weight=L1_WEIGHT, A_weight=A_WEIGHT,
    batch_norm_decoder=BATCH_NORM_DECODER, genetics_seed=GENETICS_SEED, device=DEVICE,
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS, min_epochs=1, limit_train_batches=BENCH_BATCHES,
    accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
    callbacks=[ThroughputCallback()],
    logger=False, enable_checkpointing=False, enable_progress_bar=True,
    log_every_n_steps=10, deterministic=False,
)

t0 = time.perf_counter()
trainer.fit(model=model, datamodule=AnnbatchLIVICisDataModule())
print(f"\n==== ANNBATCH cis (paper config, full genes) ====")
print(f"batches: {BENCH_BATCHES}  total wall: {time.perf_counter() - t0:.1f}s")
