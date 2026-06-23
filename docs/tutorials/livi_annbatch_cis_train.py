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
from anndata.utils import asarray
from annbatch import DatasetCollection, Loader
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback

import cellink as cl  # noqa: F401
from cellink.io import read_sgkit_zarr
from cellink.resources._utils import get_data_home
from cellink.tl.external import configure_livi_runner

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

LIVI_ROOT = "LIVI"
C_COLLECTION = "dd_C_collection.zarr"
DONOR_KEY = "donor_id"
COVARIATE_KEYS = ["pool_number", "sex"]   # paper covariates

# --- real genotype: OneK1K .vcz, read lazily (dask-backed) via cl.io ---
DATA_HOME = "/lustre/groups/ml01/workspace/lucas.arnoldt/data/cellink_data"
VCZ_SUBPATH = "onek1k/OneK1K.noGP.vcz"     # resolved under get_data_home(DATA_HOME)

# --- loader knobs ---
BATCH_SIZE = 1024                          # paper; drop if OOM (full genes is heavy)
CHUNK_SIZE = 128
PRELOAD_NCHUNKS = 16
SHUFFLE = True
DROP_LAST = True
PRELOAD_TO_GPU = True                      # needs cupy-cuda12x

# --- cis config (synthetic, deterministic; matches baseline) ---
# N_CIS_SNPS / N_TARGET_GENES each accept either:
#   * an int   -> the first N (SNPs over dd.G var axis, genes over GENE_NAMES)
#   * a list/array of names    -> exactly those
#   * a boolean mask over the axis -> the selected entries
N_CIS_SNPS = 2996                          # paper value
N_TARGET_GENES = 2000
CIS_GENES_PER_SNP = 5
CELL_STATE_CIS = False                      # paper "cell-state" variant

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
MAX_EPOCHS = 10

seed_everything(SEED, workers=True)
runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
LIVI = runner.get_livi_class()
print(f"device: {DEVICE}")


# %% [markdown]
# ## Global categorical mappings (donor + covariates) and gene names

# %%
def _categories(series):
    """Global category list for an obs column (categorical -> its categories, else unique)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(pd.unique(series))


_collection = DatasetCollection(C_COLLECTION)
GENE_NAMES = list(ad.io.read_elem(list(_collection)[0]["var"]).index)
N_GENES = len(GENE_NAMES)

# annbatch >= 0.2.0: pull all needed obs columns once, concatenated across the
# whole collection, instead of iterating the groups by hand.
_obs = _collection.obs(columns=[DONOR_KEY] + COVARIATE_KEYS)

DONOR_CATEGORIES = _categories(_obs[DONOR_KEY])
DONOR_DTYPE = pd.CategoricalDtype(categories=DONOR_CATEGORIES, ordered=False)
N_DONORS = len(DONOR_CATEGORIES)

COVAR_DTYPES = {k: pd.CategoricalDtype(categories=_categories(_obs[k]), ordered=False)
                for k in COVARIATE_KEYS}
COVARIATES_DIMS = [len(COVAR_DTYPES[k].categories) for k in COVARIATE_KEYS]
print(f"n_donors: {N_DONORS}  n_genes: {N_GENES}  covariates_dims: {COVARIATES_DIMS}")


# Encode donor + covariates to global integer codes ONCE, as torch tensors in
# collection.obs() row order. The loader (return_index=True) yields batch["index"]
# = global positions in that SAME order (use_collection adds datasets in
# _dataset_keys order, which is also how collection.obs() concatenates), so the
# per-batch adapter is a pure tensor gather -- no pandas lookup, codes stay on GPU.
_code_dev = torch.device(DEVICE)


def _code_tensor(column, dtype, what):
    codes = column.astype(dtype).cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"{what} value outside global categories")
    return torch.as_tensor(np.ascontiguousarray(codes), dtype=torch.long, device=_code_dev)


DONOR_CODES = _code_tensor(_obs[DONOR_KEY], DONOR_DTYPE, DONOR_KEY)
COVAR_CODES = {k: _code_tensor(_obs[k], COVAR_DTYPES[k], k) for k in COVARIATE_KEYS}


# %% [markdown]
# ## Lazy cis genotype (donors x N) + known_cis mask (N x genes)

# %%
def _resolve_indices(spec, names, what):
    """int (first N) | sequence of names | boolean mask -> integer positions into `names`."""
    names = [str(n) for n in names]
    n = len(names)
    if isinstance(spec, (int, np.integer)):
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


class CisGenotype:
    """Lazily-loaded cis-eQTL genotype + deterministic synthetic known-cis mask.

    The OneK1K ``.vcz`` is opened with :func:`cl.io.read_sgkit_zarr`, whose ``X``
    stays dask-backed, so the full genotype matrix is never materialized and the
    expression is never touched -- only the selected cis-SNP columns are computed.

    ``cis_snps`` / ``target_genes`` each accept an int (first N), a sequence of
    names, or a boolean mask over the respective axis. Resulting tensors:

    * ``gt_array``  : (n_donors x n_cis_snps) float32 presence, row-aligned to
      ``donor_categories`` (the streamed expression's donor order).
    * ``known_cis`` : (n_cis_snps x n_genes) int64 mask over ``gene_names``.
    """

    def __init__(
        self,
        donor_categories,
        gene_names,
        *,
        data_home=None,
        vcz_subpath=VCZ_SUBPATH,
        cis_snps=2996,
        target_genes=2000,
        cis_genes_per_snp=5,
        seed=42,
        device="cpu",
    ):
        gene_names = [str(g) for g in gene_names]
        donor_categories = list(donor_categories)
        dev = torch.device(device)

        gdata = read_sgkit_zarr(get_data_home(data_home) / vcz_subpath)  # dask-backed, lazy
        snp_idx = _resolve_indices(cis_snps, gdata.var_names, "cis SNPs")
        gene_idx = _resolve_indices(target_genes, gene_names, "target genes")

        self.gt_array = self._presence(gdata, snp_idx, donor_categories).to(dev)
        self.known_cis = self._known(len(snp_idx), len(gene_names), gene_idx,
                                     cis_genes_per_snp, seed).to(dev)

    @property
    def n_cis_snps(self):
        return self.gt_array.shape[1]

    @staticmethod
    def _presence(gdata, snp_idx, donor_categories):
        # Only the selected SNP columns are pulled out of the lazy store here.
        gt_block = (asarray(gdata[:, snp_idx].X) > 0).astype(np.float32)
        pos = {d: i for i, d in enumerate(gdata.obs_names)}
        missing = [c for c in donor_categories if c not in pos]
        if missing:
            raise ValueError(f"{len(missing)} expression donors absent from genotype, e.g. {missing[:3]}")
        rows = [pos[c] for c in donor_categories]
        return torch.from_numpy(np.ascontiguousarray(gt_block[rows]))

    @staticmethod
    def _known(n_snps, n_genes, gene_idx, cis_genes_per_snp, seed):
        rng = np.random.default_rng(seed)
        known = np.zeros((n_snps, n_genes), dtype=np.int64)
        for i in range(n_snps):
            known[i, rng.choice(gene_idx, size=cis_genes_per_snp, replace=False)] = 1
        return torch.from_numpy(known)


cis = CisGenotype(
    DONOR_CATEGORIES, GENE_NAMES,
    data_home=DATA_HOME, vcz_subpath=VCZ_SUBPATH,
    cis_snps=N_CIS_SNPS, target_genes=N_TARGET_GENES,
    cis_genes_per_snp=CIS_GENES_PER_SNP, seed=SEED, device=DEVICE,
)
GT_ARRAY, KNOWN_CIS = cis.gt_array, cis.known_cis
print(f"GT_array: {tuple(GT_ARRAY.shape)}  known_cis: {tuple(KNOWN_CIS.shape)}")


# %% [markdown]
# ## annbatch -> LIVI cis-batch adapter (with covariates)

# %%
def load_x_with_index(group):
    # The adapter gathers codes by batch["index"] (return_index=True), so only X
    # is needed here. One obs column (DONOR_KEY) is kept solely so the first batch
    # can self-check that batch["index"] aligns with collection.obs() order.
    return ad.AnnData(
        X=ad.io.sparse_dataset(group["X"]),
        obs=ad.io.read_elem(group["obs"])[[DONOR_KEY]],
    )


def _check_index_alignment(batch, pos):
    """One-time guard: gathered donor codes must match the batch's own donor_id."""
    obs = batch.get("obs")
    if obs is None:
        return
    expected = obs[DONOR_KEY].astype(DONOR_DTYPE).cat.codes.to_numpy()
    got = DONOR_CODES[pos].cpu().numpy()
    if not np.array_equal(expected, got):
        raise RuntimeError(
            "batch['index'] does not align with collection.obs() row order; "
            "positional code gather would mislabel cells."
        )


class LIVICisBatchAdapter:
    def __init__(self, loader):
        self._loader = loader
        self._checked = False

    def __iter__(self):
        for batch in self._loader:
            x = batch["X"]
            if x.layout != torch.strided:
                x = x.to_dense()
            # Pure tensor gather: batch["index"] are global positions into the
            # code tensors (built in collection.obs() order). Codes live on GPU.
            pos = torch.as_tensor(batch["index"], dtype=torch.long, device=DONOR_CODES.device)
            y = DONOR_CODES[pos]                                      # cells
            if not self._checked:
                _check_index_alignment(batch, pos)
                self._checked = True
            gt_cells = GT_ARRAY[y]                                    # cells x N
            covariates = [COVAR_CODES[k][pos].to(x.device) for k in COVARIATE_KEYS]
            yield {
                "x": x,
                "y": y.to(x.device),
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
            return_index=True,
        ).use_collection(self.collection, load_adata=load_x_with_index)
        return LIVICisBatchAdapter(loader)


# %% [markdown]
# ## Throughput callback + train (capped at BENCH_BATCHES)

# %%
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*no logger configured.*",
)
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
    n_cis_snps=cis.n_cis_snps, cell_state_cis=CELL_STATE_CIS,
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

# %%
