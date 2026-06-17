# %% [markdown]
# # LIVI on the annbatch dataloader (speed test)
#
# Goal: train the **LIVI model itself** (no celltype filtering, no genotype /
# cis-eQTL part for now) using the `annbatch` on-disk dataloader instead of the
# default in-memory `LIVIDataModule`.
#
# The default LIVI dataloader (`src/data_modules/livi_data.py`) loads the whole
# AnnData into memory and yields per-batch dicts:
#
#     {"x": cells x genes float32,
#      "y": donor integer codes (long),
#      "size_factor": cells x 1 float32}
#
# We reproduce exactly that batch contract on top of an `annbatch.Loader` reading
# the sharded zarr collection `dd_C_collection.zarr` (the real OneK1K 1k dataset,
# processed for annbatch compatibility).

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

import cellink as cl  # noqa: F401  (ensures cellink import path is set up)
from cellink.tl.external import configure_livi_runner

# zarrs codec pipeline is required for fast local-filesystem zarr reads
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

LIVI_ROOT = "LIVI"
COLLECTION_PATH = "dd_C_collection.zarr"
DONOR_KEY = "donor_id"

# Loader / training knobs (tuned for a single H100)
BATCH_SIZE = 2048
CHUNK_SIZE = 256
PRELOAD_NCHUNKS = 16
SHUFFLE = True
DROP_LAST = True
MAX_EPOCHS = 3
SEED = 42

seed_everything(SEED, workers=True)

runner = configure_livi_runner(livi_root=LIVI_ROOT, execution_mode="python_api", device="auto")
DEVICE = runner.resolve_device()
LIVI = runner.get_livi_class()
print(f"device: {DEVICE}")


# %% [markdown]
# ## Global donor -> integer code mapping
#
# The model's donor embedding (`D_context`, `V_persistent`) has `y_dim` rows, so
# every cell's `donor_id` must map to a fixed integer in `[0, n_donors)`. We build
# that mapping once from the collection's `donor_id` categories (read cheaply from
# the categorical sub-element, without pulling the full obs frame).

# %%
def read_donor_categories(collection_path: str, donor_key: str) -> list:
    """Return the union of donor_id categories across all groups in the collection."""
    cats: list = []
    seen = set()
    for group in DatasetCollection(collection_path):
        donor_elem = ad.io.read_elem(group["obs"][donor_key])
        if hasattr(donor_elem, "categories"):
            values = list(donor_elem.categories)
        else:
            values = pd.unique(np.asarray(donor_elem)).tolist()
        for v in values:
            if v not in seen:
                seen.add(v)
                cats.append(v)
    return cats


donor_categories = read_donor_categories(COLLECTION_PATH, DONOR_KEY)
DONOR_DTYPE = pd.CategoricalDtype(categories=donor_categories, ordered=False)
N_DONORS = len(donor_categories)
print(f"n_donors: {N_DONORS}")


# %% [markdown]
# ## annbatch loader -> LIVI batch adapter
#
# `load_only_x_and_donor` keeps the per-group AnnData minimal (only `X` + the
# donor column) so we don't pay to read all 30 obs columns. `LIVIBatchAdapter`
# turns each annbatch batch into the `{x, y, size_factor}` dict the LIVI model's
# `prepare_batch` expects, and is re-iterable so Lightning can reuse it across
# epochs.

# %%
def load_only_x_and_donor(group: zarr.Group) -> ad.AnnData:
    return ad.AnnData(
        X=ad.io.sparse_dataset(group["X"]),
        obs=ad.io.read_elem(group["obs"])[[DONOR_KEY]],
    )


class LIVIBatchAdapter:
    """Wrap an annbatch ``Loader`` and yield LIVI-format batch dicts."""

    def __init__(self, loader: Loader, donor_dtype: pd.CategoricalDtype):
        self._loader = loader
        self._donor_dtype = donor_dtype

    def __iter__(self):
        for batch in self._loader:
            x = batch["X"]  # (cells, genes) float32, already on GPU if preloaded
            # annbatch returns sparse (CSR) X; the LIVI decoder needs dense x and a
            # dense size_factor (the stock LIVIDataset also densifies via todense()).
            if x.layout != torch.strided:
                x = x.to_dense()
            donor = batch["obs"][DONOR_KEY]
            codes = np.ascontiguousarray(donor.astype(self._donor_dtype).cat.codes.to_numpy())
            if (codes < 0).any():
                raise ValueError("Found donor_id values outside the global donor categories.")
            y = torch.as_tensor(codes, dtype=torch.long, device=x.device)
            size_factor = x.sum(dim=1, keepdim=True)
            yield {"x": x, "y": y, "size_factor": size_factor}

    def __len__(self):
        return len(self._loader)


class AnnbatchLIVIDataModule(LightningDataModule):
    def __init__(
        self,
        collection_path: str,
        donor_dtype: pd.CategoricalDtype,
        batch_size: int,
        chunk_size: int,
        preload_nchunks: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ):
        super().__init__()
        self.collection_path = collection_path
        self.donor_dtype = donor_dtype
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.preload_nchunks = preload_nchunks
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def setup(self, stage=None):
        self.collection = DatasetCollection(self.collection_path)

    def train_dataloader(self):
        loader = Loader(
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            preload_nchunks=self.preload_nchunks,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            to_torch=True,
            preload_to_gpu=True,
            rng=np.random.default_rng(self.seed),
        ).use_collection(self.collection, load_adata=load_only_x_and_donor)
        return LIVIBatchAdapter(loader, self.donor_dtype)


# %% [markdown]
# ## Throughput callback

# %%
class ThroughputCallback(Callback):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.epoch_start = None
        self.n_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.perf_counter()
        self.n_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.n_batches += 1

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.perf_counter() - self.epoch_start
        n_cells = self.n_batches * self.batch_size
        print(
            f"[epoch {trainer.current_epoch}] {self.n_batches} batches, "
            f"{n_cells:,} cells in {dt:.1f}s -> {n_cells / dt:,.0f} cells/s, "
            f"{dt / max(self.n_batches, 1) * 1000:.1f} ms/batch"
        )


# %% [markdown]
# ## Build datamodule + model and train

# %%
datamodule = AnnbatchLIVIDataModule(
    collection_path=COLLECTION_PATH,
    donor_dtype=DONOR_DTYPE,
    batch_size=BATCH_SIZE,
    chunk_size=CHUNK_SIZE,
    preload_nchunks=PRELOAD_NCHUNKS,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
    seed=SEED,
)

# Infer n_genes from the collection var
N_GENES = ad.io.read_elem(list(DatasetCollection(COLLECTION_PATH))[0]["var"]).shape[0]
print(f"n_genes: {N_GENES}")

model = LIVI(
    x_dim=N_GENES,
    z_dim=15,
    y_dim=N_DONORS,
    n_DxC_factors=100,
    n_persistent_factors=5,
    n_cis_snps=0,
    encoder_hidden_dims=[256, 128, 64],
    learning_rate=8e-4,
    warmup_epochs_vae=1,   # short warmup so DxC/V paths are exercised in the speed test
    warmup_epochs_G=0,
    covariates_dims=None,
    l1_weight=1e-3,
    A_weight=1e-3,
    device=DEVICE,
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=1,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    callbacks=[ThroughputCallback(BATCH_SIZE)],
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    log_every_n_steps=10,
    deterministic=False,
)

t0 = time.perf_counter()
trainer.fit(model=model, datamodule=datamodule)
print(f"\nTotal trainer.fit wall time: {time.perf_counter() - t0:.1f}s")
