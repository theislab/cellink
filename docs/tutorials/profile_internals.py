import time
import asyncio
import numpy as np
import pandas as pd
import anndata as ad
from annbatch import DatasetCollection, Loader
from annbatch.samplers import RandomSampler
import zarr.core.sync as zsync
from annbatch.loader import _cupy_dtype

# Setup collections to mimic the training run
print("Setting up loader_G...")
rs1 = RandomSampler(
    batch_size=1024, chunk_size=128, preload_nchunks=64,
    replacement=False, drop_last=True,
    rng=np.random.default_rng(42)    
)
loader_C = Loader(
    return_index=True,
    batch_sampler=rs1,
).use_collection(DatasetCollection("dd_C_collection.zarr"))

obs = pd.concat(x["donor_id"] for x in loader_C._obs)

dd_G_collection = DatasetCollection("dd_G_collection.zarr")
dd_G_obs_names = []
for group in dd_G_collection:
    dd_G_obs_names.extend(ad.io.read_elem(group["obs"]).index.tolist())

# Import DonorSampler
from profile_yields import DonorSampler

# Just sample the first 120 unique donors
unique_donors = list(obs.unique())[:120]

donor_sampler = DonorSampler(categories=unique_donors, obs_names=dd_G_obs_names, preload_nbatches=120)
loader_G = Loader(
    return_index=True,
    batch_sampler=donor_sampler,
).use_collection(dd_G_collection)

print("Preparing to run one load request of loader_G...")
# Trigger any lazy imports/initializations (CUDA setup, etc.)
iter_G = iter(loader_G)
next(iter_G)

# Now time the internal steps of Loader.__iter__ manually for one load request
print("\nProfiling internal operations:")

# Get the first load request
load_request = next(iter(donor_sampler.sample(loader_G.n_obs)))
chunks_to_load = load_request["chunks"]
splits = load_request["splits"]

# 1. slices to slices with array index
t0 = time.perf_counter()
dataset_index_to_slices = loader_G._slices_to_slices_with_array_index(chunks_to_load, use_original_space=False)
t_slices = time.perf_counter() - t0
print(f"1. _slices_to_slices_with_array_index: {t_slices:.4f}s")

# 2. Allocate output buffer
t0 = time.perf_counter()
out = loader_G._allocate_out(dataset_index_to_slices)
t_alloc = time.perf_counter() - t0
print(f"2. _allocate_out: {t_alloc:.4f}s")

# 3. Zarr index / disk read
t0 = time.perf_counter()
raw_out = zsync.sync(loader_G._index_datasets(dataset_index_to_slices))
t_read = time.perf_counter() - t0
print(f"3. _index_datasets (Zarr Disk Read): {t_read:.4f}s")

# 4. Host-to-Device Copy (asarray)
t0 = time.perf_counter()
in_memory_data = loader_G._np_module.asarray(raw_out)
t_copy = time.perf_counter() - t0
print(f"4. Host-to-Device Copy (_np_module.asarray): {t_copy:.4f}s")

# 5. Accumulate obs
t0 = time.perf_counter()
concatenated_obs = loader_G._maybe_accumulate_obs(dataset_index_to_slices)
t_obs = time.perf_counter() - t0
print(f"5. _maybe_accumulate_obs: {t_obs:.4f}s")

# 6. Accumulate indices
t0 = time.perf_counter()
in_memory_indices = loader_G._maybe_accumulate_indices(chunks_to_load)
t_indices = time.perf_counter() - t0
print(f"6. _maybe_accumulate_indices: {t_indices:.4f}s")

# 7. Loop over splits (fancy indexing, 120 batches)
t0 = time.perf_counter()
for split in splits:
    data = in_memory_data[split]
    X_val = data if not loader_G._to_torch else to_torch(data, loader_G._preload_to_gpu)
    obs_val = concatenated_obs.iloc[split] if concatenated_obs is not None else None
    index_val = in_memory_indices[split] if in_memory_indices is not None else None
t_loop = time.perf_counter() - t0
print(f"7. Loop over splits (fancy indexing, 120 batches): {t_loop:.4f}s")

# 8. Optimized Loop over splits using slices/views
t0 = time.perf_counter()
for split in splits:
    # Use slice/view when split is a contiguous array
    start, stop = split[0], split[-1] + 1
    data = in_memory_data[start:stop]
    X_val = data if not loader_G._to_torch else to_torch(data, loader_G._preload_to_gpu)
    obs_val = concatenated_obs.iloc[start:stop] if concatenated_obs is not None else None
    index_val = in_memory_indices[start:stop] if in_memory_indices is not None else None
t_opt = time.perf_counter() - t0
print(f"8. Optimized Loop over splits (slice indexing, 120 batches): {t_opt:.4f}s")
