import time
import h5py
from annbatch import DatasetCollection
import anndata as ad
from annbatch.samplers import RandomSampler
from annbatch import Loader
import shutil
import numpy as np
import pandas as pd
import os
from annbatch import write_sharded
import zarr
import math
from annbatch.abc import Sampler
from annbatch.utils import split_given_size

class DonorSampler(Sampler):
    def __init__(self, categories, obs_names, preload_nbatches=4):
        self._batch_size = 1
        self._preload_nbatches = preload_nbatches
        self._categories = list(categories)
        
        name_to_idx = {name: i for i, name in enumerate(obs_names)}
        self.indices = [name_to_idx[cat] for cat in self._categories if cat in name_to_idx]
        
    @property
    def batch_size(self) -> int:
        return self._batch_size
        
    @property
    def shuffle(self) -> bool:
        return False
        
    def n_batches(self, n_obs: int) -> int:
        return math.ceil(len(self.indices) / self._batch_size)
        
    def n_iters(self, n_obs: int) -> int:
        return self.n_batches(n_obs)
        
    def validate(self, n_obs: int) -> None:
        if any(idx >= n_obs or idx < 0 for idx in self.indices):
            raise ValueError("Some mapped indices are out of bounds for the dataset.")
            
    def _sample(self, n_obs: int):
        step_size = self._preload_nbatches * self._batch_size
        for i in range(0, len(self.indices), step_size):
            block_indices = self.indices[i : i + step_size]
            chunks = [slice(idx, idx + 1) for idx in block_indices]
            splits = split_given_size(np.arange(len(block_indices)), self._batch_size)
            yield {
                "chunks": chunks,
                "splits": splits,
            }

def load_obs_from_random_sampler(sampler, obs):
    "gives the unique donor ids per batch."
    for lr in sampler.sample(len(obs)):
        chunks_key = lr.get("chunks", lr.get("requests"))
        indices = np.concatenate([np.arange(x.start,x.stop) for x in chunks_key])
        sampled_obs = obs.iloc[indices]
        for split in lr["splits"]:
            yield sampled_obs.iloc[split].unique()

def train_yields(n_epochs, batch_size, chunk_size, preload_nchunks, preload_ndonors):
    rs1 = RandomSampler(
        batch_size=batch_size, chunk_size=chunk_size, preload_nchunks=preload_nchunks,
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
    
    for epoch in range(n_epochs):
        flattened_donors = []
        n_donors_per_batch = []
        for batch_donors in load_obs_from_random_sampler(rs1, obs):
            flattened_donors.extend(batch_donors)
            n_donors_per_batch.append(len(batch_donors))
        
        donor_sampler = DonorSampler(categories=flattened_donors, obs_names=dd_G_obs_names, preload_nbatches=preload_ndonors)
        loader_G = Loader(
            return_index=True,
            batch_sampler=donor_sampler,
            preload_to_gpu=False,
        ).use_collection(dd_G_collection)
        loader_G_iter = iter(loader_G)

        n_donors_per_batch_iter = iter(n_donors_per_batch)
        for batch in loader_C:
            n_donors = next(n_donors_per_batch_iter)
            donors_g = []
            for _ in range(n_donors):
                batch_g = next(loader_G_iter)
                donors_g.append(batch_g)
            batch["donor_g"] = donors_g
            yield batch

a = train_yields(1, batch_size=1024, chunk_size=128, preload_nchunks=64, preload_ndonors=1024)

print("Pre-warming/Running first next(a)...")
b = next(a)

print("Timing 20 iterations...")
t0 = time.perf_counter()
for _ in range(20):
    b = next(a)
t_elapsed = time.perf_counter() - t0
print(f"Total time for 20 iterations: {t_elapsed:.4f}s")
print(f"Average time per iteration (next(a)): {t_elapsed/20:.4f}s ({t_elapsed/20 * 1000:.1f} ms)")
