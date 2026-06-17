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

def prepare_loader_G(preload_ndonors=120):
    dd_G_collection = DatasetCollection("dd_G_collection.zarr")
    dd_G_obs_names = []
    for group in dd_G_collection:
        dd_G_obs_names.extend(ad.io.read_elem(group["obs"]).index.tolist())
    
    # Generate a fixed sequence of donor names directly from the collection obs names
    rng = np.random.default_rng(42)
    # 500 requests is plenty for pre-warming and timing iterations
    flattened_donors = rng.choice(dd_G_obs_names, size=500, replace=True).tolist()
    
    donor_sampler = DonorSampler(categories=flattened_donors, obs_names=dd_G_obs_names, preload_nbatches=preload_ndonors)
    loader_G = Loader(
        return_index=True,
        batch_sampler=donor_sampler,
        preload_to_gpu=True,
    ).use_collection(dd_G_collection)
    
    return loader_G

if __name__ == "__main__":
    print("Preparing loader_G...")
    loader_G = prepare_loader_G(preload_ndonors=10)
    loader_G_iter = iter(loader_G)

    print("Pre-warming/Running first 5 iterations of next(loader_G_iter)...")
    for _ in range(5):
        b = next(loader_G_iter)

    print("Timing 50 iterations...")
    t0 = time.perf_counter()
    for _ in range(50):
        b = next(loader_G_iter)
    t_elapsed = time.perf_counter() - t0
    
    print(f"Total time for 50 iterations: {t_elapsed:.4f}s")
    print(f"Average time per iteration (next(loader_G_iter)): {t_elapsed/50:.4f}s ({t_elapsed/50 * 1000:.2f} ms)")
