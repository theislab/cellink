import time
from annbatch import DatasetCollection
import anndata as ad
from annbatch.samplers import RandomSampler
from annbatch import Loader
import numpy as np
import pandas as pd
import math
from annbatch.abc import Sampler
from annbatch.utils import split_given_size


class DonorSampler(Sampler):
    """Custom sampler to yield donor observations in a specific sequence,
    loading preload_nbatches batches from disk at a time to minimize I/O overhead.
    """

    def __init__(self, categories, obs_names, preload_nbatches=4):
        self._batch_size = 1
        self._preload_nbatches = preload_nbatches
        self._categories = list(categories)

        # Map category names to their corresponding integer indices in dd_G
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

            # Load the block of rows from disk
            chunks = [slice(idx, idx + 1) for idx in block_indices]

            # Split the preloaded rows into batches of size batch_size
            splits = split_given_size(np.arange(len(block_indices)), self._batch_size)

            yield {
                "chunks": chunks,
                "splits": splits,
            }


def load_obs_from_random_sampler(sampler, obs):
    """Gives the unique donor ids per batch."""
    for lr in sampler.sample(len(obs)):
        chunks_key = lr.get("chunks", lr.get("requests"))
        indices = np.concatenate([np.arange(x.start, x.stop) for x in chunks_key])
        sampled_obs = obs.iloc[indices]
        for split in lr["splits"]:
            yield sampled_obs.iloc[split].unique()


def profile_train_yields(n_epochs, batch_size, chunk_size, preload_nchunks, preload_ndonors, max_batches=20):
    print("Initializing samplers and collections...")

    t_start = time.perf_counter()
    rs1 = RandomSampler(
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        replacement=False,
        drop_last=True,
        rng=np.random.default_rng(42),
    )

    loader_C = Loader(
        return_index=True,
        batch_sampler=rs1,
    ).use_collection(DatasetCollection("dd_C_collection.zarr"))

    obs = pd.concat(x["donor_id"] for x in loader_C._obs)

    # Get observation names (donor IDs) of dd_G from the collection
    dd_G_collection = DatasetCollection("dd_G_collection.zarr")
    dd_G_obs_names = []
    for group in dd_G_collection:
        dd_G_obs_names.extend(ad.io.read_elem(group["obs"]).index.tolist())

    t_init = time.perf_counter() - t_start
    print(f"Initialization took {t_init:.4f} seconds.")

    total_C_time = 0.0
    total_G_time = 0.0
    total_donor_sampler_setup = 0.0
    total_other_time = 0.0

    batch_count = 0

    for _epoch in range(n_epochs):
        t0 = time.perf_counter()
        flattened_donors = []
        n_donors_per_batch = []
        for batch_donors in load_obs_from_random_sampler(rs1, obs):
            flattened_donors.extend(batch_donors)
            n_donors_per_batch.append(len(batch_donors))

        donor_sampler = DonorSampler(
            categories=flattened_donors, obs_names=dd_G_obs_names, preload_nbatches=preload_ndonors
        )
        loader_G = Loader(
            return_index=True,
            batch_sampler=donor_sampler,
        ).use_collection(dd_G_collection)
        loader_G_iter = iter(loader_G)

        loader_C_iter = iter(loader_C)
        n_donors_per_batch_iter = iter(n_donors_per_batch)

        total_donor_sampler_setup += time.perf_counter() - t0

        while True:
            # Time fetching cells from loader_C
            t0 = time.perf_counter()
            try:
                batch = next(loader_C_iter)
            except StopIteration:
                break
            total_C_time += time.perf_counter() - t0

            # Time fetching donors from loader_G
            t0 = time.perf_counter()
            try:
                n_donors = next(n_donors_per_batch_iter)
            except StopIteration:
                break

            donors_g = []
            for _ in range(n_donors):
                try:
                    batch_g = next(loader_G_iter)
                    donors_g.append(batch_g)
                except StopIteration:
                    break
            total_G_time += time.perf_counter() - t0

            # Post-processing time (creating output batch structure)
            t0 = time.perf_counter()
            batch["donor_g"] = donors_g
            total_other_time += time.perf_counter() - t0

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                print(f"Reached max_batches limit ({max_batches}). Stopping profile run.")
                break

        if max_batches is not None and batch_count >= max_batches:
            break

    print("\n--- Profiling Results ---")
    print(f"Total batches processed: {batch_count}")
    print(
        f"Total time spent fetching cells (loader_C): {total_C_time:.4f}s (avg: {total_C_time/batch_count:.4f}s per batch)"
    )
    print(
        f"Total time spent fetching donors (loader_G): {total_G_time:.4f}s (avg: {total_G_time/batch_count:.4f}s per batch)"
    )
    print(f"Donor Sampler Setup time: {total_donor_sampler_setup:.4f}s")
    print(f"Other loop overhead: {total_other_time:.4f}s")

    total_profiled = total_C_time + total_G_time + total_donor_sampler_setup + total_other_time
    print(f"Total profiled loop time: {total_profiled:.4f}s")
    print(f"Cells Fetching contribution: {(total_C_time / total_profiled) * 100:.2f}%")
    print(f"Donors Fetching contribution: {(total_G_time / total_profiled) * 100:.2f}%")


if __name__ == "__main__":
    profile_train_yields(1, batch_size=1024, chunk_size=128, preload_nchunks=64, preload_ndonors=120, max_batches=20)
