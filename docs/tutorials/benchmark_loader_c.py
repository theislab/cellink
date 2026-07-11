import time
from annbatch import DatasetCollection
from annbatch.samplers import RandomSampler
from annbatch import Loader
import numpy as np


def prepare_loader_C(batch_size=1024, chunk_size=128, preload_nchunks=64):
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

    return loader_C


if __name__ == "__main__":
    print("Preparing loader_C...")
    loader_C = prepare_loader_C(batch_size=1024, chunk_size=128, preload_nchunks=64)
    loader_C_iter = iter(loader_C)

    print("Pre-warming/Running first 5 iterations of next(loader_C_iter)...")
    for _ in range(5):
        b = next(loader_C_iter)

    print("Timing 50 iterations...")
    t0 = time.perf_counter()
    for _ in range(50):
        b = next(loader_C_iter)
    t_elapsed = time.perf_counter() - t0

    print(f"Total time for 50 iterations: {t_elapsed:.4f}s")
    print(f"Average time per iteration (next(loader_C_iter)): {t_elapsed/50:.4f}s ({t_elapsed/50 * 1000:.2f} ms)")
