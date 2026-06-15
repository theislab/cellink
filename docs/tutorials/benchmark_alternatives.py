import time
import zarr
import numpy as np
import anndata as ad
import dask.array as da
from annbatch import DatasetCollection

# Open the zarr group
z_group = zarr.open("dd_G_collection.zarr/dataset_0", mode="r")

# 1. Zarr direct read (using Zarr array indexing)
z_arr = z_group["X"]
print(f"Zarr array shape: {z_arr.shape}, chunks: {z_arr.chunks}")

t0 = time.perf_counter()
row_zarr = z_arr[0]
t_zarr = time.perf_counter() - t0
print(f"Zarr direct read of 1 row: {t_zarr:.4f}s")

# 2. Zarr direct slice read (using slice)
t0 = time.perf_counter()
row_zarr_slice = z_arr[0:1]
t_zarr_slice = time.perf_counter() - t0
print(f"Zarr slice read of 1 row [0:1]: {t_zarr_slice:.4f}s")

# 3. Dask array read
dask_arr = da.from_array(z_arr, chunks=z_arr.chunks)

t0 = time.perf_counter()
row_dask = dask_arr[0].compute()
t_dask = time.perf_counter() - t0
print(f"Dask array read (compute) of 1 row: {t_dask:.4f}s")

