# PGEN → AnnData Conversion (`cellink-pgen`)

`cellink-pgen` converts one or more PLINK2 `.pgen` files into a single AnnData-compatible Zarr v3 store. It streams dosage data in variant blocks, keeping memory use bounded regardless of dataset size, and produces output directly readable by `cellink.io.read_pgen_zarr`.

## Installation

`pgenlib` is required and installed via the optional extra:

```bash
pip install cellink[pgen]
```

## Basic Usage

```bash
# Single file, dense matrix (common variants)
cellink-pgen input.pgen -o output.zarr

# Single file, sparse matrix (rare variants)
cellink-pgen rare.pgen -o rare.zarr --sparse

# Merge multiple PGEN files into one matrix (same samples, disjoint variant sets)
cellink-pgen rare.pgen common.pgen -o combined.zarr
```

When multiple files are provided, variants are concatenated column-wise. All files must contain the **same samples in the same order** — obs is taken from the first file's `.psam`.

## Loading the Output

```python
from cellink.io import read_pgen_zarr

adata = read_pgen_zarr("combined.zarr")
# X is Dask-backed for dense stores; trigger computation with:
X = adata.X.compute()
```

---

## Dense vs Sparse Storage

The choice affects both write strategy and how `X` is stored on disk.

**Dense** (default): Genotype blocks are streamed directly into a pre-allocated Zarr dataset. `X` is stored as a chunked `int8` array and returned as a Dask-backed array — reads remain lazy.

```bash
cellink-pgen common.pgen -o common.zarr
```

**Sparse**: Each variant block is converted to a `scipy.csr_matrix`, accumulated in memory, then written via `AnnData.write_zarr`. Use this when variant density is low (e.g. rare variant sets), as it avoids storing large stretches of zeros.

```bash
cellink-pgen rare.pgen -o rare.zarr --sparse
```

---

## Chunking and Memory Control

| Argument | Default | Description |
|---|---|---|
| `--chunk-samples` | `4096` | Zarr chunk size along the sample axis |
| `--chunk-variants` | `2048` | Zarr chunk size along the variant axis |
| `--memory-limit` | `10.0` | Max RAM per read block, in GB |

The read block size (`vblock`) is computed as:

```
vblock = min(chunk_variants, floor(memory_limit_gb × 1024³ / n_samples), n_variants)
```

This means `--memory-limit` acts as a hard cap: no single `pgenlib.read_range` call will allocate more than the specified amount. Each block is written immediately to Zarr before the next is read, keeping the working set small.

Chunk sizes directly affect downstream Dask task graphs. Smaller chunks increase parallelism but add scheduling overhead; the defaults (`4096 × 2048`) are a reasonable starting point for biobank-scale data.

---

## Compression

Blosc with bitshuffle is applied to the dense Zarr dataset:

| Argument | Default | Options |
|---|---|---|
| `--compressor` | `zstd` | `zstd`, `lz4`, `zlib` |
| `--compression-level` | `7` | `1`–`9` |

Bitshuffle is particularly effective for low-cardinality integer arrays like dosage-encoded genotypes (values 0, 1, 2), often yielding better compression than byteshuffle at the same level.

---

## Other Options

| Argument | Description |
|---|---|
| `--max-variants N` | Cap variants per input file (useful for testing) |
| `--max-samples N` | Cap samples taken from the `.psam` |

---

## Output Structure

For dense mode, the Zarr store is written manually to keep `X` out of AnnData's normal write path (which would load it into memory). The layout is AnnData-compatible:

```
output.zarr/
├── X/          ← chunked int8 Zarr array, encoding-type: "array"
├── obs/        ← DataFrame from .psam (index: FID_IID)
├── var/        ← DataFrame from .pvar
├── uns/
├── obsm/
├── varm/
├── obsp/
├── varp/
└── layers/
```

Metadata is consolidated with `zarr.consolidate_metadata` at the end, making subsequent opens faster.

For sparse mode, `AnnData.write_zarr` handles the full write after the CSR matrix is assembled.
