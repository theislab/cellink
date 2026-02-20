import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Literal, Optional, Union

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.experimental import read_dispatched
from anndata.io import read_elem
from tqdm.auto import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def read_pgen_zarr(store: str | Path) -> ad.AnnData:
    """
    Lazily read an AnnData Zarr v3 store written by `stream_pgen_to_zarr`.

    This function reconstructs an :class:`anndata.AnnData` object from a Zarr
    store while keeping the primary data matrix (`X`) backed by Dask arrays.
    It is designed for large genotype matrices that cannot be loaded fully
    into memory.

    The reader preserves:
      - Dense X stored as a Zarr array (returned as a Dask-backed array)
      - Sparse matrices (CSR/CSC)
      - DataFrames (obs, var)
      - Awkward arrays
      - Standard AnnData container structure

    Parameters
    ----------
    store : str or pathlib.Path
        Path to a Zarr directory created by `stream_pgen_to_zarr`
        or a compatible AnnData Zarr v3 store.

    Returns
    -------
    anndata.AnnData
        AnnData object with:
        - `X` as a Dask-backed array (for dense storage)
        - `obs` and `var` as pandas DataFrames
        - empty container groups (`uns`, `obsm`, `varm`, `layers`, etc.)
          if present in the store

    Notes
    -----
    - The returned object is **lazy** when X is dense. Computation is triggered
      only when `.compute()` or in-memory materialization is requested.
    - For sparse X written via `stream_pgen_to_zarr(..., sparse=True)`,
      the matrix is loaded as a SciPy sparse matrix.
    - This function relies on AnnData's experimental dispatched I/O API.

    Examples
    --------
    >>> import cellink
    >>> adata = cellink.io.read_pgen_zarr("genotypes.zarr")
    >>> adata
    AnnData object with n_obs x n_vars = ...

    >>> # Trigger computation
    >>> X = adata.X.compute()
    """
    f = zarr.open(str(store), mode="r")

    def callback(func, elem_name: str, elem, iospec):
        if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
            return ad.AnnData(**{
                k: read_dispatched(v, callback)
                for k, v in dict(elem).items()
                if not k.startswith("raw.")
            })
        elif elem_name == "/X" and iospec.encoding_type in (
            "dataframe",
            "csr_matrix",
            "csc_matrix",
            "awkward-array",
        ):
            return read_elem(elem)
        elif elem_name == "/X" and iospec.encoding_type == "array":
            return da.from_zarr(elem)
        else:
            return func(elem)

    return read_dispatched(f, callback=callback)

def stream_pgen_to_zarr(
    pgen_path: Union[str, list[str]],
    output_path: str,
    *,
    max_variants: Optional[int] = None,
    max_samples: Optional[int] = None,
    chunk_samples: int = 4096,
    chunk_variants: int = 2048,
    memory_limit_gb: float = 10.0,
    compressor: str = "zstd",
    compression_level: int = 7,
    sparse: bool = False,
    sparse_format: Literal["csc", "csr"] = "csc",
) -> ad.AnnData:
    """
    Stream one or more PGEN files → a single Zarr v3 AnnData-compatible store.

    When multiple pgen_path entries are provided, they are treated as disjoint
    variant sets for the SAME set of samples (e.g. rare + common split pgens).
    Variants are concatenated column-wise; obs (samples) must be identical
    across all inputs and are taken from the first file.

    Parameters
    ----------
    pgen_path : str or list[str]
        Path(s) to .pgen file(s). Extension optional.
    output_path : str
        Output Zarr v3 directory.
    max_variants : int, optional
        Cap total variants (applied per file, then summed).
    max_samples : int, optional
        Cap samples (applied once from the first file).
    chunk_samples : int
        Zarr chunk size along sample axis.
    chunk_variants : int
        Zarr chunk size along variant axis.
    memory_limit_gb : float
        Max RAM per read block in GB.
    compressor : str
        Blosc compressor name ('zstd', 'lz4', 'zlib').
    compression_level : int
        Compression level 1-9.
    sparse : bool
        If True, accumulate a scipy CSR matrix and write as AnnData sparse X.
        If False (default), stream directly into a dense Zarr dataset.
        Use sparse=True for rare variants (low density); dense for common.
    sparse_format : {"csc", "csr"}
        Sparse matrix format when ``sparse=True``. Default is 'csc', which is
        more efficient for variant-wise access (e.g. association tests, per-variant
        filtering). Use 'csr' if your workload is primarily sample-wise (e.g.
        per-sample operations).

    Returns
    -------
    AnnData with Dask-backed X.
    """
    try:
        import pgenlib
    except ImportError:
        raise ImportError("pgenlib is required for `stream_pgen_to_zarr`. Install with `pip install cellink[pgen]`.")

    if isinstance(pgen_path, str):
        pgen_paths = [pgen_path]
    else:
        pgen_paths = list(pgen_path)

    def _base(p: str) -> str:
        return p[:-5] if p.endswith(".pgen") else p

    readers = []
    n_variants_per_file = []
    for p in pgen_paths:
        b = _base(p)
        pgen_file = Path(b + ".pgen")
        if not pgen_file.exists():
            raise FileNotFoundError(f"PGEN file not found: {pgen_file}")
        reader = pgenlib.PgenReader(bytes(str(pgen_file), "utf-8"))
        nv = reader.get_variant_ct()
        ns = reader.get_raw_sample_ct()
        if max_variants:
            nv = min(nv, max_variants)
        n_variants_per_file.append(nv)
        readers.append((reader, ns, _base(p)))
        logger.info(f"  {pgen_file.name}: {ns:,} samples x {reader.get_variant_ct():,} variants "
                    f"(using {nv:,})")

    raw_sample_counts = [ns for _, ns, _ in readers]
    if len(set(raw_sample_counts)) > 1:
        raise ValueError(
            f"All pgen files must have the same number of samples. "
            f"Got: {raw_sample_counts}"
        )
    n_samples_total = raw_sample_counts[0]
    n_samples = min(max_samples, n_samples_total) if max_samples else n_samples_total
    n_variants_total = sum(n_variants_per_file)

    logger.info(f"Total: {n_samples:,} samples × {n_variants_total:,} variants "
                f"({'sparse' if sparse else 'dense'})")

    first_base = readers[0][2]
    psam_file  = Path(first_base + ".psam")
    logger.info(f"Loading sample metadata from {psam_file} ...")
    psam = pd.read_csv(psam_file, sep="\t")
    psam.columns = [c.lower().replace("#", "") for c in psam.columns]
    if "fid" in psam.columns and "iid" in psam.columns:
        psam["genotype_id"] = psam["fid"].astype(str) + "_" + psam["iid"].astype(str)
        psam = psam.set_index("genotype_id")
    psam = psam.iloc[:n_samples].copy()
    psam.columns = psam.columns.astype(str)
    psam.index   = psam.index.astype(str)

    pvar_frames = []
    for nv, (_, _, base) in zip(n_variants_per_file, readers):
        pvar_file = Path(base + ".pvar")
        pv = pd.read_csv(pvar_file, sep="\t", comment="#", header=None)
        pv = pv.iloc[:nv].copy()
        pv.columns = pv.columns.astype(str)
        pv.index   = pv.index.astype(str)
        pvar_frames.append(pv)
    pvar = pd.concat(pvar_frames, ignore_index=True)
    pvar.index = pvar.index.astype(str)

    output_path = Path(output_path)
    blosc = BloscCodec(
        cname=compressor,
        clevel=compression_level,
        shuffle=BloscShuffle.bitshuffle,
    )

    if sparse:
        if sparse_format not in ("csr", "csc"):
            raise ValueError(f"sparse_format must be 'csr' or 'csc', got {sparse_format!r}")

        csr_chunks: list[sp.csr_matrix] = []

        for (reader, n_raw, base), nv in zip(readers, n_variants_per_file):
            pgen_file = Path(base + ".pgen")
            logger.info(f"Reading (sparse) {pgen_file.name} ...")
            bytes_per_variant  = n_raw
            max_vblock = max(100, int((memory_limit_gb * 1024 ** 3) / bytes_per_variant))
            vblock     = min(chunk_variants, max_vblock, nv)
            n_blocks   = (nv + vblock - 1) // vblock

            with tqdm(total=nv, desc=f"  {pgen_file.stem}", unit="var") as pbar:
                for i in range(n_blocks):
                    start = i * vblock
                    end   = min(start + vblock, nv)
                    width = end - start
                    buf   = np.zeros((n_raw, width), dtype=np.int8)
                    reader.read_range(start, end, geno_int_out=buf, sample_maj=1)
                    buf[buf < 0] = 0
                    csr_chunks.append(sp.csr_matrix(buf[:n_samples, :].astype(np.int8)))
                    del buf
                    pbar.update(width)
                    if i % 5 == 0:
                        gc.collect()

        logger.info(f"Stacking {len(csr_chunks)} CSR chunks ...")
        X = sp.hstack(csr_chunks, format=sparse_format)
        del csr_chunks
        logger.info(
            f"Sparse X: {X.shape}, nnz={X.nnz:,}, "
            f"density={X.nnz / (X.shape[0] * X.shape[1]) * 100:.4f}%"
        )
        adata = ad.AnnData(X=X, obs=psam, var=pvar)
        logger.info(f"Writing sparse AnnData → {output_path} ...")
        adata.write_zarr(output_path)

    else:
        root = zarr.open_group(output_path, mode="w")
        logger.info(
            f"Creating dense X ({n_samples:,} × {n_variants_total:,}) "
            f"chunks=({chunk_samples}, {chunk_variants}) ..."
        )
        Xz = root.create_dataset(
            "X",
            shape=(n_samples, n_variants_total),
            chunks=(chunk_samples, chunk_variants),
            dtype="i1",
            compressors=(blosc,),
        )
        Xz.attrs["encoding-type"] = "array"
        Xz.attrs["encoding-version"] = "0.2.0"

        col_offset = 0
        for (reader, n_raw, base), nv in zip(readers, n_variants_per_file):
            pgen_file = Path(base + ".pgen")
            logger.info(f"Reading (dense) {pgen_file.name} ...")
            bytes_per_variant  = n_raw
            max_vblock = max(100, int((memory_limit_gb * 1024 ** 3) / bytes_per_variant))
            vblock     = min(chunk_variants, max_vblock, nv)
            n_blocks   = (nv + vblock - 1) // vblock

            with tqdm(total=nv, desc=f"  {pgen_file.stem}", unit="var") as pbar:
                for i in range(n_blocks):
                    start = i * vblock
                    end   = min(start + vblock, nv)
                    width = end - start
                    buf   = np.zeros((n_raw, width), dtype=np.int8)
                    reader.read_range(start, end, geno_int_out=buf, sample_maj=1)
                    buf[buf < 0] = 0
                    Xz[:, col_offset + start: col_offset + end] = buf[:n_samples, :]
                    del buf
                    pbar.update(width)
                    if i % 5 == 0:
                        gc.collect()

            col_offset += nv

        logger.info("Writing obs / var / metadata ...")
        ad.io.write_elem(root, "obs", psam)
        ad.io.write_elem(root, "var", pvar)
        for gname in ["uns", "obsm", "varm", "obsp", "varp", "layers"]:
            root.create_group(gname)
        zarr.consolidate_metadata(root.store)

    logger.info(f"✓ Done → {output_path}")
