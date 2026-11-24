from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

PathLike = str | os.PathLike

_BED_MAGIC = b"\x6c\x1b"
_VARIANT_MAJOR = 1

# genotype value (copies of the column-5 .bim allele A1, 3 = missing) -> 2-bit BED code
_VALUE_TO_CODE = np.array([0b11, 0b10, 0b00, 0b01], dtype=np.uint8)


def _bed_bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def encode_bed_genotypes(values: np.ndarray) -> np.ndarray:
    """Pack a ``(variants, samples)`` matrix of genotype values into BED bytes.

    Parameters
    ----------
    values
        Integer array with values 0, 1, 2 (copies of the column-5 ``.bim`` allele A1)
        and 3 for missing calls.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(variants, ceil(samples / 4))``.
    """
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D (variants, samples) array, got shape {values.shape}")
    n_variants, n_samples = values.shape
    codes = _VALUE_TO_CODE[values]
    pad = (-n_samples) % 4
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    codes = codes.reshape(n_variants, -1, 4)
    packed = codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)
    return packed.astype(np.uint8, copy=False)


def _dosage_to_bed_values(block: np.ndarray, *, warned: list[bool]) -> np.ndarray:
    """Map a float/integer dosage block to BED values (0, 1, 2, 3 = missing)."""
    block = np.asarray(block, dtype=np.float64)
    missing = np.isnan(block)
    if not np.all(missing):
        observed = block[~missing]
        rounded = np.rint(observed)
        if (rounded < 0).any() or (rounded > 2).any():
            raise ValueError(
                "PLINK 1 binary files only store hard calls 0, 1, 2 (copies of `a1`) or missing; "
                f"found values in [{observed.min()}, {observed.max()}]."
            )
        if not warned[0] and not np.array_equal(rounded, observed):
            warnings.warn(
                "Non-integer dosages are rounded to the nearest hard call (0, 1, 2) when written to "
                "PLINK 1 binary format.",
                UserWarning,
                stacklevel=4,
            )
            warned[0] = True
        block = np.where(missing, 3.0, np.rint(block))
    else:
        block = np.full(block.shape, 3.0)
    return block.astype(np.uint8)


def _dense_block(X, start: int, stop: int) -> np.ndarray:
    """Return columns ``start:stop`` of a (samples, variants) matrix as a dense numpy array.

    Negative values of a signed integer matrix are missing calls, as in the int8 genotype
    stores of :func:`cellink.io.read_zarr_hd` and :func:`cellink.io.read_pgen_zarr` with
    ``backend="zarr"``, and become ``nan``.
    """
    block = X[:, start:stop]
    if hasattr(block, "compute"):
        block = block.compute()
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block)
    if np.issubdtype(block.dtype, np.signedinteger):
        block = np.where(block < 0, np.float32(np.nan), block.astype(np.float32))
    return block


def write_bed(path: PathLike, X, *, variant_chunk_size: int | None = None) -> None:
    """Write a variant-major ``.bed`` file from a ``(samples, variants)`` dosage matrix.

    Parameters
    ----------
    path
        Output ``.bed`` path.
    X
        ``(samples, variants)`` matrix (numpy, dask, scipy sparse, or anything that
        supports 2D slicing) of genotype values 0, 1, 2 (copies of the column-5
        ``.bim`` allele A1) and ``nan`` for missing. Non-integer dosages are rounded
        to the nearest hard call.
    variant_chunk_size
        Number of variants encoded at once. Defaults to roughly 16 million genotypes
        per chunk.
    """
    n_samples, n_variants = X.shape
    if variant_chunk_size is None:
        variant_chunk_size = max(1, (1 << 24) // max(n_samples, 1))
    warned = [False]
    with open(path, "wb") as fh:
        fh.write(_BED_MAGIC + bytes([_VARIANT_MAJOR]))
        for start in range(0, n_variants, variant_chunk_size):
            stop = min(start + variant_chunk_size, n_variants)
            values = _dosage_to_bed_values(_dense_block(X, start, stop), warned=warned)
            fh.write(encode_bed_genotypes(values.T).tobytes())


def write_fam(path: PathLike, fam: pd.DataFrame) -> None:
    """Write a tab-separated ``.fam`` file.

    ``fam`` needs the columns ``fid``, ``iid``, ``father``, ``mother``, ``sex`` and
    ``phenotype``; every value is written as text.
    """
    cols = ["fid", "iid", "father", "mother", "sex", "phenotype"]
    out = pd.DataFrame({c: pd.Series(np.asarray(fam[c])).astype("string") for c in cols})
    out.to_csv(path, sep="\t", header=False, index=False, encoding="ascii", lineterminator="\n")


def write_bim(path: PathLike, bim: pd.DataFrame) -> None:
    """Write a tab-separated ``.bim`` file.

    ``bim`` needs the columns ``chrom``, ``snp``, ``cm``, ``pos``, ``a1`` (column 5,
    PLINK's A1, the allele genotype values count; ALT under the PLINK 2 convention)
    and ``a2`` (column 6, PLINK's A2; REF).
    """
    out = pd.DataFrame(
        {
            "chrom": pd.Series(np.asarray(bim["chrom"])).astype("string"),
            "snp": pd.Series(np.asarray(bim["snp"])).astype("string"),
            "cm": pd.Series(np.asarray(bim["cm"], dtype=np.float64)),
            "pos": pd.Series(np.asarray(bim["pos"])).astype(np.int64),
            "a1": pd.Series(np.asarray(bim["a1"])).astype("string"),
            "a2": pd.Series(np.asarray(bim["a2"])).astype("string"),
        }
    )
    out.to_csv(path, sep="\t", header=False, index=False, encoding="ascii", lineterminator="\n")


def write_plink1(
    prefix: PathLike,
    X,
    fam: pd.DataFrame,
    bim: pd.DataFrame,
    *,
    variant_chunk_size: int | None = None,
) -> None:
    """Write a PLINK 1 binary file set ``{prefix}.bed``, ``.bim`` and ``.fam``.

    Parameters
    ----------
    prefix
        Output path without extension.
    X
        ``(samples, variants)`` genotype matrix counting copies of ``bim["a1"]``, the
        column-5 allele.
    fam
        Sample table, see :func:`write_fam`.
    bim
        Variant table, see :func:`write_bim`.
    variant_chunk_size
        See :func:`write_bed`.
    """
    prefix = str(prefix)
    n_samples, n_variants = X.shape
    if len(fam) != n_samples:
        raise ValueError(f"fam has {len(fam)} rows but X has {n_samples} samples")
    if len(bim) != n_variants:
        raise ValueError(f"bim has {len(bim)} rows but X has {n_variants} variants")
    write_bed(f"{prefix}.bed", X, variant_chunk_size=variant_chunk_size)
    write_fam(f"{prefix}.fam", fam)
    write_bim(f"{prefix}.bim", bim)
