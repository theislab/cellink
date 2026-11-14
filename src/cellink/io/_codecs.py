from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.dtype import parse_data_type
from zarr.registry import register_codec

_UINT8 = parse_data_type("uint8", zarr_format=3)


@dataclass(frozen=True)
class Pack2Bit(ArrayArrayCodec):
    """Zarr v3 codec storing genotype calls (0-3) as 2 bits each, four per byte.

    Packs along the last (variant) axis, first call in the two highest bits.
    Chunks must span a multiple of 4 variants.
    """

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict) -> Pack2Bit:
        return cls()

    def to_dict(self) -> dict:
        return {"name": "cellink_pack2bit"}

    def validate(self, *, shape, dtype, chunk_grid) -> None:
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if chunk_shape is not None and chunk_shape[-1] % 4:
            raise ValueError(f"cellink_pack2bit needs a last-axis chunk size divisible by 4, got {chunk_shape[-1]}")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        *lead, m = chunk_spec.shape
        return ArraySpec(
            shape=(*lead, m // 4),
            dtype=_UINT8,
            fill_value=_UINT8.cast_scalar(0),
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    async def _encode_single(self, chunk_array, chunk_spec):
        x = chunk_array.as_numpy_array()
        if x.size and (x.min() < 0 or x.max() > 3):
            raise ValueError("cellink_pack2bit stores values 0-3 only")
        q = x.astype(np.uint8).reshape(*x.shape[:-1], -1, 4)
        packed = (q[..., 0] << 6) | (q[..., 1] << 4) | (q[..., 2] << 2) | q[..., 3]
        return chunk_spec.prototype.nd_buffer.from_numpy_array(packed)

    async def _decode_single(self, chunk_array, chunk_spec):
        p = chunk_array.as_numpy_array()
        out = np.empty((*p.shape, 4), dtype=np.int8)
        out[..., 0], out[..., 1], out[..., 2], out[..., 3] = p >> 6, (p >> 4) & 3, (p >> 2) & 3, p & 3
        return chunk_spec.prototype.nd_buffer.from_numpy_array(out.reshape(*p.shape[:-1], -1))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length // 4


register_codec("cellink_pack2bit", Pack2Bit)


def to_genotype_codes(x, missing_value: float | None = None) -> tuple[np.ndarray, int, int]:
    """Convert a genotype block to 0/1/2 int8 codes for 2-bit storage.

    - Integer input: values 0-2 are kept, negative values are missing, and
      values above 2 are rejected.
    - Float input: NaN and ``missing_value`` (e.g. PLINK's -9) are missing.
      Other values are rounded to the nearest integer, halves to even (0.5 to 0,
      1.5 to 2), and must lie in [-0.5, 2.5).
    - Missing values become 0.

    Returns the codes, the number of non-integer values that were rounded, and
    the number of missing values. Raises ValueError if any value does not round
    to 0, 1 or 2.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        missing = x < 0
        bad = x > 2
        r, n_frac = x, 0
    else:
        x = x.astype(np.float32, copy=False)
        missing = ~np.isfinite(x)
        if missing_value is not None:
            missing |= x == missing_value
        xv = np.where(missing, 0, x)
        bad = (xv < -0.5) | (xv >= 2.5)
        r = np.rint(xv)
        n_frac = int(np.count_nonzero(r != xv))
    if bad.any():
        raise ValueError("values outside the 0-2 genotype range cannot be stored with 2-bit encoding")
    return np.where(missing, 0, r).astype(np.int8), n_frac, int(np.count_nonzero(missing))
