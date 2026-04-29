import importlib
from typing import Any

from ._export import to_plink, write_variants_to_vcf
from ._readwrite import read_dd, read_h5_dd, read_zarr_dd
from ._sgkit import from_sgkit_dataset, read_bgen, read_plink, read_sgkit_zarr
from ._pgen import stream_pgen_to_zarr, read_pgen_zarr

# Lazy re-exports for the optional `annbatch` extra. We don't want importing
# `cellink.io` to fail when annbatch isn't installed -- only the relevant
# attribute access should error, with a clear hint pointing at the extra.
_annbatch_exports = {
    "write_annbatch_collection": "_annbatch",
    "open_annbatch_loader": "_annbatch",
}

__all__ = [
    "to_plink",
    "write_variants_to_vcf",
    "read_dd",
    "read_h5_dd",
    "read_zarr_dd",
    "from_sgkit_dataset",
    "read_bgen",
    "read_plink",
    "read_sgkit_zarr",
    "stream_pgen_to_zarr",
    "read_pgen_zarr",
    *_annbatch_exports,
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional-extra symbols in `cellink.io`.

    Currently used for the `annbatch` extra: we only attempt the import when
    `cellink.io.write_annbatch_collection` (or sibling) is actually accessed,
    and surface a clear ImportError pointing at the extra otherwise.
    """
    if name in _annbatch_exports:
        module_name = _annbatch_exports[name]
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ImportError as e:
            raise ImportError(
                f"Cannot import `{name}` from `cellink.io.{module_name}`: "
                "this feature requires the `annbatch` extra. Install with:\n\n"
                "    pip install cellink[annbatch]"
            ) from e
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
