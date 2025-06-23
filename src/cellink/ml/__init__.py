import importlib
from typing import Any

__all__ = ["MILDataset", "mil_collate_fn", "DonorMILModel"]

# map public names to their source submodule
_submodules = {
    "MILDataset": "dataset",
    "mil_collate_fn": "dataset",
    "DonorMILModel": "model",
}


def __getattr__(name: str) -> Any:
    """Lazy import from cellink.ml

    Lazily import `cellink.ml.<submodule>` only when you do
    `cellink.ml.<name>`.  If torch/lightning aren’t installed,
    this raises a clear ImportError pointing at the [ml] extra.
    """
    if name in _submodules:
        module_name = _submodules[name]
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import `{name}` from `cellink.ml.{module_name}`: "
                "this feature requires `torch` and `pytorch-lightning`. "
                "Install with:\n\n    pip install cellink[ml]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # so that `dir(cellink.ml)` lists our public API
    return __all__
