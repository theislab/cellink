import importlib
from typing import Any

from ._jaxqtl import read_jaxqtl_results, run_jaxqtl
from ._ld import calculate_ld
from ._pc import calculate_pcs
from ._tensorqtl import read_tensorqtl_results, run_tensorqtl

__all__ = [
    "read_jaxqtl_results",
    "run_jaxqtl",
    "calculate_ld",
    "calculate_pcs",
    "read_tensorqtl_results",
    "run_tensorqtl",
    "run_mixmil",
]


def __getattr__(name: str) -> Any:
    if name == "run_mixmil":
        try:
            module = importlib.import_module(f"{__name__}._mixmil")
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                "Cannot import `run_mixmil`: this feature requires `torch` and `mixmil`. "
                "Install with:\n\n    pip install cellink[mixmil]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
