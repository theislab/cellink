import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")

from importlib.metadata import version

from . import io, pl, pp, tl
from ._core.donordata import DonorData

__all__ = ["DonorData", "pl", "pp", "tl", "io"]

__version__ = version("cellink")
