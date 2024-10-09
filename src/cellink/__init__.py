import warnings
from importlib.metadata import version

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
)

__all__ = ["DonorData", "pl", "pp", "tl", "io"]

from . import io, pl, pp, tl  # noqa: E402

__version__ = version("cellink")

from ._core.donordata import DonorData
