import warnings
from importlib.metadata import version

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
)

from . import io, pl, pp, tl  # noqa: E402

__all__ = ["pl", "pp", "tl", "io"]

__version__ = version("cellink")
