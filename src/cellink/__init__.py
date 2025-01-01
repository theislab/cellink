import warnings
from importlib.metadata import version

from . import io, pl, pp, tl
from ._core.donordata import DonorData
from ._core.donordata_dataloader import DonorDataModel, DonorDataBaseModel

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
)


__all__ = ["DonorData", "DonorDataModel", "DonorDataBaseModel", "pl", "pp", "tl", "io"]

__version__ = version("cellink")
