import warnings
from pathlib import Path

import h5py
from anndata._io.specs.registry import read_elem
from mudata import MuData

from .._core import DonorData

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


def read_donordata_object(
    path: str | Path,
) -> DonorData:
    """Read donor data from specified file paths for both the gene expression data (G) and cell-type data (C),
    and return a DonorData object containing these two datasets.

    Parameters
    ----------
    path : str | Path
        Path to the DonorData data file.

    Returns
    -------
    DonorData
        A DonorData object containing:
        - `G`: Gene expression data as an `AnnData` object or a `mu.Dataset` object.
        - `C`: Cell-type data as an `AnnData` object.

    Example
    -------
    dd = read_donordata_object('path/to/donor_data.dd.h5')
    """
    with h5py.File(path, "r") as f:
        G = read_elem(f["G"])
        C = read_elem(f["C"])
        if len(G.keys()) > 1:
            obs = G["obs"]["obs"]
            uns = G["uns"]["uns"]
            G = MuData({key: G[key][key] for key in G.keys() if key not in ["obs", "uns"]})
            G.obs = obs
            G.uns = uns
        else:
            G = G["G"]
        if len(C.keys()) > 1:
            obs = C["obs"]["obs"]
            uns = C["uns"]["uns"]
            C = MuData({key: C[key][key] for key in C.keys() if key not in ["obs", "uns"]})
            C.obs = obs
            C.uns = uns
        else:
            C = C["C"]

        donor_id = f.attrs.get("donor_id", "donor")
        var_dims_to_sync = list(f.attrs.get("var_dims_to_sync", []))

        uns = {}
        uns_group = f.get("uns")
        if uns_group:
            for key in uns_group:
                uns[key] = uns_group[key][()]

    dd = DonorData(G=G, C=C, donor_id=donor_id, var_dims_to_sync=var_dims_to_sync, uns=uns).copy()

    return dd
