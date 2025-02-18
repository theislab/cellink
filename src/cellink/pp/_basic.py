import anndata
import dask.array as da
import numpy as np
from .._core import DonorData

def cell_level_obs_filter(
    dd: DonorData,
    cell_level_key: str = None,
    cell_level_values: str | list | np.ndarray = None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """\
    Filter DonorData based on cell level observations information.

    Parameters
    ----------
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix.
    """

    if type(cell_level_values) == str:
        dd = dd[..., dd.C.obs[cell_level_key] == cell_level_values, :].copy()
    elif type(cell_level_values) == list:
        dd = dd[..., dd.C.obs[cell_level_key].isin(cell_level_values), :].copy()
    elif type(cell_level_values) == np.ndarray:
        dd = dd[..., np.isin(dd.C.obs[cell_level_key].values, cell_level_values), :].copy()
    
    if inplace:
        return None
    return dd.copy() if copy else dd

def donor_level_obs_filter(
    dd: DonorData,
    donor_level_key: str = None,
    donor_level_values: str | list | np.ndarray = None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """\
    Filter DonorData based on donor level observations information.

    Parameters
    ----------
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix.
    """

    if type(donor_level_values) == str:
        dd = dd[dd.G.obs[donor_level_key] == donor_level_values, :].copy()
    elif type(donor_level_values) == list:
        dd = dd[dd.G.obs[donor_level_key].isin(donor_level_values), :].copy()
    elif type(donor_level_values) == np.ndarray:
        dd = dd[np.isin(dd.G.obs[donor_level_key].values, donor_level_values), :].copy()
    
    if inplace:
        return None
    return dd.copy() if copy else dd

def donor_level_var_filter(
    dd: DonorData,
    *,
    donor_level_key: str = None,
    donor_level_values: str | list | np.ndarray = None,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """\
    Filter DonorData based on cell level variables information.

    Parameters
    ----------
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix.
    """

    if type(donor_level_values) == str:
        dd = dd[:, dd.G.var[donor_level_key] == donor_level_values, :].copy()
    elif type(donor_level_values) == list:
        dd = dd[:, dd.C.var[donor_level_key].isin(donor_level_values), :].copy()
    elif type(donor_level_values) == np.ndarray:
        dd = dd[:, np.isin(dd.C.var[donor_level_key].values, donor_level_values), :].copy()
    
    if inplace:
        return None
    return dd.copy() if copy else dd

def low_abundance_filter(
    adata: anndata.AnnData,
    *,
    abundance_threshold: float = 1e-6,
    method: str = "mean",
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """\
    Filter out low-abundance metabolites.

    This function removes metabolites (features) with abundance below a specified threshold.
    Users can choose whether to filter based on the mean or median abundance across samples.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`, where rows correspond to samples
        and columns to metabolites.
    abundance_threshold
        The minimum abundance level to retain a metabolite. Features with abundance below this
        threshold will be filtered out.
    method
        Method to calculate abundance: "mean" or "median".
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        Whether to modify a copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns the AnnData object with filtered metabolites or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, (da.Array, np.ndarray)):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    if method == "mean":
        abundance = da.mean(X, axis=0) if isinstance(X, da.Array) else np.mean(X, axis=0)
    elif method == "median":
        abundance = da.median(X, axis=0) if isinstance(X, da.Array) else np.median(X, axis=0)
    else:
        raise NotImplementedError("Method must be either 'mean' or 'median'.")

    abundance_filter = abundance >= abundance_threshold
    abundance_filter = abundance_filter.compute() if isinstance(abundance_filter, da.Array) else abundance_filter

    adata = adata[:, abundance_filter].copy()

    if inplace:
        adata.X = adata.X
        return None
    return adata.copy() if copy else adata

def missing_values_filter(
    adata: anndata.AnnData,
    *,
    max_missing_ratio: float = 0.2,
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """\
    This function removes features that have a proportion of missing values
    greater than the specified threshold.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
    max_missing_ratio
        Maximum allowable proportion of missing values for a protein to be retained.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns the AnnData object with filtered values or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, (da.Array, np.ndarray)):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    is_missing = da.isnan(X) if isinstance(X, da.Array) else np.isnan(X)
    missing_ratio = da.mean(is_missing, axis=0) if isinstance(X, da.Array) else np.mean(is_missing, axis=0)

    valid_features = (missing_ratio <= max_missing_ratio)
    valid_features = valid_features.compute() if isinstance(valid_features, da.Array) else valid_features

    adata = adata[:, valid_features]

    if inplace:
        adata.X = adata.X
        return None
    return adata.copy() if copy else adata

def log_transform(
    adata: anndata.AnnData,
    *,
    base: float = 2,
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """\
    Apply log transformation to protein abundance values.

    This function applies a log transformation to the abundance values,
    which can help stabilize variance and reduce skewness in the data.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
    base
        The base of the logarithm (e.g., 2 or 10).
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns the AnnData object with log-transformed data or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, (da.Array, np.ndarray)):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    log_base = np.log(base)
    X_log = da.log1p(X) / log_base if isinstance(X, da.Array) else np.log1p(X) / log_base

    adata.X = X_log

    if inplace:
        return None
    return adata.copy() if copy else adata


def normalize(
    adata: anndata.AnnData,
    *,
    method: str = "zscore",
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """\
    This function normalizes values using the specified method.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
    method
        The normalization method. Options are:
        - 'zscore': z-score normalization (mean=0, std=1 for each protein).
        - 'minmax': Scale each protein to the range [0, 1].
        - 'median': Subtract the median of each protein.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns the AnnData object with normalized protein abundances or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, (da.Array, np.ndarray)):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    if method == "zscore":
        mean = da.mean(X, axis=0) if isinstance(X, da.Array) else np.mean(X, axis=0)
        std = da.std(X, axis=0) if isinstance(X, da.Array) else np.std(X, axis=0)
        X_norm = (X - mean) / std
    elif method == "minmax":
        min_val = da.min(X, axis=0) if isinstance(X, da.Array) else np.min(X, axis=0)
        max_val = da.max(X, axis=0) if isinstance(X, da.Array) else np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val)
    elif method == "median":
        median = da.median(X, axis=0) if isinstance(X, da.Array) else np.median(X, axis=0)
        X_norm = X - median
    else:
        raise ValueError("Invalid method. Choose from 'zscore', 'minmax', or 'median'.")

    adata.X = X_norm

    if inplace:
        return None
    return adata.copy() if copy else adata
