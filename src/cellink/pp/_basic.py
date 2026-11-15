import anndata
import dask.array as da
import numpy as np

from cellink._core import DonorData


def _replace_levels(dd: DonorData, subset: DonorData) -> None:
    """Make `dd` hold the levels of `subset`, so an in-place filter reaches the caller's object."""
    if subset is not dd:
        dd._G, dd._C = subset._G, subset._C
        dd._var_dims_to_sync = subset._var_dims_to_sync


def cell_level_obs_filter(
    dd: DonorData,
    cell_level_key: str = None,
    cell_level_values: str | list | np.ndarray = None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """
    Filter DonorData based on cell level observations information.

    Parameters
    ----------
    inplace
        Subset `dd` in place and return None, or return the subset.
    copy
        With ``inplace=False``, return a copy instead of an object that shares data with `dd`.
        Ignored when ``inplace`` is True.

    Returns
    -------
    None if `inplace` is True (then `dd` is subset), otherwise the subset DonorData.

    Example
    -------
    >>> celltype_key = "celltype"
    >>> cell_type = "CD8"
    >>> cl.pp.cell_level_obs_filter(dd, celltype_key, cell_type, inplace=False, copy=True)
    """
    subset = dd
    if isinstance(cell_level_values, str):
        subset = dd.sel(C_obs=dd.C.obs[cell_level_key] == cell_level_values)
    elif isinstance(cell_level_values, list):
        subset = dd.sel(C_obs=dd.C.obs[cell_level_key].isin(cell_level_values))
    elif isinstance(cell_level_values, np.ndarray):
        subset = dd.sel(C_obs=np.isin(dd.C.obs[cell_level_key].values, cell_level_values))

    if inplace:
        _replace_levels(dd, subset)
        return None
    return subset.copy() if copy else subset


def donor_level_obs_filter(
    dd: DonorData,
    donor_level_key: str = None,
    donor_level_values: str | list | np.ndarray = None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """
    Filter DonorData based on donor level observations information.

    Parameters
    ----------
    inplace
        Subset `dd` in place and return None, or return the subset.
    copy
        With ``inplace=False``, return a copy instead of an object that shares data with `dd`.
        Ignored when ``inplace`` is True.

    Returns
    -------
    None if `inplace` is True (then `dd` is subset), otherwise the subset DonorData.

    Example
    -------
    >>> donor_key = "disease_state"
    >>> disease_key = "COVID-19"
    >>> cl.pp.cell_level_obs_filter(dd, donor_key, disease_key, inplace=False, copy=True)
    """
    subset = dd
    if isinstance(donor_level_values, str):
        subset = dd.sel(G_obs=dd.G.obs[donor_level_key] == donor_level_values)
    elif isinstance(donor_level_values, list):
        subset = dd.sel(G_obs=dd.G.obs[donor_level_key].isin(donor_level_values))
    elif isinstance(donor_level_values, np.ndarray):
        subset = dd.sel(G_obs=np.isin(dd.G.obs[donor_level_key].values, donor_level_values))

    if inplace:
        _replace_levels(dd, subset)
        return None
    return subset.copy() if copy else subset


def donor_level_var_filter(
    dd: DonorData,
    *,
    donor_level_key: str = None,
    donor_level_values: str | list | np.ndarray = None,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """
    Filter DonorData based on cell level variables information.

    Parameters
    ----------
    inplace
        Subset `dd` in place and return None, or return the subset.
    copy
        With ``inplace=False``, return a copy instead of an object that shares data with `dd`.
        Ignored when ``inplace`` is True.

    Returns
    -------
    None if `inplace` is True (then `dd` is subset), otherwise the subset DonorData.

    Example
    -------
    >>> chromosome_key = "chromosome"
    >>> chromosome_value = 2
    >>> cl.pp.cell_level_obs_filter(dd, chromosome_key, chromosome_value, inplace=False, copy=True)
    """
    subset = dd
    if isinstance(donor_level_values, str):
        subset = dd.sel(G_var=dd.G.var[donor_level_key] == donor_level_values)
    elif isinstance(donor_level_values, list):
        subset = dd.sel(G_var=dd.G.var[donor_level_key].isin(donor_level_values))
    elif isinstance(donor_level_values, np.ndarray):
        subset = dd.sel(G_var=np.isin(dd.G.var[donor_level_key].values, donor_level_values))

    if inplace:
        _replace_levels(dd, subset)
        return None
    return subset.copy() if copy else subset


def low_abundance_filter(
    adata: anndata.AnnData,
    *,
    abundance_threshold: float = 1e-6,
    method: str = "mean",
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """
    Filter out low-abundance metabolites.

    This function removes metabolites (features) with abundance below a specified threshold.
    Users can choose whether to filter based on the mean or median abundance across samples.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` x `n_vars`, where rows correspond to samples
        and columns to metabolites.
    abundance_threshold
        The minimum abundance level to retain a metabolite. Features with abundance below this
        threshold will be filtered out.
    method
        Method to calculate abundance: "mean" or "median".
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        With ``inplace=False``, return a copy instead of a view of `adata`. Ignored when
        ``inplace`` is True.

    Returns
    -------
    Returns the AnnData object with filtered metabolites or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, da.Array | np.ndarray):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    if method == "mean":
        abundance = da.mean(X, axis=0) if isinstance(X, da.Array) else np.mean(X, axis=0)
    elif method == "median":
        abundance = da.median(X, axis=0) if isinstance(X, da.Array) else np.median(X, axis=0)
    else:
        raise NotImplementedError("Method must be either 'mean' or 'median'.")

    abundance_filter = abundance >= abundance_threshold
    abundance_filter = abundance_filter.compute() if isinstance(abundance_filter, da.Array) else abundance_filter

    if inplace:
        adata._inplace_subset_var(abundance_filter)
        return None
    subset = adata[:, abundance_filter]
    return subset.copy() if copy else subset


def missing_values_filter(
    adata: anndata.AnnData,
    *,
    max_missing_ratio: float = 0.2,
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """Remove features that have a proportion of missing values greater than the specified threshold.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` x `n_vars`.
    max_missing_ratio
        Maximum allowable proportion of missing values for a protein to be retained.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        With ``inplace=False``, return a copy instead of a view of `adata`. Ignored when
        ``inplace`` is True.

    Returns
    -------
    Returns the AnnData object with filtered values or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, da.Array | np.ndarray):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    is_missing = da.isnan(X) if isinstance(X, da.Array) else np.isnan(X)
    missing_ratio = da.mean(is_missing, axis=0) if isinstance(X, da.Array) else np.mean(is_missing, axis=0)

    valid_features = missing_ratio <= max_missing_ratio
    valid_features = valid_features.compute() if isinstance(valid_features, da.Array) else valid_features

    if inplace:
        adata._inplace_subset_var(valid_features)
        return None
    subset = adata[:, valid_features]
    return subset.copy() if copy else subset


def log_transform(
    adata: anndata.AnnData,
    *,
    base: float = 2,
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """
    Apply log transformation to protein abundance values.

    This function applies a log transformation to the abundance values,
    which can help stabilize variance and reduce skewness in the data.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` x `n_vars`.
    base
        The base of the logarithm (e.g., 2 or 10).
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        With ``inplace=False``, write the result into a copy of `adata` and leave `adata`
        unchanged. Ignored when ``inplace`` is True.

    Returns
    -------
    Returns the AnnData object with log-transformed data or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, da.Array | np.ndarray):
        raise ValueError("adata.X must be a Dask or NumPy array.")

    log_base = np.log(base)
    X_log = da.log1p(X) / log_base if isinstance(X, da.Array) else np.log1p(X) / log_base

    if inplace:
        adata.X = X_log
        return None
    adata_out = adata.copy() if copy else adata
    adata_out.X = X_log
    return adata_out


def normalize(
    adata: anndata.AnnData,
    *,
    method: str = "zscore",
    inplace: bool = True,
    copy: bool = True,
) -> anndata.AnnData | None:
    """Normalize values using the specified method.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` x `n_vars`.
    method
        The normalization method. Options are:
        - 'zscore': z-score normalization (mean=0, std=1 for each protein).
        - 'minmax': Scale each protein to the range [0, 1].
        - 'median': Subtract the median of each protein.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        With ``inplace=False``, write the result into a copy of `adata` and leave `adata`
        unchanged. Ignored when ``inplace`` is True.

    Returns
    -------
    Returns the AnnData object with normalized protein abundances or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, da.Array | np.ndarray):
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

    if inplace:
        adata.X = X_norm
        return None
    adata_out = adata.copy() if copy else adata
    adata_out.X = X_norm
    return adata_out
