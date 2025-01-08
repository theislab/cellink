import anndata
import dask.array as da


def low_abundance_filter(
    adata: anndata.AnnData,
    *,
    abundance_threshold: float = 1e-6,
    method: str = "mean",
    inplace: bool = True,
    copy: bool = False,
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

    if method not in {"mean", "median"}:
        raise ValueError("Method must be either 'mean' or 'median'.")

    if method == "mean":
        abundance = da.mean(X, axis=0)
    else:  # method == "median"
        abundance = da.median(X, axis=0)

    # Apply the abundance filter
    abundance_filter = abundance >= abundance_threshold
    abundance_filter = abundance_filter.compute()

    adata = adata[:, abundance_filter]

    if inplace:
        return None
    return adata


def missing_values_filter(
    adata: anndata.AnnData,
    *,
    max_missing_ratio: float = 0.2,
    inplace: bool = True,
    copy: bool = False,
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
    missing_ratio = da.mean(is_missing, axis=0)

    valid_proteins = (missing_ratio <= max_missing_ratio).compute()
    adata = adata[:, valid_proteins]

    if inplace:
        return None
    return adata


def log_transform(
    adata: anndata.AnnData,
    *,
    base: float = 2,
    inplace: bool = True,
    copy: bool = False,
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

    if inplace:
        adata.X = X_log
        return None
    return adata.copy() if copy else adata


def normalize(
    adata: anndata.AnnData,
    *,
    method: str = "zscore",
    inplace: bool = True,
    copy: bool = False,
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
        mean = da.mean(X, axis=0)
        std = da.std(X, axis=0)
        X_norm = (X - mean) / std
    elif method == "minmax":
        min_val = da.min(X, axis=0)
        max_val = da.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val)
    elif method == "median":
        median = da.median(X, axis=0)
        X_norm = X - median
    else:
        raise ValueError("Invalid method. Choose from 'zscore', 'minmax', or 'median'.")

    if inplace:
        adata.X = X_norm
        return None
    return adata.copy() if copy else adata
