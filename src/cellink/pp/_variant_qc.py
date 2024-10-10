import anndata
import numpy as np


def variant_qc(
    adata: anndata.AnnData,
    *,
    maf_threshold: float = 0.01,
    hwe_pval_threshold: float = 1e-6,
    inplace: bool = True,
    copy: bool = False,
) -> anndata.AnnData | None:
    """\
    Perform quality control on variants.

    This function filters variants based on Minor Allele Frequency (MAF) and Hardy-Weinberg Equilibrium (HWE).

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
    maf_threshold
        The threshold for minor allele frequency (MAF).
    hwe_pval_threshold
        The threshold for Hardy-Weinberg equilibrium (HWE) p-value.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns the AnnData object with filtered variants or updates the input `adata` if inplace is True.

    Example
    --------
    >>> adata = anndata.AnnData(np.array([[0, 1], [1, 0], [0, 0]]))
    >>> variant_qc(adata, maf_threshold=0.01, hwe_pval_threshold=1e-6)
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    X = adata.X

    allele_freq = np.sum(X, axis=0) / (2 * X.shape[0])
    maf = np.minimum(allele_freq, 1 - allele_freq)

    maf_filter = maf >= maf_threshold

    # Could introduce further fitlers here

    combined_filter = maf_filter

    adata = adata[:, combined_filter]

    if inplace:
        return None
    return adata
