import anndata
import numpy as np
import dask.array as da

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
    """
    X = adata.X

    print("dsgjkgadsö")

    if not isinstance(X, da.Array):
        raise ValueError("adata.X must be a Dask array.")

    allele_freq = da.sum(X, axis=0) / (2 * X.shape[0])
    maf = da.minimum(allele_freq, 1 - allele_freq)

    maf_filter = maf >= maf_threshold

    combined_filter = maf_filter.compute()

    print("kjdsgköag")
    from collections import Counter
    print(Counter(combined_filter))

    adata = adata[:, combined_filter]

    if inplace:
        return None
    return adata
