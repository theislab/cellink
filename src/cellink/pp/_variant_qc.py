import anndata
import dask.array as da
import numpy as np
from scipy import stats


def _hwe_chisq_pvalue(counts: np.ndarray) -> np.ndarray:
    """Hardy-Weinberg equilibrium p-value per variant from genotype counts.

    Parameters
    ----------
    counts
        Array of shape ``(3, n_vars)`` with the number of donors carrying 0, 1 and 2 alleles.

    Returns
    -------
    np.ndarray
        The 1-df chi-square p-value per variant; 1 for monomorphic or uncalled variants.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (2 * counts[0] + counts[1]) / (2 * n)
        expected = np.stack([n * p**2, 2 * n * p * (1 - p), n * (1 - p) ** 2])
        terms = np.where(expected > 0, (counts - expected) ** 2 / expected, 0.0)
    return stats.chi2.sf(terms.sum(axis=0), df=1)


def variant_qc(
    adata: anndata.AnnData,
    *,
    maf_threshold: float = 0.01,
    hwe_pval_threshold: float = 1e-6,
    inplace: bool = True,
    copy: bool = False,
) -> anndata.AnnData | None:
    """Perform quality control on variants.

    This function filters variants based on Minor Allele Frequency (MAF) and Hardy-Weinberg Equilibrium (HWE).
    The MAF is computed over the called genotypes of each variant (missing ``nan`` calls are skipped).
    The HWE p-value comes from a 1-df chi-square goodness-of-fit test on the counts of hard calls
    (dosages of exactly 0, 1 and 2; missing and non-integer dosages are not counted). This is not
    PLINK's exact test (``--hwe``) and is less reliable for rare variants with few expected
    minor-allele homozygotes.

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
    maf_threshold
        The threshold for minor allele frequency (MAF).
    hwe_pval_threshold
        Variants with a Hardy-Weinberg equilibrium (HWE) p-value below this threshold are removed.
    inplace
        Whether to update `adata` or return a copy of the new AnnData object.
    copy
        With ``inplace=False``, return a copy instead of a view of `adata`. Ignored when
        ``inplace`` is True.

    Returns
    -------
    Returns the AnnData object with filtered variants or updates the input `adata` if inplace is True.
    """
    X = adata.X

    if not isinstance(X, da.Array):
        raise ValueError("adata.X must be a Dask array.")

    n_called = da.sum(~da.isnan(X), axis=0)
    allele_freq = da.nansum(X, axis=0) / (2 * da.maximum(n_called, 1))
    maf = da.minimum(allele_freq, 1 - allele_freq)

    maf_filter = maf >= maf_threshold

    genotype_counts = da.stack([(X == g).sum(axis=0) for g in (0, 1, 2)])
    maf_filter, genotype_counts = da.compute(maf_filter, genotype_counts)
    hwe_filter = _hwe_chisq_pvalue(genotype_counts) >= hwe_pval_threshold

    combined_filter = np.asarray(maf_filter) & hwe_filter

    if inplace:
        adata._inplace_subset_var(combined_filter)
        return None
    subset = adata[:, combined_filter]
    return subset.copy() if copy else subset
