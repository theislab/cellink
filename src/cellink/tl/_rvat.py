import numpy as np
import pandas as pd
from scipy.stats import beta

from cellink.at.gwas import GWAS
from cellink.at.skat import _skat_test


def _get_burden(G, var_scores, burden_agg_fct):
    if burden_agg_fct == "sum":
        burdens = np.dot(G, var_scores)
    elif burden_agg_fct == "mean":
        burdens = np.dot(G, var_scores) / G.shape[1]
    elif burden_agg_fct == "max":
        geno_view = G[:, :, np.newaxis]  # Shape: (samples, variants, 1)
        scores_view = var_scores[np.newaxis, :, :]  # Shape: (1, variants, annotations)
        # weight variants by their annotation scores
        weighted_vars = geno_view * scores_view  # (samples, variants, annotations)
        # max across weighted variants
        burdens = np.max(weighted_vars, axis=1)  # Shape: (samples, annotations)
    else:
        raise NotImplementedError(f"Aggregation function '{burden_agg_fct}' is not implemented.")

    return burdens


def beta_weighting(values, beta_weights=(1, 25)):
    """
    Apply beta weighting to a set of values using the Beta probability density function. This is the standard weighting of variants

    Parameters
    ----------
    values : np.ndarray
        The input values to be weighted. These should be in the range [0, 1].
    beta_weights: tuple
        A tuple of two positive numbers representing the alpha and beta parameters of the Beta distribution. Defaults to (1, 25).

    Returns
    -------
    np.ndarray: The weighted values computed using the Beta probability density function.
    """
    weighted = beta.pdf(values, beta_weights[0], beta_weights[1])

    return weighted


def run_burden_test(G, Y, F, gene, annotation_cols, burden_agg_fct="sum", run_lrt=True):
    """
    Perform a burden test for genetic association analysis.

    Parameters
    ----------
    G : AnnData
        An AnnData object containing genotype data. The `X` attribute should
        contain the genotype matrix, and the `varm["variant_annotation"]`
        attribute should contain variant annotations.
    Y : np.ndarray
        Phenotype data for the samples.
    F : np.ndarray
        Covariate matrix for the samples.
    gene : str
        The name of the gene being tested.
    annotation_cols : list of str
        List of column names in `G.varm["variant_annotation"]` to use for
        calculating variant scores.
    burden_agg_fct : str (default="sum")
        Aggregation function to compute the burden score. Options include
        "sum", "mean", "max".
    run_lrt : bool (default=True)
        Whether to compute the likelihood ratio test (LRT) for the burden test.

    Returns
    -------
    rdf : pandas.DataFrame
        A DataFrame containing the results of the burden test with the
        following columns:
        - "burden_gene": The gene name whose burden was used.
        - "egene": The gene name that was tested (expression, Y).
        - "weight_col": The annotation columns used for the burden test.
        - "burden_agg_fct": The aggregation function used.
        - "pv": P-values from the GWAS analysis.
        - "beta": Effect sizes from the GWAS analysis.
        - "betaste": Standard errors of the effect sizes.
        - "lrt" (if `run_lrt` is True): LRT statistics from the GWAS analysis.
    """
    gwas = GWAS(Y, F)  # move this outside of the function and provide gwas object as input?
    genotypes = G.X
    var_scores = G.varm["variant_annotation"][annotation_cols].to_numpy()

    burdens = _get_burden(genotypes, var_scores, burden_agg_fct)
    gwas.test_association(burdens)
    rdf = pd.DataFrame(
        {
            "burden_gene": gene,
            "egene": gene,
            "weight_col": annotation_cols,
            "burden_agg_fct": burden_agg_fct,
            "pv": list(gwas.getPv().ravel()),
            "beta": list(gwas.getBetaSNP().ravel()),
            "betaste": list(gwas.getBetaSNPste().ravel()),
            "lrt": list(gwas.getLRT().ravel()),
        }
    )

    return rdf


def run_skat_test(G, Y, F, gene):
    # TODO implement alternative weights once skat_test supports this
    skat = _skat_test(Y, G.X, F)
    rdict = {
        "burden_gene": gene,
        "egene": gene,
        "weight_col": "maf_beta",  # TODO change once alternative weighting is implemented
        "pv": skat,
    }
    return rdict
