import numpy as np
import pandas as pd

from cellink.tl.gwas import GWAS


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


def burden_test(G, Y, F, gene, annotation_cols, burden_agg_fct="sum", run_lrt=True):
    """
    Perform a burden test for genetic association analysis.

    Parameters
    ----------
    G : AnnData
        An AnnData object containing genotype data. The `X` attribute should
        contain the genotype matrix, and the `varm["variant_annotation"]`
        attribute should contain variant annotations.
    Y : array-like
        Phenotype data for the samples.
    F : array-like
        Covariate matrix for the samples.
    gene : str
        The name of the gene being tested.
    annotation_cols : list of str
        List of column names in `G.varm["variant_annotation"]` to use for
        calculating variant scores.
    burden_agg_fct : str, optional (default="sum")
        Aggregation function to compute the burden score. Options include
        "sum", "mean", "max".
    run_lrt : bool, optional (default=True)
        Whether to compute the likelihood ratio test (LRT) for the burden test.

    Returns
    -------
    rdf : pandas.DataFrame
        A DataFrame containing the results of the burden test with the
        following columns:
        - "burden_gene": The gene name whose burden was used.
        - "egene": The gene name that was tested (expression, Y).
        - "burden_type": The annotation columns used for the burden test.
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
    gwas.process(burdens)
    rdf = pd.DataFrame(
        {
            "burden_gene": gene,
            "egene": gene,
            "burden_type": annotation_cols,
            "burden_agg_fct": burden_agg_fct,
            "pv": list(gwas.getPv().reshape(-1)),
            "beta": list(gwas.getBetaSNP().reshape(-1)),
            "betaste": list(gwas.getBetaSNPste().reshape(-1)),
        }
    )
    if run_lrt:
        rdf["lrt"] = list(gwas.getLRT().reshape(-1))

    return rdf
