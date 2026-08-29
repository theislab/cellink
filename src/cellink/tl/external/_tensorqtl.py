import glob
import gzip
import importlib.util
import logging
import os
import pickle
import shutil
import subprocess
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata.utils import asarray

from cellink._core import DonorData
from cellink.io import to_plink

logger = logging.getLogger(__name__)


def read_tensorqtl_results(
    prefix: str = None,
    mode: str = None,
    cis_output: bool | str = None,
    interaction_df: bool | str = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame]:
    """
    Read TensorQTL result files.

    Parameters
    ----------
    prefix : str, optional
        File prefix used for generating intermediate input/output files. Required for most modes.
    mode : str
        Mode of TensorQTL run (e.g., "cis_nominal", "cis", "trans", "cis_susie").
    cis_output : str or bool, default=True
        - If a string, specifies the path to the cis_output file for modes like `cis_independent` or `cis_susie`.
        - If True, the function will automatically attempt to find the cis_output file in `file_paths`.
        - If False, the cis_output will not be read.
    interaction_df : str or bool, default=True
        - If a string, specifies the path to the interaction terms file for `cis_nominal` mode.
        - If True, the function will attempt to automatically find the interaction file in `file_paths`.
        - If False, the interaction file will not be read.

    Returns
    -------
    pd.DataFrame or tuple
        Parsed results depending on the mode.
    """
    if mode == "cis_nominal":
        cis_qtl_pairs = pd.concat(
            [pd.read_parquet(path) for path in glob.glob(f"{prefix}.cis_qtl_pairs.*.parquet")], axis=0
        )
        if cis_output is not None:
            cis_qtl_signif_pairs = pd.read_parquet(f"{prefix}.cis_qtl.signif_pairs.parquet")
        else:
            cis_qtl_signif_pairs = None
        if interaction_df is not None:
            cis_qtl_top_assoc = pd.read_csv(f"{prefix}.cis_qtl_top_assoc.txt.gz", sep="\t")
        else:
            cis_qtl_top_assoc = None
        results = (cis_qtl_pairs, cis_qtl_signif_pairs, cis_qtl_top_assoc)
    elif mode == "cis":
        results = pd.read_csv(f"{prefix}.cis_qtl.txt.gz", sep="\t")
    elif mode == "cis_independent":
        results = pd.read_csv(f"{prefix}.cis_independent_qtl.txt.gz", sep="\t")
    elif mode == "trans":
        results = pd.read_parquet(f"{prefix}.trans_qtl_pairs.parquet")
    elif mode == "cis_susie" or mode == "trans_susie":
        with open(f"{prefix}.SuSiE.pickle", "rb") as f:
            susie = pickle.load(f)
        susie_summary = pd.read_parquet(f"{prefix}.SuSiE_summary.parquet")
        results = (susie, susie_summary)
    return results


def build_known_cis_eqtls_from_tensorqtl(
    tensorqtl_parquet_path: str,
    gene_names: list[str],
    max_snps_per_gene: int = 1,
    pval_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Build a ``known_cis_eqtls`` annotation (variants x genes, binary) from a
    completed TensorQTL nominal cis-scan, for use as a fine-mapping prior.

    Parameters
    ----------
    tensorqtl_parquet_path : str
        Path to a TensorQTL nominal cis-scan parquet file with ``gene``,
        ``variant_id``, and ``pval`` columns.
    gene_names : list[str]
        Genes to build the annotation for.
    max_snps_per_gene : int, default=1
        Number of lowest-p-value variant(s) to select per gene.
    pval_threshold : float, optional
        If given, only variants with ``pval <= pval_threshold`` are eligible.

    Returns
    -------
    pd.DataFrame
        Binary variant x gene matrix, 1 where that variant is the selected
        cis-eQTL for that gene.
    """
    df = pd.read_parquet(tensorqtl_parquet_path, columns=["gene", "variant_id", "pval"])
    df = df[df["gene"].isin(set(gene_names))]
    if pval_threshold is not None:
        df = df[df["pval"] <= pval_threshold]
    df = df.sort_values("pval").groupby("gene", sort=False).head(max_snps_per_gene)
    if df.empty:
        raise ValueError(
            f"No cis-eQTL pairs survived filtering from {tensorqtl_parquet_path} "
            f"(pval_threshold={pval_threshold}); cannot build known_cis_eqtls."
        )
    snps = df["variant_id"].unique()
    genes_with_hits = df["gene"].unique()
    known = pd.DataFrame(0, index=snps, columns=genes_with_hits, dtype=int)
    known.values[
        pd.Index(snps).get_indexer(df["variant_id"]),
        pd.Index(genes_with_hits).get_indexer(df["gene"]),
    ] = 1
    return known


def _map_susie_with_prior_weights(
    genotype_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    phenotype_pos_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    prior_weights: dict[str, pd.Series],
    L: int = 10,
    window: int = 1000000,
    max_iter: int = 500,
    maf_threshold: float = 0,
    scaled_prior_variance: float = 0.2,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    estimate_residual_variance: bool = True,
    estimate_prior_variance: bool = True,
    tol: float = 1e-3,
) -> tuple[pd.DataFrame, dict]:
    """SuSiE fine-mapping with a per-variant prior weight instead of SuSiE's
    default uniform prior over variants in the cis-window. Uses the same
    per-phenotype data preparation as ``tensorqtl.susie.map()`` (monomorphic
    + MAF filtering, covariate residualization); the one real difference
    is that ``susie.susie()`` is called with ``prior_weights`` supplied
    (``tensorqtl.susie.map()`` itself never exposes this, see the call site
    in ``run_tensorqtl``'s ``cis_susie`` branch for why).

    Parameters
    ----------
    prior_weights : dict[str, pd.Series]
        Keyed by phenotype_id; each value is a Series of raw (unnormalized)
        per-variant weights indexed by variant_id, covering (at least) that
        phenotype's own cis-window. Variants in the window missing from the
        Series are given the Series' own median as a neutral (not zero)
        weight. A hard zero would make them structurally unselectable
        regardless of signal, a stronger claim than a missing score
        justifies. Weights are renormalized to sum to 1 automatically.

    Returns
    -------
    Same shape as ``tensorqtl.susie.map(..., summary_only=False)``:
    ``(susie_summary_df, {phenotype_id: {pip, sets, converged, elbo, niter, lbf_variable}})``.
    """
    import torch
    from tensorqtl import genotypeio
    from tensorqtl.core import Residualizer, calculate_maf, impute_mean
    from tensorqtl.susie import susie as susie_fit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError("No valid phenotypes found.")

    susie_summary = []
    susie_res = {}
    for phenotype, genotypes, genotype_range, phenotype_id in igc.generate_data(verbose=True):
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)
        variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1] + 1].rename("variant_id")

        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t &= maf_t >= maf_threshold
        if mask_t.any():
            genotypes_t = genotypes_t[mask_t]
            mask = mask_t.cpu().numpy().astype(bool)
            variant_ids = variant_ids[mask]
            genotype_range = genotype_range[mask]
        if genotypes_t.shape[0] == 0:
            logger.warning(f"skipping {phenotype_id} (no valid variants)")
            continue

        if phenotype_id not in prior_weights:
            raise KeyError(
                f"prior_weights has no entry for phenotype_id={phenotype_id!r}; "
                f"provide a Series for every phenotype passed through cis_output, or omit it from cis_output."
            )
        pw = prior_weights[phenotype_id]
        aligned = pw.reindex(variant_ids)
        n_scored = aligned.notna().sum()
        fallback = pw.median()
        aligned = aligned.fillna(fallback)
        logger.info(
            f"{phenotype_id}: {genotypes_t.shape[0]} variants survive monomorphic+MAF filter "
            f"({genotypes_t.shape[1]} samples); {n_scored} of those have a real prior weight, "
            f"remaining {len(variant_ids) - n_scored} given the pool median ({fallback:.4g}) as a neutral prior"
        )
        prior_weights_t = torch.tensor(aligned.values / aligned.values.sum(), dtype=torch.float32).to(device)

        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
        genotypes_res_t = residualizer.transform(genotypes_t)
        phenotype_res_t = residualizer.transform(phenotype_t.reshape(1, -1))

        res = susie_fit(
            genotypes_res_t.T, phenotype_res_t.T, L=L, scaled_prior_variance=scaled_prior_variance,
            prior_weights=prior_weights_t, coverage=coverage, min_abs_corr=min_abs_corr,
            estimate_residual_variance=estimate_residual_variance, estimate_prior_variance=estimate_prior_variance,
            tol=tol, max_iter=max_iter,
        )

        af_t = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
        res["pip"] = pd.DataFrame({"pip": res["pip"], "af": af_t.cpu().numpy()}, index=variant_ids)
        logger.info(f"{phenotype_id}: converged={res['converged']}, niter={res['niter']}, "
                    f"elbo={res['elbo'][-1] if len(res['elbo']) else None}, "
                    f"max_pip={res['pip']['pip'].max():.6f}, sets['cs'] is None={res['sets']['cs'] is None}")
        if res["sets"]["cs"] is not None:
            if res["converged"]:
                for c in sorted(res["sets"]["cs"], key=lambda x: int(x.replace("L", ""))):
                    cs = res["sets"]["cs"][c]
                    p = res["pip"].iloc[cs].copy().reset_index()
                    p["cs_id"] = c.replace("L", "")
                    p.insert(0, "phenotype_id", phenotype_id)
                    susie_summary.append(p)
                res["lbf_variable"] = res["lbf_variable"][res["sets"]["cs_index"]]
        copy_keys = ["pip", "sets", "converged", "elbo", "niter", "lbf_variable"]
        susie_res[phenotype_id] = {k: res[k] for k in copy_keys}

    susie_summary_df = (
        pd.concat(susie_summary, axis=0).rename(columns={"snp": "variant_id"}).reset_index(drop=True)
        if susie_summary else pd.DataFrame()
    )
    drop_ids = [k for k in susie_res if susie_res[k]["sets"]["cs"] is None]
    for k in drop_ids:
        del susie_res[k]
    return susie_summary_df, susie_res


def _run_tensorqtl_python_api(
    mode: str,
    phenotype_df: pd.DataFrame,
    phenotype_pos_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    genotype_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    prefix: str,
    permutations: int,
    cis_output: str,
    interaction_df: str,
    susie_loci: str,
    window: int,
    pval_threshold: float,
    maf_threshold: float,
    maf_threshold_interaction: float,
    return_dense: bool,
    batch_size: int,
    disable_beta_approx: bool,
    warn_monomorphic: bool,
    max_effects: int,
    fdr: float,
    qvalue_lambda: float,
    seed: int,
    prior_weights: dict[str, pd.Series] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame]:
    """Run TensorQTL via the Python API directly (no subprocess/file export)."""
    try:
        from tensorqtl import cis, post, susie, trans
    except ImportError as e:
        raise ImportError(
            "tensorqtl is required for `run_tensorqtl` with `use_python_api=True`. "
            "Install with `pip install tensorqtl`. Please also install rpy2 and R package qvalue."
        ) from e

    if mode == "cis_nominal":
        if prefix is None:
            raise ValueError("If mode is 'cis_nominal', then a prefix must be given.")

        interaction_term = None
        if interaction_df is not None:
            interaction_term = (
                pd.read_csv(interaction_df, sep="\t", index_col=0)
                if isinstance(interaction_df, str)
                else interaction_df
            )

        signif_df = None
        if cis_output is not None:
            if isinstance(cis_output, str):
                signif_df = (
                    pd.read_parquet(cis_output)
                    if cis_output.endswith(".parquet")
                    else pd.read_csv(cis_output, sep="\t", index_col=0)
                )
            else:
                signif_df = cis_output

        cis.map_nominal(
            genotype_df,
            variant_df,
            phenotype_df,
            phenotype_pos_df,
            prefix,
            covariates_df=covariates_df,
            interaction_df=interaction_term,
            maf_threshold_interaction=maf_threshold_interaction,
            window=window,
        )

        if signif_df is not None:
            from tensorqtl.post import get_significant_pairs

            signif_pairs_df = get_significant_pairs(signif_df, prefix, fdr=fdr)
            signif_pairs_df.to_parquet(f"{prefix}.cis_qtl.signif_pairs.parquet")

        cis_qtl_pairs = pd.concat(
            [pd.read_parquet(path) for path in glob.glob(f"{prefix}.cis_qtl_pairs.*.parquet")], axis=0
        )
        cis_qtl_signif_pairs = (
            pd.read_parquet(f"{prefix}.cis_qtl.signif_pairs.parquet") if signif_df is not None else None
        )
        cis_qtl_top_assoc = (
            pd.read_csv(f"{prefix}.cis_qtl_top_assoc.txt.gz", sep="\t") if interaction_term is not None else None
        )
        results = (cis_qtl_pairs, cis_qtl_signif_pairs, cis_qtl_top_assoc)

    elif mode == "cis":
        results = cis.map_cis(
            genotype_df,
            variant_df,
            phenotype_df,
            phenotype_pos_df,
            covariates_df=covariates_df,
            nperm=permutations,
            window=window,
            beta_approx=not disable_beta_approx,
            warn_monomorphic=warn_monomorphic,
            seed=seed,
        )
        try:
            post.calculate_qvalues(results, fdr=fdr, qvalue_lambda=qvalue_lambda)
        except (NameError, AttributeError, ImportError) as e:
            raise RuntimeError(
                "tensorqtl.post.calculate_qvalues failed (R/rpy2/Bioconductor qvalue unavailable in this "
                f"environment: {type(e).__name__}: {e}). Install the real dependency (see this function's "
                "docstring); there is no fallback."
            ) from e

    elif mode == "cis_independent":
        if cis_output is None:
            raise ValueError(
                "cis_output can't be None in mode 'cis_independent'. "
                "Please provide a path to the cis permutation output (with q-values) or a DataFrame."
            )
        cis_df = pd.read_csv(cis_output, sep="\t", index_col=0) if isinstance(cis_output, str) else cis_output
        results = cis.map_independent(
            genotype_df,
            variant_df,
            cis_df,
            phenotype_df,
            phenotype_pos_df,
            covariates_df=covariates_df,
            fdr=fdr,
            fdr_col="qval",
            nperm=permutations,
            window=window,
            seed=seed,
        )

    elif mode == "trans":
        trans_df = trans.map_trans(
            genotype_df,
            phenotype_df,
            covariates_df=covariates_df,
            return_sparse=not return_dense,
            pval_threshold=pval_threshold,
            maf_threshold=maf_threshold,
            batch_size=batch_size,
        )
        results = trans.filter_cis(trans_df, phenotype_pos_df, variant_df, window=window)

    elif mode == "cis_susie":
        if cis_output is None:
            raise ValueError(
                "cis_output can't be None in mode 'cis_susie'. "
                "Please provide a path to the significant pairs file (parquet or tsv) or a DataFrame."
            )
        if isinstance(cis_output, str):
            signif_df = (
                pd.read_parquet(cis_output) if cis_output.endswith(".parquet") else pd.read_csv(cis_output, sep="\t")
            )
        else:
            signif_df = cis_output

        if "qval" in signif_df:
            signif_df = signif_df[signif_df["qval"] <= fdr]
        phenotype_ids = phenotype_df.index[phenotype_df.index.isin(signif_df["phenotype_id"].unique())]
        pheno_df_sub = phenotype_df.loc[phenotype_ids]
        pheno_pos_df_sub = phenotype_pos_df.loc[phenotype_ids]

        if prior_weights is None:
            susie_summary, susie_dict = susie.map(
                genotype_df,
                variant_df,
                pheno_df_sub,
                pheno_pos_df_sub,
                covariates_df,
                L=max_effects,
                window=window,
                summary_only=False,
                max_iter=500,
                maf_threshold=maf_threshold,
            )
        else:
            susie_summary, susie_dict = _map_susie_with_prior_weights(
                genotype_df, variant_df, pheno_df_sub, pheno_pos_df_sub, covariates_df,
                prior_weights=prior_weights, L=max_effects, window=window, max_iter=500,
                maf_threshold=maf_threshold,
            )
        results = (susie_dict, susie_summary)

    elif mode == "trans_susie":
        if susie_loci is None:
            raise ValueError(
                "susie_loci can't be None in mode 'trans_susie'. "
                "Please provide a path to the loci file (parquet or tsv) or a DataFrame."
            )
        if isinstance(susie_loci, str):
            loci_df = (
                pd.read_parquet(susie_loci) if susie_loci.endswith(".parquet") else pd.read_csv(susie_loci, sep="\t")
            )
        else:
            loci_df = susie_loci

        susie_summary, susie_dict = susie.map_loci(
            loci_df,
            genotype_df,
            variant_df,
            phenotype_df,
            covariates_df,
            L=max_effects,
            window=window,
            max_iter=500,
        )
        results = (susie_dict, susie_summary)

    else:
        raise ValueError(f"Unknown mode: '{mode}'.")

    return results


def run_tensorqtl(
    dd: DonorData,
    n_pcs: int = 50,
    mode: Literal["cis_nominal", "cis_independent", "cis", "trans", "cis_susie", "trans_susie"] = None,
    permutations: int = 10000,
    cis_output: str = None,
    interaction_df: str = None,
    susie_loci: str = None,
    window: int = 1000000,
    pval_threshold: float = 1e-5,
    logp: bool = False,
    maf_threshold: float = 0,
    maf_threshold_interaction: float = 0.05,
    dosages: bool = False,
    return_dense: bool = False,
    return_r2: bool = False,
    best_only: bool = False,
    output_text: bool = False,
    batch_size: int = 20000,
    chunk_size: int | str = None,
    disable_beta_approx: bool = False,
    warn_monomorphic: bool = True,
    max_effects: int = 10,
    fdr: float = 0.05,
    qvalue_lambda: float = None,
    seed: int = None,
    prefix: str = None,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: list[str] | None = None,
    dtype: str = "float32",
    use_python_api: bool = False,
    run: bool = True,
    read_results: bool = True,
    save_cmd_file: bool = False,
    plink_export_kwargs: dict | None = None,
    remove_intermediate_files: bool = True,
    overwrite_covariates_export: bool = True,
    overwrite_phenotype_export: bool = True,
    overwrite_plink_export: bool = True,
    prior_weights: dict[str, pd.Series] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame] | str:
    """
    Run cis- or trans-QTL mapping using TensorQTL on donor-level aggregated expression and genotype data.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing single-cell gene expression (`dd.C`) and donor-level genotype data (`dd.G`).
    mode : {'cis_nominal', 'cis_independent',
            'cis', 'trans', 'cis_susie', 'trans_susie'}, optional
        Type of QTL analysis to perform.
    prefix : str, optional
        File prefix used for generating intermediate input/output files. Required for most modes.
    cis_output : str, optional
        Path to output file for `cis_independent` and `cis_susie` modes.
    interaction_df : str, optional
        Path to interaction terms file required for `cis_nominal` mode.
    susie_loci : str, optional
        Path to SuSiE loci file required for `trans_susie` mode.
    permutations : int, default=10000
        Number of permutations used for empirical cis-QTL analysis.
    fdr : float, default=0.05
        False Discovery Rate threshold for significant hits in empirical cis-QTL mode.
    qvalue_lambda : float, optional
        Lambda parameter for q-value estimation in empirical mode.
        ``mode="cis"`` computes q-values via ``tensorqtl.post.calculate_qvalues``,
        which hard-requires R + the Bioconductor ``qvalue`` package + rpy2
        (install via ``mamba install -c conda-forge -c bioconda r-base
        bioconductor-qvalue rpy2`` into this env; also needs that env's
        ``bin/`` ahead of system R on ``PATH`` so tensorqtl's own ``which R``
        check finds it, not e.g. an LSF-wrapped ``R``). There is no
        fallback if this stack is unavailable; it raises immediately
        rather than silently substituting a different statistical method
        (Benjamini-Hochberg FDR assumes pi0=1, so it's more conservative
        than Storey's method whenever many tests are truly non-null; this
        project's own cis-mode runs silently used exactly that substitute
        for weeks before it was caught, hence no fallback exists anymore).
    window : int, default=1000000
        Genomic window (in base pairs) around phenotype for filtering cis effects.
    pval_threshold : float, default=1e-5
        P-value threshold for reporting significant QTL associations.
    maf_threshold : float, default=0
        Minimum allele frequency threshold for variants in QTL analysis.
    maf_threshold_interaction : float, default=0.05
        MAF threshold for interaction terms in `cis_nominal` mode.
    best_only : bool, default=False
        If True, only report the best association per phenotype (only applies to some modes).
    batch_size : int, default=20000
        Number of phenotype-variant pairs processed per batch (important for trans modes).
    chunk_size : int or str, optional
        Size of variant chunks processed in cis modes. Can be string like "1M" or integer base pairs.
    max_effects : int, default=10
        Maximum number of independent signals to detect in SuSiE-based modes (maps to L parameter).
    seed : int, optional
        Random seed for reproducibility, especially for permutation testing.
    logp : bool, default=False
        If True, output -log10(p-values) instead of raw p-values.
    dosages : bool, default=False
        If True, use dosage data for association testing (if available).
    return_dense : bool, default=False
        If True, return dense matrix results (applies to trans-QTL mode).
    return_r2 : bool, default=False
        If True, include r² statistics in results.
    output_text : bool, default=False
        If True, also output results as text files.
    disable_beta_approx : bool, default=False
        If True, disables approximation of beta coefficients.
    warn_monomorphic : bool, default=True
        If True, warnings are issued for monomorphic variants.
    n_pcs : int, default=50
        Number of principal components to compute from single-cell expression data, donor-aggregated
        (mean) into dd.G.obsm["X_pca"]. Only computed when "X_pca" is itself listed in
        additional_covariates; otherwise unused.
    encode_sex : bool, default=True
        If True, includes donor sex as a covariate.
    encode_age : bool, default=True
        If True, includes donor age (z-normalized if needed) as a covariate.
    additional_covariates : list of str, optional
        Additional covariates from `dd.G.obs` or `dd.G.obsm` to include in the model.
    dtype : str, default="float32"
        Data type to cast covariates and matrices for QTL model input.
    use_python_api : bool, default=False
        If True, runs TensorQTL directly via its Python API without exporting intermediate files or
        invoking a subprocess. Genotypes are loaded from `dd.G` in memory. If False (default), the
        CLI-based workflow is used, which exports PLINK, phenotype, and covariate files and calls
        TensorQTL via subprocess.
    run : bool, default=True
        If True, executes the TensorQTL command. If False, returns the constructed command as a string.
        Only applies when `use_python_api=False`.
    read_results : bool, default=True
        If True, reads and returns the result files. If False, returns the paths to the output files.
        Only applies when `use_python_api=False`.
    save_cmd_file : bool, default=False
        If True, saves the constructed TensorQTL command to a file instead of printing.
        Only applies when `use_python_api=False`.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for `to_plink` function.
        Only applies when `use_python_api=False`.
    remove_intermediate_files : bool, default=True
        If True, removes the intermediate files.
        Only applies when `use_python_api=False`.
    overwrite_covariates_export : bool, default=True
        If True, overwrites the covariates export.
    overwrite_phenotype_export : bool, default=True
        If True, overwrites the phenotype export.
    overwrite_plink_export : bool, default=True
        If True, overwrites the plink export.
    prior_weights : dict[str, pd.Series], optional
        ``mode="cis_susie"`` only, requires ``use_python_api=True``. Keyed
        by phenotype_id; each value is a Series of raw (unnormalized)
        per-variant prior weights indexed by variant_id, used in place of
        SuSiE's default flat/uniform prior over variants in the cis-window, 
        e.g. a sequence-model-predicted regulatory effect size, letting
        an orthogonal source of information sharpen fine-mapping
        resolution on credible sets the flat prior alone leaves ambiguous.
        Variants in a phenotype's window missing from its Series are given
        that Series' own median as a neutral (not zero) prior. 

    Returns
    -------
    pd.DataFrame, tuple, str, or list[str]
        Depending on mode and read_results:
        - If use_python_api=True or (run=True and read_results=True): returns pandas DataFrame(s) or tuple of results.
        - If run=True and read_results=False: returns list of output file paths.
        - If run=False: returns the constructed TensorQTL command as a string.

    Raises
    ------
    ImportError
        If required dependencies (`plink2`, `tensorqtl`) are not found in system path.

    ValueError
        If required parameters (`prefix`, `cis_output`, `susie_loci`) are not provided for the selected mode.
    """
    if plink_export_kwargs is None:
        plink_export_kwargs = {}

    if prior_weights is not None:
        if mode != "cis_susie":
            raise ValueError(f"prior_weights is only supported for mode='cis_susie', got mode={mode!r}.")
        if not use_python_api:
            raise ValueError(
                "prior_weights requires use_python_api=True; it is not expressible through tensorqtl's "
                "own CLI/subprocess interface, only via cellink's Python-API re-implementation of cis_susie."
            )

    if additional_covariates and "X_pca" in additional_covariates:
        if "X_pca" not in dd.C.obsm:
            logger.info("Calculating PCA.")
            sc.pp.pca(dd.C, n_comps=n_pcs)
        if "X_pca" not in dd.G.obsm:
            dd.aggregate(obsm="X_pca", key_added="X_pca", func="mean", verbose=True)

    dd.aggregate(key_added="PB", sync_var=True, verbose=True)

    phenotype_df = dd.G.obsm["PB"].T
    phenotype_df.index.name = "Geneid"
    phenotype_pos_df = dd.C.var[["chrom", "start", "end"]].rename(columns={"chrom": "chr"})
    phenotype_pos_df["Geneid"] = phenotype_pos_df.index

    covariate_list = []
    covariate_list.append(pd.DataFrame(np.ones((dd.shape[0], 1)), columns=["intercept"], index=phenotype_df.columns))

    if encode_sex:
        sex_codes = dd.G.obs["sex"].astype("category").cat.codes
        covariate_list.append(pd.DataFrame(sex_codes.values, columns=["sex"], index=phenotype_df.columns))

    if encode_age:
        age_values = dd.G.obs[["age"]].values.astype(dtype)
        mean = age_values.mean()
        std = age_values.std()
        tolerance = 1e-2
        already_z_normalized = np.isclose(mean, 0.0, atol=tolerance) and np.isclose(std, 1.0, atol=tolerance)
        if not already_z_normalized and std > 0:
            logger.info("Performing z-normalization of age.")
            age_values = (age_values - mean) / std
        covariate_list.append(pd.DataFrame(age_values, columns=["age"], index=phenotype_df.columns))

    if additional_covariates:
        for cov in additional_covariates:
            if cov in dd.G.obs.columns:
                covariate_df = pd.DataFrame(
                    dd.G.obs[[cov]].values.astype(dtype), columns=[cov], index=phenotype_df.columns
                )
                covariate_list.append(covariate_df)
            elif cov in dd.G.obsm:
                cov_matrix = asarray(dd.G.obsm[cov]).astype(dtype)
                if cov_matrix.ndim == 1:
                    covariate_list.append(pd.DataFrame(cov_matrix, columns=[cov], index=phenotype_df.columns))
                else:
                    covariate_list.append(
                        pd.DataFrame(
                            cov_matrix,
                            columns=[f"{cov}_{i}" for i in range(cov_matrix.shape[1])],
                            index=phenotype_df.columns,
                        )
                    )
            else:
                raise ValueError(f"Covariate '{cov}' not found in dd.G.obs or dd.G.obsm.")

    covariates_df = pd.concat(covariate_list, axis=1)

    if use_python_api:
        if importlib.util.find_spec("tensorqtl") is None:
            raise ImportError("tensorqtl is required for `run_tensorqtl`. Please install it.")

        genotype_df = pd.DataFrame(dd.G.X.T, index=dd.G.var.index, columns=dd.G.obs.index)
        variant_df = dd.G.var[["chrom", "pos"]].copy()
        variant_df["index"] = range(len(variant_df))

        return _run_tensorqtl_python_api(
            mode=mode,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            covariates_df=covariates_df,
            genotype_df=genotype_df,
            variant_df=variant_df,
            prefix=prefix,
            permutations=permutations,
            cis_output=cis_output,
            interaction_df=interaction_df,
            susie_loci=susie_loci,
            window=window,
            pval_threshold=pval_threshold,
            maf_threshold=maf_threshold,
            maf_threshold_interaction=maf_threshold_interaction,
            return_dense=return_dense,
            batch_size=batch_size,
            disable_beta_approx=disable_beta_approx,
            warn_monomorphic=warn_monomorphic,
            max_effects=max_effects,
            fdr=fdr,
            qvalue_lambda=qvalue_lambda,
            seed=seed,
            prior_weights=prior_weights,
        )

    if run:
        if shutil.which("plink2") is None:
            raise ImportError("plink2 is required for `run_tensorqtl`. Please install it.")
        if importlib.util.find_spec("tensorqtl") is None:
            raise ImportError("tensorqtl is required for `run_tensorqtl`. Please install it.")

    args = {
        "permutations": permutations,
        "window": window,
        "pval_threshold": pval_threshold,
        "logp": logp,
        "maf_threshold": maf_threshold,
        "maf_threshold_interaction": maf_threshold_interaction,
        "dosages": dosages,
        "return_dense": return_dense,
        "return_r2": return_r2,
        "best_only": best_only,
        "output_text": output_text,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "susie_loci": susie_loci,
        "disable_beta_approx": disable_beta_approx,
        "warn_monomorphic": warn_monomorphic,
        "max_effects": max_effects,
        "fdr": fdr,
        "qvalue_lambda": qvalue_lambda,
        "seed": seed,
    }

    if mode == "cis_nominal" and prefix is None:
        raise ValueError("If mode cis_nominal, then a prefix must be given.")

    if not os.path.isfile(f"{prefix}_phenotype.bed.gz") or overwrite_phenotype_export:
        phenotype_write_df = pd.concat([phenotype_pos_df, phenotype_df], axis=1)
        phenotype_write_df = phenotype_write_df.rename(columns={"chr": "#chr"})
        phenotype_write_df = phenotype_write_df.groupby("#chr", sort=False, group_keys=False).apply(
            lambda x: x.sort_values(["start", "end"])
        )
        with gzip.open(f"{prefix}_phenotype.bed.gz", "wt") as f:
            f.write("\t".join(phenotype_write_df.columns.tolist()) + "\n")
            phenotype_write_df.to_csv(f, sep="\t", header=False, index=False)

    if not os.path.isfile(f"{prefix}_donor_features.tsv") or overwrite_covariates_export:
        covariates_export_df = covariates_df.copy()
        covariates_export_df.index.name = "iid"
        covariates_export_df = covariates_export_df.T
        covariates_export_df.to_csv(f"{prefix}_donor_features.tsv", sep="\t")

    geno = prefix
    covar = f"{prefix}_donor_features.tsv"
    pheno = f"{prefix}_phenotype.bed.gz"

    if not os.path.isfile(f"{prefix}.pgen") or overwrite_plink_export:
        to_plink(dd.G, prefix, **plink_export_kwargs)
        cmd_plink_conversion = f"plink2 --bfile {geno} --make-pgen --out {geno}"
        subprocess.run(cmd_plink_conversion, check=True, shell=True)

    cmd = f"python -m tensorqtl {geno} {pheno} {prefix} --covariates {covar} --mode {mode}"

    for key, value in args.items():
        if isinstance(value, bool) and value:
            cmd += f" --{key}"
        elif value is not None and not isinstance(value, bool):
            cmd += f" --{key} {value}"

    if mode == "cis_nominal":
        if interaction_df is not None:
            cmd += f" --interaction {interaction_df}"
        if cis_output is not None:
            cmd += f" --cis_output {cis_output}"
    elif mode == "cis_independent":
        if cis_output is None:
            raise ValueError("cis_output can't be None in mode 'cis_independent'. Please provide a valid path.")
        cmd += f" --cis_output {cis_output}"
    elif mode == "cis_susie":
        if cis_output is None:
            raise ValueError("cis_output can't be None in mode 'cis_susie'. Please provide a valid path.")
        cmd += f" --cis_output {cis_output}"
    elif mode == "trans_susie":
        if susie_loci is None:
            raise ValueError("susie_loci can't be None in mode 'trans_susie'. Please provide a valid path.")
        cmd += f" --susie_loci {susie_loci}"

    if run:
        subprocess.run(cmd, check=True, shell=True)

        if remove_intermediate_files:
            extensions = [".bim", ".fam", ".bed", ".pgen", ".psam", ".pvar", "_donor_features.tsv", "_phenotype.bed.gz"]
            for ext in extensions:
                filename = prefix + ext
                if os.path.isfile(filename):
                    os.remove(filename)

        if read_results:
            results = read_tensorqtl_results(prefix, mode, cis_output=cis_output, interaction_df=interaction_df)

        return results

    else:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(cmd + "\n")
        else:
            return cmd


def run_dense_trans_scan(
    dd: DonorData,
    variant_id: str,
    gene_of_interest: str | None = None,
    n_pcs: int = 20,
    window: int = 1_000_000,
    maf_threshold: float = 0.05,
    batch_size: int = 20000,
    encode_sex: bool | None = None,
    encode_age: bool = False,
    additional_covariates: list[str] | None = None,
    prefix: str | None = None,
    use_python_api: bool = True,
) -> pd.DataFrame:
    """
    Dense, unfiltered genome-wide trans-QTL scan of one fixed variant against every gene.

    A thin, specific mode of ``run_tensorqtl``: subsets ``dd.G`` to exactly
    one variant and calls ``run_tensorqtl(..., mode="trans", pval_threshold=1.0)``,
    which bypasses TensorQTL's own sparse write-time filter (normally, trans
    mode only keeps pairs below some p-value threshold). The result is the
    full, unfiltered rank of every gene's association with that one variant,
    useful for asking whether a candidate trans hit is a real standout or
    one of many similarly-ranked near-ties.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing single-cell gene expression (`dd.C`) and
        donor-level genotype data (`dd.G`). Both are used as-is; any
        celltype/expression-level filtering is the caller's responsibility.
    variant_id : str
        The single variant to test against every gene, matched exactly
        against ``dd.G.var_names``.
    gene_of_interest : str, optional
        If given, logs that gene's rank and p-value in the returned scan
        (or a warning if it's absent, e.g. excluded by ``window``).
    n_pcs : int, default=20
        Number of leading genotype PCs kept from ``dd.G.obsm["gPCs"]`` (if
        present) when passed through as an additional covariate.
    window : int, default=1_000_000
        Genomic window (in base pairs) used to drop pairs where the variant
        falls within a gene's own cis-window (``tensorqtl.trans.filter_cis``),
        so real cis effects don't masquerade as "dense trans" hits.
    maf_threshold : float, default=0.05
        Minimum MAF for the target variant to be tested; lower this for a
        real, low-frequency (but not vanishingly rare) variant that the
        default excludes.
    batch_size : int, default=20000
        Number of phenotype-variant pairs processed per batch, passed
        through to ``run_tensorqtl``.
    encode_sex : bool, optional
        If True, includes donor sex as a covariate (donors with missing sex
        are dropped first, since PLINK export hard-casts sex to int32). If
        None (default), auto-detected from whether ``dd.G.obs["sex"])`` has
        any real signal.
    encode_age : bool, default=False
        If True, includes donor age as a covariate. Default False, matching
        this scan's typical use as a discovery/ranking tool rather than a
        fully covariate-adjusted association test.
    additional_covariates : list of str, optional
        Additional covariates from `dd.G.obs` or `dd.G.obsm` to include in
        the model. If None (default), ``"gPCs"`` is used automatically when
        present in ``dd.G.obsm``.
    prefix : str, optional
        File prefix for intermediate files. Only used when
        ``use_python_api=False``; unused (no files are written) otherwise.
    use_python_api : bool, default=True
        If True (default), runs TensorQTL directly via its Python API
        without exporting intermediate files or invoking a subprocess. If
        False, requires ``prefix`` and falls back to the CLI-based workflow
        (see ``run_tensorqtl``).

    Returns
    -------
    pd.DataFrame
        One row per gene, columns as returned by ``run_tensorqtl(mode="trans")``
        (``phenotype_id``, ``variant_id``, ``beta``, ``pval``, ...) plus
        ``chrom``/``start``/``end`` (from ``dd.C.var``) and ``rank_by_pval``
        (1 = lowest p-value), sorted by ascending p-value.

    Raises
    ------
    ValueError
        If ``variant_id`` does not match exactly one entry in ``dd.G.var_names``.
    """
    var_mask = np.asarray(dd.G.var_names == variant_id)
    n_hits = int(var_mask.sum())
    if n_hits != 1:
        raise ValueError(f"expected exactly 1 match for variant_id={variant_id!r} in dd.G.var_names, found {n_hits}")

    G_target = dd.G[:, var_mask].copy()
    dd_scan = DonorData(G=G_target, C=dd.C, donor_id=dd.donor_id)

    if encode_sex is None:
        encode_sex = "sex" in dd_scan.G.obs.columns and dd_scan.G.obs["sex"].notna().any()
    if encode_sex and "sex" in dd_scan.G.obs.columns and dd_scan.G.obs["sex"].isna().any():
        n_before = dd_scan.G.n_obs
        dd_scan = DonorData(
            G=dd_scan.G[dd_scan.G.obs["sex"].notna()].copy(), C=dd_scan.C, donor_id=dd_scan.donor_id
        )
        logger.info(
            f"run_dense_trans_scan: dropped {n_before - dd_scan.G.n_obs} donor(s) with missing sex "
            "(needed for sex-covariate PLINK export)."
        )

    if "gPCs" in dd_scan.G.obsm:
        dd_scan.G.obsm["gPCs"] = dd_scan.G.obsm["gPCs"][:, :n_pcs]
    if additional_covariates is None:
        additional_covariates = ["gPCs"] if "gPCs" in dd_scan.G.obsm else None

    logger.info(
        f"run_dense_trans_scan: dense trans scan of {variant_id!r} against {dd_scan.C.n_vars} genes "
        f"({dd_scan.G.n_obs} donors); mode='trans', pval_threshold=1.0."
    )
    df = run_tensorqtl(
        dd_scan,
        mode="trans",
        pval_threshold=1.0,
        maf_threshold=maf_threshold,
        batch_size=batch_size,
        n_pcs=n_pcs,
        window=window,
        prefix=prefix,
        encode_sex=encode_sex,
        encode_age=encode_age,
        additional_covariates=additional_covariates,
        use_python_api=use_python_api,
    )

    gene_pos = dd_scan.C.var[["chrom", "start", "end"]].copy()
    gene_pos.index.name = "phenotype_id"
    df = df.merge(gene_pos, left_on="phenotype_id", right_index=True, how="left")
    df = df.sort_values("pval").reset_index(drop=True)
    df.insert(0, "rank_by_pval", np.arange(1, len(df) + 1))

    if gene_of_interest is not None:
        goi_row = df[df["phenotype_id"] == gene_of_interest]
        if len(goi_row) == 1:
            logger.info(
                f"run_dense_trans_scan: {gene_of_interest} rank={int(goi_row['rank_by_pval'].iloc[0])}, "
                f"pval={goi_row['pval'].iloc[0]:.6e} (of {len(df)} genes)."
            )
        else:
            logger.warning(
                f"run_dense_trans_scan: {gene_of_interest} not found in the scan output "
                f"({len(goi_row)} rows); likely excluded by the cis-window filter or absent from dd.C."
            )

    return df
