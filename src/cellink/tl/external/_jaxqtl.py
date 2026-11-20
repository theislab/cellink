import gzip
import logging
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata.utils import asarray

from cellink._core import DonorData
from cellink._core.schema import require_genotypes
from cellink._extras import install_hint
from cellink.io import to_plink

logger = logging.getLogger(__name__)

_CHOICES = {
    "mode": ("cis", "nominal", "trans"),
    "model": ("gaussian", "poisson", "nb"),
    "test": ("score", "wald"),
    "solver": ("cholesky", "cg", "qr"),
    "platform": ("cpu", "gpu", "tpu"),
    "pcs_normalization": ("none", "library-size", "tmm"),
    "pcs_transform": ("none", "log1p"),
}


def _test_name(test: str, spa: bool, model: str) -> str:
    """Name of the variant test in jaxqtl's output file names."""
    if test == "wald":
        return "wald"
    return "score.spa" if spa and model != "gaussian" else "score"


def _result_paths(prefix: str, mode: str, test: str, acat: bool) -> list[str]:
    """Files jaxqtl writes for one scan."""
    if mode == "cis":
        return [f"{prefix}.cis.{test}.{'acat' if acat else 'perm'}.parquet.gz"]
    if mode == "nominal":
        return [f"{prefix}.nominal.{test}.parquet.gz"]
    return [f"{prefix}.trans.{test}.sumstats.parquet.gz", f"{prefix}.trans.{test}.variant.info.parquet.gz"]


def read_jaxqtl_results(
    prefix: str,
    mode: Literal["cis", "nominal", "trans"] = "cis",
    test: Literal["score", "score.spa", "wald"] = "score",
    acat: bool = False,
) -> pd.DataFrame:
    """
    Read the results of a jaxQTL scan.

    jaxQTL names its output files after the subcommand and the variant test:
    ``{prefix}.cis.{test}.perm.parquet.gz`` (``acat`` instead of ``perm`` with ``--acat``),
    ``{prefix}.nominal.{test}.parquet.gz`` and ``{prefix}.trans.{test}.sumstats.parquet.gz``,
    whose variant annotation ``{prefix}.trans.{test}.variant.info.parquet.gz`` is merged in on ``snp``.

    Parameters
    ----------
    prefix : str
        The ``--out`` prefix of the jaxQTL run.
    mode : {'cis', 'nominal', 'trans'}, default='cis'
        The jaxQTL subcommand that was run.
    test : {'score', 'score.spa', 'wald'}, default='score'
        The variant test: ``score.spa`` for the score test with ``--spa`` (jaxQTL applies SPA only to
        the count models).
    acat : bool, default=False
        Whether the ``cis`` scan used ``--acat``.

    Returns
    -------
    pd.DataFrame
        The jaxQTL results.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist. jaxQTL writes none when no gene passes its filters.
    """
    if mode not in _CHOICES["mode"]:
        raise ValueError(f"mode must be one of {_CHOICES['mode']}, got {mode!r}.")
    path, *info = _result_paths(prefix, mode, test, acat)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"jaxQTL result file {path} not found. jaxQTL writes none when no gene passes its filters "
            f"(see {prefix}.log)."
        )
    results = pd.read_parquet(path)
    if info:
        results = results.merge(pd.read_parquet(info[0]), on="snp", how="left")
    return results


def _as_list(values) -> list[str] | None:
    """Names given as one string or a sequence, as a list."""
    if values is None:
        return None
    return [values] if isinstance(values, str) else [str(v) for v in values]


def _id_file(ids, path: str) -> tuple[str, bool]:
    """Path of a file with one ID per line, and whether it was written here.

    ``ids`` is either such a file or the IDs, which are then written to ``path``.
    """
    if isinstance(ids, str | os.PathLike):
        return os.fspath(ids), False
    pd.Series(list(ids), dtype=str).to_csv(path, index=False, header=False)
    return path, True


def run_jaxqtl(
    dd: DonorData,
    mode: Literal["cis", "nominal", "trans"] = "cis",
    prefix: str | None = None,
    out: str | None = None,
    model: Literal["gaussian", "poisson", "nb"] = "nb",
    test: Literal["score", "wald"] = "score",
    robust_se: bool = False,
    spa: bool = False,
    acat: bool = False,
    nperm: int = 1000,
    window: int = 500_000,
    tss_centered: bool = False,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: list[str] | None = None,
    n_pcs: int = 50,
    dtype: str = "float32",
    covar_name: Sequence[str] | None = None,
    rm_covar: Sequence[str] | None = None,
    normalize_covar: bool = True,
    one_hot: bool = False,
    no_intercept: bool = False,
    expression_pcs: int | None = 2,
    pcs_normalization: Literal["none", "library-size", "tmm"] = "library-size",
    pcs_transform: Literal["none", "log1p"] = "log1p",
    pcs_libsize: str | None = None,
    pcs_libsize_name_from_covar: str | None = None,
    offset: str | None = None,
    offset_name_from_covar: str | None = None,
    set_offset_from_libsize: bool | None = None,
    counts_layer: str | None = None,
    libsize_key: str | None = None,
    keep: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
    min_indiv_expr_pct: float | None = None,
    min_gene_expr_pct: float | None = None,
    chrom: str | None = None,
    genes: Sequence[str] | None = None,
    gene_list: str | None = None,
    rm_genes: Sequence[str] | None = None,
    exclude_gene_list: str | None = None,
    maf: float | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    gtol: float | None = None,
    step_size: float | None = None,
    solver: Literal["cholesky", "cg", "qr"] | None = None,
    seed: int | None = None,
    platform: Literal["cpu", "gpu", "tpu"] | None = None,
    verbose: bool = False,
    run: bool = True,
    read_results: bool = True,
    save_cmd_file: str | None = None,
    plink_export_kwargs: dict | None = None,
    remove_intermediate_files: bool = True,
    overwrite_covariates_export: bool = True,
    overwrite_phenotype_export: bool = True,
    overwrite_plink_export: bool = True,
    extra_args: list[str] | None = None,
) -> pd.DataFrame | str | list[str]:
    """
    Run cis- or trans-eQTL mapping with jaxQTL on donor-level genotypes and pseudobulk expression.

    This function writes the jaxQTL inputs from a `DonorData` object (genotypes, a phenotype BED and a
    covariate table), builds the ``jaxqtl`` commands and optionally runs them. If `expression_pcs` is set,
    ``jaxqtl compute-pcs`` first appends gene expression PCs to the covariates. Every option of
    ``jaxqtl cis``, ``nominal``, ``trans`` and ``compute-pcs`` (jaxqtl 0.3) is a parameter, named after
    the jaxQTL flag; options left at None are not passed, so jaxQTL's defaults apply. The genotypes are
    written as PLINK 1 hard calls (:func:`cellink.io.to_plink`, ``--bfile``).

    The pseudobulk phenotype depends on the model. The count models (``nb``, ``poisson``) get the
    raw counts summed per donor (from `counts_layer`, or `dd.C.X`), fitted with the log library
    size as offset, as jaxQTL intends; a warning is logged if the sums are not non-negative
    integers, which means the data are probably normalized. ``gaussian`` gets the mean of
    `dd.C.X` per donor.

    Parameters
    ----------
    dd : DonorData
        Object containing donor-level genotype (`dd.D`) and cell-level expression data (`dd.C`).
    mode : {'cis', 'nominal', 'trans'}, default='cis'
        The jaxQTL subcommand: ``cis`` reports the lead variant per gene with a gene-level p-value,
        ``nominal`` every cis variant-gene pair and ``trans`` a genome-wide scan.
    prefix : str, optional
        Prefix for the input files written for jaxQTL. Defaults to "jaxqtl_temp".
    out : str, optional
        Output prefix for the jaxQTL results (``--out``). Defaults to `prefix`.
    model : {'gaussian', 'poisson', 'nb'}, default='nb'
        Model for the expression (``--model``). ``nb`` and ``poisson`` model summed raw counts,
        ``gaussian`` the mean expression per donor.
    test : {'score', 'wald'}, default='score'
        Variant test (``--test``). jaxQTL recommends ``score`` for ``cis`` and ``wald`` for ``nominal``.
    robust_se : bool, default=False
        Huber-White sandwich standard errors (``--robust-se``); requires ``test='wald'``.
    spa : bool, default=False
        Saddle-point approximation of the score-test p-values (``--spa``); requires ``test='score'`` and
        a count model.
    acat : bool, default=False
        Gene-level p-values by ACAT instead of the Beta-approximated permutation test (``--acat``);
        ``mode='cis'`` only.
    nperm : int, default=1000
        Permutations for the Beta approximation (``--nperm``); used by ``mode='cis'`` without `acat`.
    window : int, default=500000
        Cis window in base pairs (``--window``).
    tss_centered : bool, default=False
        Center the cis window on the TSS instead of spanning TSS to TES (``--tss-centered``).
    encode_sex : bool, default=True
        If True, adds sex as a categorical covariate using `dd.D.obs['sex']`. Raises if any donor has no sex.
    encode_age : bool, default=True
        If True, adds age as a numeric covariate from `dd.D.obs['age']`.
    additional_covariates : list of str, optional
        Columns of `dd.D.obs` or keys of `dd.D.obsm` to add as covariates. Non-numeric columns are
        written as text and need `one_hot`.
    n_pcs : int, default=50
        Number of principal components computed from the single-cell expression data and donor-aggregated
        (mean) into dd.D.obsm["X_pca"], only when "X_pca" is listed in `additional_covariates`.
    dtype : str, default='float32'
        Data type of the numeric covariates.
    covar_name : sequence of str, optional
        Use only these covariates (``--covar-name``). Cannot be combined with `rm_covar`.
    rm_covar : sequence of str, optional
        Leave these covariates out (``--rm-covar``). Cannot be combined with `covar_name`.
    normalize_covar : bool, default=True
        Scale the covariates to mean 0 and variance 1 (``--normalize-covar``).
    one_hot : bool, default=False
        One-hot encode text covariates (``--one-hot``).
    no_intercept : bool, default=False
        Do not add an intercept to the covariates (``--no-intercept``).
    expression_pcs : int, optional, default=2
        Number of gene expression PCs computed by ``jaxqtl compute-pcs`` (``--num-pcs``) and added to the
        covariates, as ``ExprPC1``, ``ExprPC2``, .... They are computed from the donors the scan analyzes
        (`keep`, `exclude` and `min_indiv_expr_pct` are passed on) and from all genes. 0 or None skips
        ``compute-pcs``.
    pcs_normalization : {'none', 'library-size', 'tmm'}, default='library-size'
        Expression normalization before the PCA (``compute-pcs --normalization``).
    pcs_transform : {'none', 'log1p'}, default='log1p'
        Transformation after the normalization (``compute-pcs --transform``).
    pcs_libsize : str, optional
        Two-column (``iid``, ``libsize``) TSV of raw library sizes for the PCA normalization
        (``compute-pcs --libsize``). By default jaxQTL sums the counts of the phenotype file.
    pcs_libsize_name_from_covar : str, optional
        Covariate column with raw library sizes for the PCA normalization
        (``compute-pcs --libsize-name-from-covar``).
    offset : str, optional
        Two-column (sample ID, offset) TSV of fixed model offsets (``--offset``), e.g. log library sizes
        computed from all genes when `dd.C` holds only a subset of them.
    offset_name_from_covar : str, optional
        Covariate column used as fixed model offset (``--offset-name-from-covar``).
    set_offset_from_libsize : bool, optional
        Use the log library size as offset (``--set-offset-from-libsize``). jaxQTL computes it as the
        donor's total count over the genes written to the phenotype file, i.e. the genes in `dd.C`.
        Defaults to True for the count models (``nb``, ``poisson``) unless `offset`,
        `offset_name_from_covar` or `libsize_key` is given, and to False for ``gaussian``. Only one of `offset`,
        `offset_name_from_covar` and `set_offset_from_libsize` can be given.
    counts_layer : str, optional
        Layer of `dd.C` with raw counts, summed per donor for the count models. Defaults to `dd.C.X`.
        Not used with ``gaussian``.
    libsize_key : str, optional
        Column of `dd.C.obs` with each cell's total count over all genes, e.g. ``"nCount_RNA"``. Summed
        per donor, these are the library sizes: their log is the offset of the count models (passed as
        ``--offset``), and they normalize the expression PCA (``compute-pcs --libsize``). Without it,
        jaxQTL sums the counts of the phenotype file, i.e. over the genes in `dd.C`, which
        underestimates the library sizes when `dd.C` holds only some of the genes.
    keep : str or sequence of str, optional
        Donors to analyze (``--keep``): a file with one ID per line, or the IDs. Cannot be combined
        with `exclude`.
    exclude : str or sequence of str, optional
        Donors to leave out (``--exclude``): a file with one ID per line, or the IDs.
    min_indiv_expr_pct : float, optional
        Keep donors expressing more than this fraction of genes (``--min-indiv-expr-pct``).
    min_gene_expr_pct : float, optional
        Keep genes expressed in more than this fraction of donors (``--min-gene-expr-pct``).
    chrom : str, optional
        Restrict the scan to this chromosome label (``--chr``).
    genes : sequence of str, optional
        Genes to analyze (``--genes``).
    gene_list : str, optional
        File with the genes to analyze, one per line (``--gene-list``).
    rm_genes : sequence of str, optional
        Genes to leave out (``--rm-genes``).
    exclude_gene_list : str, optional
        File with the genes to leave out, one per line (``--exclude-gene-list``). Only one of `genes`,
        `gene_list`, `rm_genes` and `exclude_gene_list` can be given.
    maf : float, optional
        Exclude variants with a minor allele frequency below this threshold (``--maf``).
    max_iter : int, optional
        Maximum number of GLM iterations (``--max-iter``).
    tol : float, optional
        Absolute GLM likelihood-change tolerance (``--tol``).
    gtol : float, optional
        GLM gradient tolerance (``--gtol``).
    step_size : float, optional
        Initial GLM step size (``--step-size``).
    solver : {'cholesky', 'cg', 'qr'}, optional
        Linear solver for the model fits (``--solver``).
    seed : int, optional
        Seed of jaxQTL's random number generator (``--seed``), for the permutations and the PCA.
    platform : {'cpu', 'gpu', 'tpu'}, optional
        Hardware backend (``--platform``).
    verbose : bool, default=False
        Detailed jaxQTL logging (``--verbose``).
    run : bool, default=True
        If True, runs jaxQTL. If False, only writes the inputs and returns the command(s).
    read_results : bool, default=True
        If True, reads and returns the results as a DataFrame; otherwise returns the output file paths.
    save_cmd_file : str, optional
        With ``run=False``, also writes the command(s) to this file.
    plink_export_kwargs : dict, optional
        Additional keyword arguments for :func:`cellink.io.to_plink`.
    remove_intermediate_files : bool, default=True
        If True, removes the input files after the run.
    overwrite_covariates_export : bool, default=True
        If True, overwrites the covariates export.
    overwrite_phenotype_export : bool, default=True
        If True, overwrites the phenotype export.
    overwrite_plink_export : bool, default=True
        If True, overwrites the genotype export.
    extra_args : list of str, optional
        Further arguments appended to the scan command verbatim.

    Returns
    -------
    pd.DataFrame, str, or list[str]
        If run=True and read_results=True, a DataFrame of QTL mapping results (see
        :func:`read_jaxqtl_results`).
        If run=True and read_results=False, the paths of the result files and the log.
        If run=False, the jaxQTL command(s) as a string (``compute-pcs`` and the scan joined by ``&&``).

    Raises
    ------
    ImportError
        If the ``jaxqtl`` executable is not found.
    ValueError
        If an option has an invalid value or options that exclude each other are combined, or a
        covariate is not found in the DonorData object or is missing for some donors.

    Notes
    -----
    jaxqtl 0.3.1 fails in two cases, both inside jaxqtl: with ``model='gaussian'``, the permutation
    calibration of ``mode='cis'`` diverges (``acat=True`` and ``mode='nominal'`` work), and
    ``mode='trans'`` cannot write its results. Its ``--dosage`` reader needs dosages in every genotype
    record, which PGEN files store only for variants with non-integer values, so `run_jaxqtl` tests
    hard calls.
    """
    require_genotypes(dd)
    for name, value in (
        ("mode", mode),
        ("model", model),
        ("test", test),
        ("solver", solver),
        ("platform", platform),
        ("pcs_normalization", pcs_normalization),
        ("pcs_transform", pcs_transform),
    ):
        if value is not None and value not in _CHOICES[name]:
            raise ValueError(f"{name} must be one of {_CHOICES[name]}, got {value!r}.")
    count_model = model in ("nb", "poisson")
    if set_offset_from_libsize is None:
        set_offset_from_libsize = (
            count_model and offset is None and offset_name_from_covar is None and libsize_key is None
        )
    exclusive = [
        ("keep", "exclude"),
        ("covar_name", "rm_covar"),
        ("genes", "gene_list", "rm_genes", "exclude_gene_list"),
        ("offset", "offset_name_from_covar", "set_offset_from_libsize"),
        ("pcs_libsize", "pcs_libsize_name_from_covar"),
    ]
    given = {
        "keep": keep,
        "exclude": exclude,
        "covar_name": covar_name,
        "rm_covar": rm_covar,
        "genes": genes,
        "gene_list": gene_list,
        "rm_genes": rm_genes,
        "exclude_gene_list": exclude_gene_list,
        "offset": offset,
        "offset_name_from_covar": offset_name_from_covar,
        "set_offset_from_libsize": set_offset_from_libsize or None,
        "pcs_libsize": pcs_libsize,
        "pcs_libsize_name_from_covar": pcs_libsize_name_from_covar,
    }
    for group in exclusive:
        used = [name for name in group if given[name] is not None]
        if len(used) > 1:
            raise ValueError(f"Pass only one of {', '.join(f'`{n}`' for n in group)}; got {used}.")
    if robust_se and test != "wald":
        raise ValueError("`robust_se` requires test='wald'.")
    if spa and (test != "score" or model == "gaussian"):
        raise ValueError("`spa` applies to the score test of the count models (test='score', model 'nb' or 'poisson').")
    if acat and mode != "cis":
        raise ValueError("`acat` applies to mode='cis' only.")
    if (pcs_libsize or pcs_libsize_name_from_covar) and pcs_normalization == "none":
        raise ValueError(
            "`pcs_libsize` and `pcs_libsize_name_from_covar` need pcs_normalization 'library-size' or 'tmm'."
        )

    if plink_export_kwargs is None:
        plink_export_kwargs = {}
    if not prefix:
        prefix = "jaxqtl_temp"
    if not out:
        out = prefix

    if run and shutil.which("jaxqtl") is None:
        raise ImportError(
            f"The jaxqtl executable is required for `run_jaxqtl`. Install it with `{install_hint('jaxqtl')}` and "
            "make sure `jaxqtl` is on your PATH."
        )

    if additional_covariates and "X_pca" in additional_covariates:
        if "X_pca" not in dd.C.obsm:
            logger.info("Calculating PCA.")
            sc.pp.pca(dd.C, n_comps=n_pcs)
        if "X_pca" not in dd.D.obsm:
            dd.aggregate(obsm="X_pca", key_added="X_pca", func="mean", verbose=True)

    if count_model:
        dd.aggregate(key_added="PB", layer=counts_layer, func="sum", sync_var=True, verbose=True)
    else:
        dd.aggregate(key_added="PB", sync_var=True, verbose=True)
    phenotype_df = dd.D.obsm["PB"].T
    phenotype_df.index.name = "Geneid"
    if count_model:
        values = phenotype_df.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if (values < 0).any() or not np.allclose(values, np.round(values)):
            logger.warning(
                f"model={model!r} expects raw counts, but the summed expression is not non-negative "
                f"integers; {'dd.C.X' if counts_layer is None else f'layer {counts_layer!r}'} is probably "
                "normalized. Pass `counts_layer` with raw counts or use model='gaussian'."
            )

    pheno = f"{prefix}_phenotype.bed.gz"
    if not os.path.isfile(pheno) or overwrite_phenotype_export:
        phenotype_pos_df = dd.C.var[["chrom", "start", "end"]].rename(columns={"chrom": "chr"})
        missing_pos = phenotype_pos_df[["start", "end"]].isna().any(axis=1)
        if missing_pos.any():
            logger.warning(f"Dropping {int(missing_pos.sum())} genes without start/end coordinates.")
        # jaxqtl parses "chrom:start-end" regions and rejects float coordinates such as "16584954.0"
        phenotype_pos_df = phenotype_pos_df[~missing_pos].astype({"start": "int64", "end": "int64"})
        phenotype_pos_df["Geneid"] = phenotype_pos_df.index
        phenotype_write_df = pd.concat([phenotype_pos_df, phenotype_df.loc[phenotype_pos_df.index]], axis=1)
        with gzip.open(pheno, "wt") as f:
            f.write("#" + "\t".join(phenotype_write_df.columns.tolist()) + "\n")
            phenotype_write_df.to_csv(f, sep="\t", header=False, index=False)

    covar = f"{prefix}_donor_features.tsv"
    if not os.path.isfile(covar) or overwrite_covariates_export:
        covariate_list = []

        if encode_sex:
            n_missing = int(dd.D.obs["sex"].isna().sum())
            if n_missing:
                raise ValueError(f"{n_missing} donor(s) have no sex in dd.D.obs; drop them or pass encode_sex=False.")
            sex_codes = dd.D.obs["sex"].astype("category").cat.codes
            covariate_list.append(pd.DataFrame(sex_codes.values, columns=["sex"], index=phenotype_df.columns))

        if encode_age:
            age_values = dd.D.obs[["age"]].values.astype(dtype)
            covariate_list.append(pd.DataFrame(age_values, columns=["age"], index=phenotype_df.columns))

        for cov in additional_covariates or []:
            if cov in dd.D.obs.columns:
                column = dd.D.obs[cov]
                if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
                    values = column.to_numpy().astype(dtype)
                elif one_hot:
                    values = column.astype(str).to_numpy()
                else:
                    raise ValueError(f"Covariate '{cov}' is not numeric; pass one_hot=True to one-hot encode it.")
                covariate_list.append(pd.DataFrame({cov: values}, index=phenotype_df.columns))
            elif cov in dd.D.obsm:
                cov_matrix = asarray(dd.D.obsm[cov]).astype(dtype)
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
                raise ValueError(f"Covariate '{cov}' not found in dd.D.obs or dd.D.obsm.")

        covariates_df = (
            pd.concat(covariate_list, axis=1) if covariate_list else pd.DataFrame(index=phenotype_df.columns)
        )
        missing = covariates_df.columns[covariates_df.isna().any()].tolist()
        if missing:
            raise ValueError(
                f"Covariate(s) {missing} are missing for some donors; drop those donors or leave them out."
            )
        covariates_df.index.name = "iid"
        covariates_df.to_csv(covar, sep="\t")

    geno_ext = (".bed", ".bim", ".fam")
    if not all(os.path.isfile(prefix + ext) for ext in geno_ext) or overwrite_plink_export:
        to_plink(dd.D, prefix, **plink_export_kwargs)

    written = [pheno, covar, *(prefix + ext for ext in geno_ext)]
    keep_file = exclude_file = None
    if keep is not None:
        keep_file, new = _id_file(keep, f"{prefix}_keep.txt")
        written += [keep_file] if new else []
    if exclude is not None:
        exclude_file, new = _id_file(exclude, f"{prefix}_exclude.txt")
        written += [exclude_file] if new else []

    if libsize_key is not None:
        totals = pd.to_numeric(dd.C.obs[libsize_key], errors="coerce")
        libsize = (
            totals.groupby(dd.C.obs[dd.donor_id].astype(str).to_numpy()).sum().reindex(phenotype_df.columns.astype(str))
        )
        if libsize.isna().any() or (libsize <= 0).any():
            raise ValueError(f"dd.C.obs[{libsize_key!r}] gives no positive library size for some donors.")
        libsize_file = f"{prefix}_libsize.tsv"
        pd.DataFrame({"iid": phenotype_df.columns, "libsize": libsize.to_numpy()}).to_csv(
            libsize_file, sep="\t", index=False
        )
        written.append(libsize_file)
        if count_model and offset is None and offset_name_from_covar is None and not set_offset_from_libsize:
            offset = f"{prefix}_offset.tsv"
            offsets = pd.DataFrame({"iid": phenotype_df.columns, "offset": np.log(libsize.to_numpy())})
            offsets.to_csv(offset, sep="\t", index=False)
            written.append(offset)
        if pcs_libsize is None and pcs_libsize_name_from_covar is None and pcs_normalization != "none":
            pcs_libsize = libsize_file

    runtime = []
    if seed is not None:
        runtime += ["--seed", str(seed)]
    if platform is not None:
        runtime += ["--platform", platform]
    if verbose:
        runtime.append("--verbose")
    samples = []
    if keep_file is not None:
        samples += ["--keep", keep_file]
    if exclude_file is not None:
        samples += ["--exclude", exclude_file]
    if min_indiv_expr_pct is not None:
        samples += ["--min-indiv-expr-pct", str(min_indiv_expr_pct)]

    commands = []
    if expression_pcs:
        covar_with_pcs = f"{prefix}_donor_features_pcs.tsv"
        pcs_cmd = ["jaxqtl", "compute-pcs", "--pheno", pheno, "--covar", covar, "--num-pcs", str(expression_pcs)]
        pcs_cmd += ["--normalization", pcs_normalization, "--transform", pcs_transform, *samples]
        if pcs_libsize is not None:
            pcs_cmd += ["--libsize", pcs_libsize]
        if pcs_libsize_name_from_covar is not None:
            pcs_cmd += ["--libsize-name-from-covar", pcs_libsize_name_from_covar]
        pcs_cmd += [*runtime, "--out", covar_with_pcs]
        commands.append(pcs_cmd)
        written += [covar_with_pcs, f"{covar_with_pcs}.variance.tsv", f"{covar_with_pcs}.log"]
        covar = covar_with_pcs

    cmd = ["jaxqtl", mode, "--bfile", prefix, "--pheno", pheno, "--covar", covar]

    def add_opt(flag, value):
        if value is not None:
            cmd.extend([flag, *map(str, value)] if isinstance(value, list | tuple) else [flag, str(value)])

    def add_flag(flag, enabled):
        if enabled:
            cmd.append(flag)

    add_opt("--model", model)
    add_opt("--test", test)
    add_flag("--robust-se", robust_se)
    add_flag("--spa", spa)
    add_flag("--acat", acat)
    if mode == "cis" and not acat:
        add_opt("--nperm", nperm)
    add_opt("--window", window)
    add_flag("--tss-centered", tss_centered)
    add_opt("--covar-name", _as_list(covar_name))
    add_opt("--rm-covar", _as_list(rm_covar))
    add_flag("--normalize-covar", normalize_covar)
    add_flag("--one-hot", one_hot)
    add_flag("--no-intercept", no_intercept)
    add_opt("--offset", offset)
    add_opt("--offset-name-from-covar", offset_name_from_covar)
    add_flag("--set-offset-from-libsize", set_offset_from_libsize)
    cmd.extend(samples)
    add_opt("--min-gene-expr-pct", min_gene_expr_pct)
    add_opt("--chr", chrom)
    add_opt("--genes", _as_list(genes))
    add_opt("--gene-list", gene_list)
    add_opt("--rm-genes", _as_list(rm_genes))
    add_opt("--exclude-gene-list", exclude_gene_list)
    add_opt("--maf", maf)
    add_opt("--max-iter", max_iter)
    add_opt("--tol", tol)
    add_opt("--gtol", gtol)
    add_opt("--step-size", step_size)
    add_opt("--solver", solver)
    cmd.extend(runtime)
    add_opt("--out", out)
    if extra_args:
        cmd.extend(map(str, extra_args))
    commands.append(cmd)
    command_str = " && ".join(shlex.join(c) for c in commands)

    if not run:
        if save_cmd_file:
            with open(save_cmd_file, "w") as f:
                f.write(command_str + "\n")
        return command_str

    for c in commands:
        logger.info(f"Running: {shlex.join(c)}")
        subprocess.run(c, check=True)

    test_name = _test_name(test, spa, model)
    if read_results:
        results = read_jaxqtl_results(out, mode=mode, test=test_name, acat=acat)
    else:
        outputs = [*_result_paths(out, mode, test_name, acat), f"{out}.log"]
        results = [p for p in outputs if os.path.isfile(p)]

    if remove_intermediate_files:
        for filename in written:
            if os.path.isfile(filename):
                os.remove(filename)

    return results
