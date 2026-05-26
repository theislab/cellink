import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def _resolve_gene_map(gene_map: "str | Path | pd.Series | None") -> "pd.Series | None":
    """Return a Series indexed by gene symbol with ENSG values, or None."""
    if gene_map is None:
        return None
    if isinstance(gene_map, pd.Series):
        return gene_map
    return pd.read_csv(gene_map, sep="\t").set_index("gene_name")["ensg_id"]


def _to_ensg(index: pd.Index, gmap: "pd.Series | None") -> pd.Index:
    """Map an index through gmap (if provided) and keep only ENSG IDs."""
    idx = index.astype(str)
    if gmap is not None:
        idx = pd.Index([gmap.get(g, g) for g in idx])
    return idx


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def load_ensembl_to_entrez_map(map_tsv: str | Path) -> pd.Series:
    """
    Load a mapping TSV with columns:
      ensembl_gene_id   entrez_id
    Returns a Series indexed by ENSG (upper, no version) with values as string Entrez IDs.
    """
    map_tsv = Path(map_tsv)
    df = pd.read_csv(map_tsv, sep="\t", dtype=str)

    # Accept a few common header variants
    col_ens = None
    for c in ["ensembl_gene_id", "ENSG", "ensembl", "gene_id"]:
        if c in df.columns:
            col_ens = c
            break
    if col_ens is None:
        raise ValueError(f"Mapping file missing Ensembl column. Found: {list(df.columns)}")

    col_ent = None
    for c in ["entrez_id", "entrezgene", "entrez", "ENTREZID", "ncbi_gene_id"]:
        if c in df.columns:
            col_ent = c
            break
    if col_ent is None:
        raise ValueError(f"Mapping file missing Entrez column. Found: {list(df.columns)}")

    ens = df[col_ens].astype(str).str.strip().str.upper().str.replace(r"\..*$", "", regex=True)
    ent = df[col_ent].astype(str).str.strip()

    m = pd.Series(ent.values, index=ens.values)
    m = m[~m.index.duplicated(keep="first")]
    return m


def genesets_dir_to_entrez_gmt(
    *,
    geneset_dir: str | Path = "ldsc_genesets",
    out_gmt: str | Path | None = None,
    ensembl_to_entrez_tsv: str | Path | None = None,
    pattern: str = "*.GeneSet",
    description: str = "S-LDSC derived gene set",
    include_control: bool = False,
    control_name: str = "Control",
    remove_version_suffix: bool = True,
    uppercase: bool = True,
    min_genes: int = 1,
    sort_genes: bool = False,
    dedup_genes: bool = True,
    drop_unmapped: bool = True,
    allow_mygene_fallback: bool = False,
    species: str = "human",
    output_basename: str = "genesets.gmt",
) -> Path:
    """
    Convert *.GeneSet -> MAGMA .gmt without ID conversion.

    Defaults:
      - reads from ./ldsc_genesets
      - writes to a sibling directory ./magma_genesets/genesets.gmt
        (magma_genesets is created if needed)
    """
    geneset_dir = Path(geneset_dir).resolve()

    # ---- Default output location: sibling magma_genesets next to ldsc_genesets ----
    if out_gmt is None:
        magma_dir = geneset_dir.parent / "magma_genesets"
        magma_dir.mkdir(parents=True, exist_ok=True)
        out_gmt = magma_dir / output_basename
    else:
        out_gmt = Path(out_gmt)
        out_gmt.parent.mkdir(parents=True, exist_ok=True)

    # ---- Find input GeneSet files ----
    files = sorted(geneset_dir.glob(pattern))
    if not include_control:
        files = [p for p in files if p.stem != control_name]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {geneset_dir}")

    if ensembl_to_entrez_tsv is not None:
        logger.warning("`ensembl_to_entrez_tsv` is ignored. No Ensembl→Entrez conversion is performed.")
    if allow_mygene_fallback:
        logger.warning("`allow_mygene_fallback` is ignored. No Ensembl→Entrez conversion is performed.")
    if not drop_unmapped:
        logger.warning("`drop_unmapped=False` has no effect when conversion is disabled.")
    if species != "human":
        logger.warning("`species` is ignored. No Ensembl→Entrez conversion is performed.")

    def norm_ens(g: str) -> str:
        g = str(g).strip()
        if remove_version_suffix:
            g = g.split(".", 1)[0]
        if uppercase:
            g = g.upper()
        return g

    def prepare_genes(genes: list[str]) -> list[str]:
        """Normalize and optionally deduplicate/sort gene IDs without conversion."""
        genes = [norm_ens(g) for g in genes if str(g).strip()]

        if dedup_genes:
            seen = set()
            genes = [g for g in genes if not (g in seen or seen.add(g))]
        if sort_genes:
            genes = sorted(genes)

        return genes

    n_written = 0
    n_skipped = 0

    with Path(out_gmt).open("w", encoding="utf-8") as out:
        for fp in files:
            set_name = fp.stem

            with fp.open("r", encoding="utf-8") as f:
                genes = [line.strip() for line in f if line.strip()]

            output_genes = prepare_genes(genes)

            if len(output_genes) < min_genes:
                logger.warning(f"Skipping {fp.name}: only {len(output_genes)} genes after normalization")
                n_skipped += 1
                continue

            row = [set_name, description] + output_genes
            out.write("\t".join(row) + "\n")
            n_written += 1

    logger.info(f"Wrote {n_written} gene sets to {out_gmt} (skipped {n_skipped})")
    return Path(out_gmt)


# ---------------------------------------------------------------------------
# LDSC scores → MAGMA input files
# ---------------------------------------------------------------------------

def scores_to_gmt(
    scores: pd.DataFrame,
    out_file: "str | Path",
    top_frac: float = 0.10,
    ascending: bool = False,
    gene_map: "str | Path | pd.Series | None" = None,
    set_name_prefix: str = "",
    min_genes: int = 1,
) -> Path:
    """
    Convert a genes × cell-types score DataFrame to MAGMA GMT format.

    Each cell type becomes one gene set containing the top (or bottom)
    ``top_frac`` fraction of genes ranked by score.  This is the Python-API
    equivalent of ``make_magma_gmt.py`` in the combined_pipeline.

    GMT format: ``set_name\\tNA\\tGENE1\\tGENE2\\t...`` (one line per set).

    Parameters
    ----------
    scores
        DataFrame with genes as rows and cell types as columns.  Index should
        contain gene identifiers (ENSG IDs or gene symbols).
    out_file
        Output ``.gmt`` file path.
    top_frac
        Fraction of genes to select per cell type (default 0.10 = top 10%).
    ascending
        If ``True``, select the bottom ``top_frac`` genes (lowest scores)
        instead of the top.
    gene_map
        Optional mapping from gene symbols to ENSG IDs.  Accepts:

        * ``str`` or ``Path`` — path to a two-column TSV with headers
          ``gene_name`` and ``ensg_id``.
        * ``pd.Series`` — index = gene symbol, values = ENSG ID.

        If provided, non-ENSG index entries are translated and rows that still
        do not look like ENSG IDs after mapping are dropped.
    set_name_prefix
        String prepended to every set name (e.g. ``"brainscope_scz_seismic_top"``).
    min_genes
        Minimum number of genes required to write a gene set.  Sets with fewer
        genes are skipped with a warning.

    Returns
    -------
    Path
        Path to the written GMT file.

    Examples
    --------
    >>> from cellink.tl.external import scores_to_gmt, run_magma_gsa
    >>> gmt = scores_to_gmt(specificity_df, "ExcL23_top10.gmt",
    ...                     set_name_prefix="brainscope_seismic_top")
    >>> run_magma_gsa(gene_results="scz.genes.raw", set_annot=str(gmt),
    ...               out_prefix="results/scz_gsa")

    See Also
    --------
    genesets_dir_to_entrez_gmt : Convert LDSC .GeneSet files → GMT.
    scores_to_covar : Convert scores to MAGMA gene property covariate file.
    run_magma_gsa : Run MAGMA gene-set analysis with the GMT.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    gmap = _resolve_gene_map(gene_map)
    n_genes = len(scores)
    n_select = max(1, int(n_genes * top_frac))
    selection = "bottom" if ascending else "top"

    n_written = n_skipped = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for ct in scores.columns:
            col = scores[ct].dropna()
            if len(col) == 0:
                continue

            # Rank genes
            selected_idx = col.nsmallest(n_select).index if ascending else col.nlargest(n_select).index

            # Map to ENSG and filter
            gene_ids = list(_to_ensg(selected_idx, gmap))
            gene_ids = [g for g in gene_ids if g.upper().startswith("ENSG")]

            if len(gene_ids) < min_genes:
                logger.warning("Skipping %s: only %d ENSG genes after mapping", ct, len(gene_ids))
                n_skipped += 1
                continue

            set_name = _safe_name(f"{set_name_prefix}_{ct}" if set_name_prefix else ct)
            fh.write("\t".join([set_name, "NA"] + gene_ids) + "\n")
            n_written += 1

    logger.info(
        "scores_to_gmt: wrote %d gene sets (%s %d%%, %d skipped) → %s",
        n_written, selection, int(top_frac * 100), n_skipped, out_file,
    )
    return out_file


def scores_to_covar(
    scores: pd.DataFrame,
    out_file: "str | Path",
    gene_map: "str | Path | pd.Series | None" = None,
    negate: bool = False,
) -> Path:
    """
    Convert a genes × cell-types score DataFrame to a MAGMA gene covariate file.

    All genes with at least one non-NaN score are included (no top/bottom
    threshold).  This is the Python-API equivalent of ``make_magma_covar.py``
    in the combined_pipeline.

    The covariate file is tab-delimited with ``GENE`` as the index name, one
    column per cell type, and ``NA`` for missing values:

    .. code-block:: text

        GENE          ExcL2-3   ExcL4   InhSST
        ENSG00001234  0.52      0.01    0.08
        ENSG00005678  0.11      0.43    0.22

    Parameters
    ----------
    scores
        DataFrame with genes as rows and cell types as columns.  Index should
        contain gene identifiers (ENSG IDs or gene symbols).
    out_file
        Output ``.covar`` file path.
    gene_map
        Optional mapping from gene symbols to ENSG IDs — same formats as in
        :func:`scores_to_gmt`.  Non-ENSG rows are dropped after mapping.
    negate
        If ``True``, multiply all scores by ``-1`` before writing.  Use this
        to test enrichment in genes with *low* scores (e.g. negative Vg).

    Returns
    -------
    Path
        Path to the written covariate file.

    Examples
    --------
    >>> from cellink.tl.external import scores_to_covar, run_magma_gpa
    >>> covar = scores_to_covar(specificity_df, "brainscope_seismic.covar")
    >>> run_magma_gpa(gene_results="scz.genes.raw", gene_covar=str(covar),
    ...               out_prefix="results/scz_gpa")

    See Also
    --------
    scores_to_gmt : Convert scores to GMT for gene-set analysis.
    run_magma_gpa : Run MAGMA gene property analysis with the covariate file.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    gmap = _resolve_gene_map(gene_map)

    df = scores.copy()
    df.index = _to_ensg(df.index, gmap)

    # Keep only ENSG rows
    df = df[df.index.str.upper().str.startswith("ENSG")]

    # Deduplicate: keep row with highest mean absolute score
    if df.index.duplicated().any():
        df["_mean_abs"] = df.abs().mean(axis=1)
        df = df.sort_values("_mean_abs", ascending=False)
        df = df[~df.index.duplicated(keep="first")].drop(columns=["_mean_abs"])

    df = df.dropna(how="all")

    if negate:
        df = -df

    df.columns = [_safe_name(c) for c in df.columns]
    df.index.name = "GENE"

    df.to_csv(out_file, sep="\t", na_rep="NA")
    logger.info("scores_to_covar: wrote %d genes × %d cell types → %s", len(df), df.shape[1], out_file)
    return out_file


# ---------------------------------------------------------------------------
# MAGMA steps I–III from scratch
# ---------------------------------------------------------------------------

def run_magma_annotate(
    snp_loc: str,
    gene_loc: str,
    out_prefix: str,
    magma_bin: str = "magma",
    window_kb: int = 0,
    run: bool = True,
    **kwargs,
) -> "dict[str, Any]":
    """
    Run MAGMA Step I — annotate SNPs to genes.

    Maps each SNP to the gene(s) whose transcribed region (± ``window_kb`` kb)
    overlaps its position.  Creates a ``.genes.annot`` file consumed by
    :func:`run_magma_gene_analysis`.

    Parameters
    ----------
    snp_loc
        Tab-delimited SNP location file with columns ``SNP``, ``CHR``, ``BP``.
        Can be derived from a GWAS summary statistics file.
    gene_loc
        NCBI/Ensembl gene location file.  MAGMA ships ``NCBI37.3.gene.loc``
        and ``NCBI38.gene.loc`` for GRCh37/38.  Format: ``ENTREZID CHR START
        END STRAND SYMBOL`` (tab-delimited, no header).
    out_prefix
        Prefix for output files.  Creates ``{out_prefix}.genes.annot``.
    magma_bin
        Path to the MAGMA binary (default: ``"magma"`` assumes it is on PATH).
    window_kb
        Flanking window in kb added around each gene's transcribed region
        (default 0 = gene body only).  Use e.g. 35 for ±35 kb.
    run
        If ``False``, return the command string without executing.
    **kwargs
        Additional flags passed verbatim to MAGMA as ``--key value``.

    Returns
    -------
    dict
        ``annot_file`` — path to ``.genes.annot``.
        ``files_created`` — list of output paths (if ``run=True``).
        ``command`` — command list (if ``run=False``).

    Examples
    --------
    >>> run_magma_annotate(
    ...     snp_loc="gwas_snps.txt",
    ...     gene_loc="NCBI37.3.gene.loc",
    ...     out_prefix="results/my_gwas",
    ...     window_kb=35,
    ... )

    See Also
    --------
    run_magma_gene_analysis : Step II — compute gene-level p-values.
    """
    cmd = [magma_bin, "--annotate"]
    if window_kb:
        cmd += ["window=" + str(window_kb)]
    cmd += ["--snp-loc", snp_loc, "--gene-loc", gene_loc, "--out", out_prefix]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    if not run:
        return {"command": cmd}

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    _run(cmd)
    annot_file = f"{out_prefix}.genes.annot"
    return {"annot_file": annot_file, "files_created": [annot_file]}


def run_magma_gene_analysis(
    bfile: str,
    pval_file: str,
    gene_annot: str,
    out_prefix: str,
    n_samples: "int | None" = None,
    magma_bin: str = "magma",
    run: bool = True,
    **kwargs,
) -> "dict[str, Any]":
    """
    Run MAGMA Step II — gene-level association analysis.

    Computes gene-level p-values and z-scores from GWAS SNP-level summary
    statistics, taking LD structure into account using a reference genotype
    panel.  The output ``{out_prefix}.genes.raw`` is the input to both
    :func:`run_magma_gsa` and :func:`run_magma_gpa`.

    Parameters
    ----------
    bfile
        PLINK bfile prefix (without extension) for the LD reference panel,
        e.g. ``"g1000_eur/g1000_eur"``.
    pval_file
        GWAS p-value file.  Must contain at least ``SNP`` and ``P`` columns
        (tab or space delimited).
    gene_annot
        Path to the ``.genes.annot`` file from :func:`run_magma_annotate`.
    out_prefix
        Prefix for output files.  Creates ``{out_prefix}.genes.raw`` and
        ``{out_prefix}.genes.out``.
    n_samples
        Total GWAS sample size.  Required unless ``pval_file`` contains an
        ``N`` column.  Passed as ``N=<value>`` in the ``--pval`` argument.
    magma_bin
        Path to the MAGMA binary.
    run
        If ``False``, return the command without executing.
    **kwargs
        Additional flags passed verbatim to MAGMA as ``--key value``.

    Returns
    -------
    dict
        ``gene_results`` — path to ``.genes.raw``.
        ``files_created`` — list of output paths (if ``run=True``).
        ``command`` — command list (if ``run=False``).

    Examples
    --------
    >>> run_magma_gene_analysis(
    ...     bfile="g1000_eur/g1000_eur",
    ...     pval_file="scz_gwas.txt",
    ...     gene_annot="results/my_gwas.genes.annot",
    ...     out_prefix="results/scz",
    ...     n_samples=67390,
    ... )

    See Also
    --------
    run_magma_annotate : Step I — SNP-to-gene annotation.
    run_magma_gsa : Step III — gene-set analysis.
    run_magma_gpa : Step III — gene property analysis.
    """
    pval_arg = pval_file
    if n_samples is not None:
        pval_arg = f"{pval_file} N={n_samples}"

    cmd = [
        magma_bin,
        "--bfile", bfile,
        "--pval", pval_arg,
        "--gene-annot", gene_annot,
        "--out", out_prefix,
    ]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    if not run:
        return {"command": cmd}

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    _run(cmd)
    gene_results = f"{out_prefix}.genes.raw"
    return {
        "gene_results": gene_results,
        "files_created": [gene_results, f"{out_prefix}.genes.out", f"{out_prefix}.log"],
    }


def run_magma_gsa(
    gene_results: str,
    set_annot: str,
    out_prefix: str,
    magma_bin: str = "magma",
    run: bool = True,
    **kwargs,
) -> "dict[str, Any]":
    """
    Run MAGMA Step III — gene-set analysis (GSA).

    Tests whether genes in each set have higher GWAS association signals than
    background genes, using the gene-level results from
    :func:`run_magma_gene_analysis`.  Input gene sets are supplied as a GMT
    file (see :func:`scores_to_gmt` or :func:`genesets_dir_to_entrez_gmt`).

    **LDSC → MAGMA GSA workflow**

    Two paths lead here from LDSC outputs:

    1. From continuous per-gene scores (e.g. specificity, SEISMIC):

       >>> gmt = scores_to_gmt(specificity_df, "top10.gmt")
       >>> run_magma_gsa(gene_results="scz.genes.raw", set_annot=str(gmt),
       ...               out_prefix="results/scz_gsa")

    2. From LDSC binary ``.GeneSet`` files:

       >>> gmt = genesets_dir_to_entrez_gmt(geneset_dir="ldsc_genesets",
       ...                                  out_gmt="magma_genesets/genesets.gmt")
       >>> run_magma_gsa(gene_results="scz.genes.raw", set_annot=str(gmt),
       ...               out_prefix="results/scz_gsa")

    Parameters
    ----------
    gene_results
        Path to ``.genes.raw`` from :func:`run_magma_gene_analysis`.
    set_annot
        Path to GMT-format gene-set file.  Gene identifiers must be ENSG or
        Entrez IDs consistent with those in ``gene_results``.
    out_prefix
        Prefix for output files.  Creates ``{out_prefix}.gsa.out``.
    magma_bin
        Path to the MAGMA binary.
    run
        If ``False``, return the command without executing.
    **kwargs
        Additional flags passed verbatim to MAGMA (e.g. ``model="multi"``).

    Returns
    -------
    dict
        ``results_file`` — path to ``.gsa.out``.
        ``files_created`` — list (if ``run=True``).
        ``command`` — command list (if ``run=False``).

    Examples
    --------
    >>> run_magma_gsa(
    ...     gene_results="results/scz.genes.raw",
    ...     set_annot="genesets/top10_seismic.gmt",
    ...     out_prefix="results/scz_gsa_seismic",
    ... )

    See Also
    --------
    scores_to_gmt : Build GMT from a scores DataFrame.
    genesets_dir_to_entrez_gmt : Build GMT from LDSC .GeneSet files.
    run_magma_gpa : Continuous gene property analysis (alternative to GSA).
    """
    cmd = [magma_bin, "--gene-results", gene_results, "--set-annot", set_annot, "--out", out_prefix]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    if not run:
        return {"command": cmd}

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    _run(cmd)
    results_file = f"{out_prefix}.gsa.out"
    return {"results_file": results_file, "files_created": [results_file, f"{out_prefix}.log"]}


def run_magma_gpa(
    gene_results: str,
    gene_covar: str,
    out_prefix: str,
    magma_bin: str = "magma",
    univariate: bool = False,
    run: bool = True,
    **kwargs,
) -> "dict[str, Any]":
    """
    Run MAGMA Step III — gene property analysis (GPA).

    Tests the linear association between continuous per-gene scores and GWAS
    gene-level z-scores.  Unlike GSA (which uses a top-N threshold), GPA uses
    all genes with scores as quantitative covariates.

    When ``univariate=False`` (default), all cell types are tested jointly in a
    single MAGMA call (``--gene-covar``).  This is efficient but MAGMA may drop
    highly collinear covariates.  Set ``univariate=True`` to test each cell
    type independently — this is slower but always produces a result for every
    cell type (matches ``run_magma_gpa_univariate.py`` in the pipeline).

    **LDSC → MAGMA GPA workflow**

    >>> covar = scores_to_covar(specificity_df, "brainscope_seismic.covar")
    >>> run_magma_gpa(gene_results="scz.genes.raw", gene_covar=str(covar),
    ...               out_prefix="results/scz_gpa_seismic")

    Parameters
    ----------
    gene_results
        Path to ``.genes.raw`` from :func:`run_magma_gene_analysis`.
    gene_covar
        Path to ``.covar`` file from :func:`scores_to_covar`.
    out_prefix
        Prefix for output files.  Creates ``{out_prefix}.gsa.out``.
    magma_bin
        Path to the MAGMA binary.
    univariate
        If ``True``, run each cell type (covariate column) as a separate MAGMA
        call and combine results.  Recommended when covariates are highly
        correlated (e.g. residual CV or negated scores), which causes MAGMA's
        collinearity filter to silently drop columns in joint mode.
    run
        If ``False``, return the command without executing (only valid when
        ``univariate=False``).
    **kwargs
        Additional flags forwarded to MAGMA in joint mode.

    Returns
    -------
    dict
        ``results_file`` — path to ``.gsa.out``.
        ``files_created`` — list (if ``run=True``).
        ``command`` — command list (if ``run=False``, joint mode only).

    Examples
    --------
    Joint analysis (default):

    >>> run_magma_gpa(
    ...     gene_results="results/scz.genes.raw",
    ...     gene_covar="covars/brainscope_seismic.covar",
    ...     out_prefix="results/scz_gpa_seismic",
    ... )

    Univariate analysis (safe for correlated scores):

    >>> run_magma_gpa(
    ...     gene_results="results/scz.genes.raw",
    ...     gene_covar="covars/brainscope_residual_cv.covar",
    ...     out_prefix="results/scz_gpa_residual_cv",
    ...     univariate=True,
    ... )

    See Also
    --------
    scores_to_covar : Build covariate file from a scores DataFrame.
    run_magma_gsa : Gene-set analysis (binary threshold, alternative to GPA).
    """
    out_prefix = str(out_prefix)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    if not univariate:
        cmd = [magma_bin, "--gene-results", gene_results, "--gene-covar", gene_covar, "--out", out_prefix]
        for k, v in kwargs.items():
            cmd += [f"--{k}", str(v)]

        if not run:
            return {"command": cmd}

        _run(cmd)
        results_file = f"{out_prefix}.gsa.out"
        return {"results_file": results_file, "files_created": [results_file, f"{out_prefix}.log"]}

    # -- Univariate mode: one MAGMA call per cell-type covariate column -------
    covar = pd.read_csv(gene_covar, sep="\t", index_col=0)
    cell_types = covar.columns.tolist()
    logger.info("GPA univariate: %d cell types", len(cell_types))

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for ct in cell_types:
            ct_covar = os.path.join(tmpdir, "ct.covar")
            ct_out   = os.path.join(tmpdir, "ct_out")
            covar[[ct]].to_csv(ct_covar, sep="\t", na_rep="NA")

            cmd = [magma_bin, "--gene-results", gene_results, "--gene-covar", ct_covar, "--out", ct_out]
            result = _run(cmd, check=False)
            if result.returncode != 0:
                logger.warning("MAGMA failed for %s: %s", ct, result.stderr.strip())
                continue

            gsa_file = ct_out + ".gsa.out"
            if not os.path.exists(gsa_file):
                logger.warning("No output for %s", ct)
                continue

            header = None
            with open(gsa_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if header is None:
                        header = parts
                        continue
                    row = dict(zip(header, parts))
                    row["FULL_NAME"] = ct
                    rows.append(row)
                    break

            logger.info("  %s: done", ct)

    results_file = f"{out_prefix}.gsa.out"
    with open(results_file, "w") as f:
        f.write("# UNIVARIATE GPA (each cell type tested independently)\n")
        f.write(
            f"# {'VARIABLE':<36} {'TYPE':<6} {'NGENES':>6} "
            f"{'BETA':>12} {'BETA_STD':>12} {'SE':>12} {'P':>12} FULL_NAME\n"
        )
        for row in rows:
            name = row.get("FULL_NAME", "")
            f.write(
                f"{name[:36]:<36} "
                f"{row.get('TYPE', 'COVAR'):<6} "
                f"{row.get('NGENES', 'NA'):>6} "
                f"{row.get('BETA', 'NA'):>12} "
                f"{row.get('BETA_STD', 'NA'):>12} "
                f"{row.get('SE', 'NA'):>12} "
                f"{row.get('P', 'NA'):>12} "
                f"{name}\n"
            )

    logger.info("GPA univariate: wrote %d cell types → %s", len(rows), results_file)
    return {"results_file": results_file, "files_created": [results_file]}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    p = argparse.ArgumentParser(description="Convert .GeneSet files to MAGMA .gmt without ID conversion")
    p.add_argument(
        "--geneset_dir",
        default="ldsc_genesets",
        help="Directory containing *.GeneSet (default: ldsc_genesets)",
    )
    p.add_argument(
        "--out_gmt",
        default=None,
        help="Optional output .gmt path. If omitted, writes to sibling magma_genesets/genesets.gmt",
    )
    p.add_argument(
        "--map_tsv",
        default=None,
        help="Deprecated and ignored (no Ensembl→Entrez conversion is performed).",
    )
    p.add_argument("--include_control", action="store_true")
    p.add_argument(
        "--allow_mygene_fallback",
        action="store_true",
        help="Deprecated and ignored (no Ensembl→Entrez conversion is performed).",
    )
    p.add_argument(
        "--pattern",
        default="*.GeneSet",
        help="Glob pattern for gene set files (default: *.GeneSet)",
    )
    p.add_argument(
        "--output_basename",
        default="genesets.gmt",
        help="Output filename when using default magma_genesets directory (default: genesets.gmt)",
    )
    args = p.parse_args()

    genesets_dir_to_entrez_gmt(
        geneset_dir=args.geneset_dir,
        out_gmt=args.out_gmt,
        ensembl_to_entrez_tsv=args.map_tsv,
        include_control=args.include_control,
        allow_mygene_fallback=args.allow_mygene_fallback,
        pattern=args.pattern,
        output_basename=args.output_basename,
    )
