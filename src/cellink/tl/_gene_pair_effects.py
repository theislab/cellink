from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["compare_gene_pair_effects"]


def compare_gene_pair_effects(
    eqtl_root: str | Path,
    variant_id: str,
    gene_a: str,
    gene_b: str,
    cohort: str,
    celltypes: list[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side eQTL effect of one variant on a pair of genes, across every celltype.

    For each celltype directory under ``{eqtl_root}/{cohort}/``, reads that
    celltype's own pre-computed TensorQTL nominal results
    (``{eqtl_root}/{cohort}/{celltype}/tensorqtl.parquet``, columns
    ``gene``, ``variant_id``, ``beta``, ``se``, ``pval``), and pulls out
    ``gene_a``'s and ``gene_b``'s own (beta, se, pval) at ``variant_id``.
    Only celltypes where *both* genes have a row at that variant are kept,
    so the result is directly comparable pairwise: useful for checking
    whether two genes (e.g. paralogs sharing a regulatory variant) show a
    same-signed or opposite-signed effect at a shared variant, across every
    celltype at once.

    Does no model-fitting of its own; this is pure post-processing of
    already-computed TensorQTL output (e.g. from ``run_tensorqtl(mode=
    "cis_nominal")``).

    Parameters
    ----------
    eqtl_root : str or Path
        Root directory of pre-computed eQTL results, containing one
        subdirectory per cohort.
    variant_id : str
        The variant to compare both genes' effects at, matched exactly
        against the ``variant_id`` column of each celltype's parquet.
    gene_a : str
        First gene ID (e.g. Ensembl ID). Appears first within each
        celltype's two rows in the returned DataFrame.
    gene_b : str
        Second gene ID, compared against ``gene_a`` at the same variant.
    cohort : str
        Cohort name; results are read from ``{eqtl_root}/{cohort}/``.
    celltypes : list of str, optional
        Celltypes to scan. If None (default), every subdirectory of
        ``{eqtl_root}/{cohort}/`` is scanned.

    Returns
    -------
    pd.DataFrame
        Columns ``gene``, ``variant_id``, ``beta``, ``se``, ``pval``,
        ``celltype``; two rows (``gene_a`` then ``gene_b``) per celltype
        where both genes have a row at ``variant_id``. Empty (same columns,
        zero rows) if no celltype has both.

    Raises
    ------
    FileNotFoundError
        If ``celltypes`` is None and ``{eqtl_root}/{cohort}/`` does not exist.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> from cellink.tl import compare_gene_pair_effects
    >>> tmp = tempfile.TemporaryDirectory()
    >>> root = Path(tmp.name)
    >>> celltype_dir = root / "ukb_european" / "NK_CD16"
    >>> celltype_dir.mkdir(parents=True)
    >>> pd.DataFrame(
    ...     {
    ...         "gene": ["ENSG_A", "ENSG_B", "ENSG_A"],
    ...         "variant_id": ["1:100:A:G", "1:100:A:G", "1:200:A:G"],
    ...         "beta": [0.3, -0.25, 0.1],
    ...         "se": [0.05, 0.06, 0.05],
    ...         "pval": [1e-8, 1e-6, 0.2],
    ...     }
    ... ).to_parquet(celltype_dir / "tensorqtl.parquet")
    >>> res = compare_gene_pair_effects(root, "1:100:A:G", "ENSG_A", "ENSG_B", cohort="ukb_european")
    >>> list(res["gene"])
    ['ENSG_A', 'ENSG_B']
    >>> tmp.cleanup()
    """
    eqtl_root = Path(eqtl_root)
    cohort_dir = eqtl_root / cohort

    if celltypes is None:
        if not cohort_dir.is_dir():
            raise FileNotFoundError(f"cohort directory not found: {cohort_dir}")
        celltypes = sorted(p.name for p in cohort_dir.iterdir() if p.is_dir())

    genes = [gene_a, gene_b]
    gene_order = {g: i for i, g in enumerate(genes)}
    columns = ["gene", "variant_id", "beta", "se", "pval", "celltype"]

    rows = []
    for celltype in celltypes:
        path = cohort_dir / celltype / "tensorqtl.parquet"
        if not path.exists():
            logger.info(f"compare_gene_pair_effects: skipping {celltype!r}, {path} not found.")
            continue

        df = pd.read_parquet(path, columns=["gene", "variant_id", "beta", "se", "pval"])
        sub = df[(df["variant_id"] == variant_id) & (df["gene"].isin(genes))].copy()
        if sub["gene"].nunique() < len(genes):
            logger.info(
                f"compare_gene_pair_effects: skipping {celltype!r}, only "
                f"{sub['gene'].nunique()} of {len(genes)} genes have a row at variant_id={variant_id!r}."
            )
            continue

        sub["celltype"] = celltype
        sub["_gene_order"] = sub["gene"].map(gene_order)
        sub = sub.sort_values("_gene_order").drop(columns="_gene_order")
        rows.append(sub[columns])

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)
