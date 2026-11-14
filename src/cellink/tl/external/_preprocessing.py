import logging

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

MAX_LOG_NORMALIZED_VALUE = 50


def _nonzero_sample(X, n_values: int = 100_000) -> np.ndarray:
    """Up to ``n_values`` nonzero values of ``X``, from its first stored entries."""
    values = X.data[: 2 * n_values] if issparse(X) else np.asarray(X[: min(X.shape[0], 1000)]).ravel()
    return values[values != 0][:n_values]


def _looks_like_counts(X) -> bool:
    """Whether the nonzero values of ``X`` are positive integers, as raw counts are."""
    values = _nonzero_sample(X)
    return bool(values.size and np.all(values > 0) and np.allclose(values, np.round(values)))


def _filtered_copy(
    adata: AnnData,
    tool: str,
    min_genes: int,
    min_cells: int,
    cells: np.ndarray | None = None,
    layer: str | None = None,
    obsm_keys: tuple[str, ...] = (),
) -> tuple[AnnData, np.ndarray]:
    """Copy of the expression (``X`` or ``layer``), ``obs``, ``var`` and ``obsm_keys`` of ``adata``.

    Only ``cells`` (default: all) with at least ``min_genes`` expressed genes, and the genes expressed in
    at least ``min_cells`` of them, are copied; other layers, ``raw`` and ``obsp`` are not. Returns the
    copy and the mask of copied cells.
    """
    lean = AnnData(
        X=adata.X if layer is None else adata.layers[layer],
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns={"log1p": adata.uns["log1p"]} if layer is None and "log1p" in adata.uns else None,
        obsm={key: adata.obsm[key] for key in obsm_keys if key in adata.obsm},
    )

    def rows(mask):
        return lean if mask.all() else lean[mask]

    cell_mask = np.ones(adata.n_obs, dtype=bool) if cells is None else np.array(cells, dtype=bool)
    n_selected = cell_mask.sum()
    passing, n_genes = sc.pp.filter_cells(rows(cell_mask), min_genes=min_genes, inplace=False)
    cell_mask[cell_mask] = passing
    gene_mask, n_cells = sc.pp.filter_genes(rows(cell_mask), min_cells=min_cells, inplace=False)
    if not (passing.all() and gene_mask.all()):
        logger.warning(
            f"{tool}: removed {(~passing).sum()} of {n_selected} cells with fewer than {min_genes} expressed genes "
            f"and {(~gene_mask).sum()} of {adata.n_vars} genes expressed in fewer than {min_cells} cells."
        )

    filtered = lean[cell_mask, gene_mask].copy()
    filtered.obs["n_genes"] = n_genes[passing]
    filtered.var["n_cells"] = n_cells[gene_mask]
    return filtered, cell_mask


def _log_normalized_copy(
    adata: AnnData, tool: str, min_genes: int, min_cells: int, **kwargs
) -> tuple[AnnData, np.ndarray]:
    """:func:`_filtered_copy`, log-normalized by :func:`_log_normalize_from_values`."""
    filtered, cell_mask = _filtered_copy(adata, tool, min_genes, min_cells, **kwargs)
    _log_normalize_from_values(filtered, tool)
    return filtered, cell_mask


def _log_normalize_from_values(adata: AnnData, tool: str) -> None:
    """Make ``adata.X`` log-normalized expression, deciding from its values rather than ``adata.uns["log1p"]``.

    The ``log1p`` flag alone is not reliable: it stays set when raw counts are copied back into ``X``
    after normalizing, and it is missing from many published objects whose ``X`` is already
    log-normalized. Trusting it would pass raw counts to ``tool`` in the first case and normalize twice
    in the second. Every decision is logged as a warning, so the caller sees what ``tool`` was given.
    """
    if _looks_like_counts(adata.X):
        if "log1p" in adata.uns:
            logger.warning(
                f"{tool}: adata.uns['log1p'] is set, but the expression values are integer counts. "
                "Normalizing them to 10,000 counts per cell and log1p-transforming."
            )
            del adata.uns["log1p"]
        else:
            logger.warning(
                f"{tool}: the expression values are integer counts. "
                "Normalizing them to 10,000 counts per cell and log1p-transforming."
            )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return

    if "log1p" in adata.uns:
        logger.warning(f"{tool}: the expression values are not integer counts. Using them as log-normalized.")
    else:
        logger.warning(
            f"{tool}: the expression values are not integer counts and adata.uns['log1p'] is not set. "
            "Using them as log-normalized."
        )
    values = _nonzero_sample(adata.X)
    if values.size and values.min() < 0:
        logger.warning(
            f"{tool}: the expression values include negative numbers, as scaled or covariate-regressed data do. "
            f"{tool} expects log-normalized expression, which is non-negative."
        )
    if values.size and values.max() > MAX_LOG_NORMALIZED_VALUE:
        logger.warning(
            f"{tool}: the expression values reach {values.max():.0f}, too large for log-normalized expression; "
            "they may be normalized but not log-transformed."
        )
