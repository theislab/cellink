import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_PHENOTYPE_ORDER = ["scz", "bip", "adhd", "alz", "anorexia", "tourette"]


def enrichment_scatter(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    phenotype_col: str = "phenotype",
    neglog10p_col: str = "neglog10p",
    phenotype_order: list[str] | None = None,
    title: str = "",
    figsize: tuple | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 150,
    ax=None,
) -> Figure:
    """
    Scatter plot of -log10(p) enrichments per cell type × phenotype.

    One dot per (cell type, phenotype) pair.  Dots are coloured and enlarged when
    they exceed the per-cell-type Bonferroni threshold; all others are grey.  Two
    dashed reference lines mark the per-CT and the per-CT × phenotype Bonferroni
    thresholds.  Significant cell types are listed in a legend outside the plot.

    Works identically for LDSC cell-type results (``Coefficient_P_value`` converted
    to -log10) and MAGMA GSA / GPA results (``P`` column converted to -log10) —
    the caller is responsible for building the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``cell_type_col``, ``phenotype_col``, and ``neglog10p_col``.
    cell_type_col : str, default "cell_type"
        Column containing cell-type labels.
    phenotype_col : str, default "phenotype"
        Column containing phenotype labels.
    neglog10p_col : str, default "neglog10p"
        Column containing -log10(p) values.
    phenotype_order : list of str, optional
        Preferred left-to-right ordering of phenotypes.  Phenotypes not in this
        list are appended alphabetically.  Defaults to a built-in psychiatric
        disease order (scz, bip, adhd, alz, anorexia, tourette).
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.  Defaults to ``(max(10, n_phenotypes * 1.8), 9)``.
    show : bool or None, optional
        Whether to call ``plt.show()``.  Defaults to True when no ``ax`` is provided.
    save : str or bool or None, optional
        Path to save the figure.  If True, saves as ``"enrichment_scatter.pdf"``.
    dpi : int, default 150
        Resolution for saved figures.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cell_types = sorted(df[cell_type_col].unique())
    n_ct = len(cell_types)

    order = phenotype_order if phenotype_order is not None else _PHENOTYPE_ORDER
    present = df[phenotype_col].unique()
    phenotypes = [p for p in order if p in present] + sorted(p for p in present if p not in order)
    n_pheno = len(phenotypes)

    if n_ct == 0 or n_pheno == 0:
        logger.warning("enrichment_scatter: no cell types or phenotypes found.")
        fig, ax_ = plt.subplots()
        return fig

    bonf_ct = -np.log10(0.05 / n_ct)
    bonf_ct_pheno = -np.log10(0.05 / (n_ct * n_pheno))

    sig_cts = {ct for ct in cell_types if (df.loc[df[cell_type_col] == ct, neglog10p_col] >= bonf_ct).any()}
    n_sig = len(sig_cts)
    palette = sns.color_palette("tab20", max(n_sig, 1)) if n_sig <= 20 else sns.husl_palette(n_sig, s=0.9, l=0.55)
    ct_color = dict(zip(sorted(sig_cts), palette, strict=False))
    grey = (0.75, 0.75, 0.75)

    offsets = np.linspace(-0.3, 0.3, n_ct)
    ct_offset = {ct: offsets[i] for i, ct in enumerate(cell_types)}
    pheno_idx = {p: i for i, p in enumerate(phenotypes)}

    _figsize = figsize or (max(10, n_pheno * 1.8), 9)
    if ax is None:
        fig, ax_ = plt.subplots(figsize=_figsize)
    else:
        ax_ = ax
        fig = ax_.figure

    lookup = df.set_index([phenotype_col, cell_type_col])[neglog10p_col]

    for ct in cell_types:
        color = ct_color.get(ct, grey)
        for pheno in phenotypes:
            try:
                y = float(lookup.loc[(pheno, ct)])
            except KeyError:
                continue
            x = pheno_idx[pheno] + ct_offset[ct]
            is_sig = y >= bonf_ct
            ax_.scatter(
                x,
                y,
                s=90 if is_sig else 30,
                color=color if is_sig else grey,
                alpha=1.0,
                edgecolor="black" if is_sig else None,
                linewidth=0.4 if is_sig else 0,
                zorder=3 if is_sig else 2,
            )

    ax_.axhline(bonf_ct, color="grey", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_.axhline(bonf_ct_pheno, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

    ax_.set_xticks(np.arange(n_pheno))
    ax_.set_xticklabels(phenotypes, fontsize=14, rotation=45, ha="right")
    ax_.set_xlabel("Phenotype", fontsize=15)
    ax_.set_ylabel("-log₁₀(P)", fontsize=15)
    if title:
        ax_.set_title(title, fontsize=16)
    ax_.tick_params(axis="y", labelsize=12)

    legend_handles = [
        Line2D([0], [0], color="grey", linestyle="--", label=f"Bonf. per CT  (n={n_ct})"),
        Line2D([0], [0], color="red", linestyle="--", label=f"Bonf. CT×pheno  (n={n_ct * n_pheno})"),
    ]
    for ct in sorted(sig_cts):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=ct_color[ct],
                markeredgecolor="k",
                markersize=7,
                label=ct,
            )
        )
    ax_.legend(
        handles=legend_handles,
        title="Significant cell types",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=11,
        title_fontsize=12,
        frameon=False,
        markerscale=1.5,
    )

    plt.tight_layout()

    if save:
        path = save if isinstance(save, str) else "enrichment_scatter.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig


def method_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Method A",
    label_b: str = "Method B",
    cell_type_col: str = "cell_type",
    phenotype_col: str = "phenotype",
    neglog10p_col: str = "neglog10p",
    n_cols: int = 6,
    subplot_size: float = 3.2,
    annotate_points: bool = True,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 150,
) -> Figure:
    """
    Compare -log10(p) values between two methods across shared cell types.

    Produces a grid of scatter plots — one panel per shared cell type.  Within
    each panel, each point represents a phenotype; its x-coordinate is the
    -log10(p) from ``df_a`` and its y-coordinate from ``df_b``.  A Spearman
    correlation coefficient and p-value are shown in each panel title.

    Parameters
    ----------
    df_a : pd.DataFrame
        Results for method A.  Must contain ``cell_type_col``, ``phenotype_col``,
        and ``neglog10p_col``.
    df_b : pd.DataFrame
        Results for method B.  Same schema as ``df_a``.
    label_a : str, default "Method A"
        Axis label and title component for method A.
    label_b : str, default "Method B"
        Axis label and title component for method B.
    cell_type_col : str, default "cell_type"
        Column containing cell-type labels (must be the same name in both DataFrames).
    phenotype_col : str, default "phenotype"
        Column containing phenotype labels.
    neglog10p_col : str, default "neglog10p"
        Column containing -log10(p) values.
    n_cols : int, default 6
        Number of columns in the subplot grid.
    subplot_size : float, default 3.2
        Width and height (inches) of each individual subplot.
    annotate_points : bool, default True
        Whether to annotate each point with its phenotype name (first 6 characters).
    title : str, optional
        Overall figure title.  Defaults to ``"<label_a>  vs  <label_b>"``.
    show : bool or None, optional
        Whether to call ``plt.show()``.  Defaults to True when no ``save`` path given.
    save : str or bool or None, optional
        Path to save the figure.  If True, saves as ``"method_comparison.pdf"``.
    dpi : int, default 150
        Resolution for saved figures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot_a = df_a.pivot_table(
        index=phenotype_col,
        columns=cell_type_col,
        values=neglog10p_col,
        aggfunc="mean",
    )
    pivot_b = df_b.pivot_table(
        index=phenotype_col,
        columns=cell_type_col,
        values=neglog10p_col,
        aggfunc="mean",
    )

    shared_cts = sorted(set(pivot_a.columns) & set(pivot_b.columns))
    shared_phenos = sorted(set(pivot_a.index) & set(pivot_b.index))

    if not shared_cts or not shared_phenos:
        logger.warning("method_comparison: no shared cell types or phenotypes between the two DataFrames.")
        fig, _ = plt.subplots()
        return fig

    ncols = min(n_cols, len(shared_cts))
    nrows = math.ceil(len(shared_cts) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * subplot_size, nrows * subplot_size),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, ct in enumerate(shared_cts):
        ax = axes_flat[i]
        both = (
            pd.DataFrame(
                {
                    "x": pivot_a.loc[shared_phenos, ct],
                    "y": pivot_b.loc[shared_phenos, ct],
                }
            )
            .dropna()
            .reset_index()
        )
        both.columns = [phenotype_col, "x", "y"]

        if len(both) < 3:
            ax.text(0.5, 0.5, "n<3", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(ct[:35], fontsize=7)
            continue

        r, pval = spearmanr(both["x"], both["y"])
        ax.scatter(both["x"], both["y"], s=40, alpha=0.8, color="steelblue", edgecolors="none", zorder=3)

        if annotate_points:
            for _, row in both.iterrows():
                ax.annotate(
                    str(row[phenotype_col])[:6],
                    xy=(row["x"], row["y"]),
                    fontsize=5,
                    alpha=0.7,
                    xytext=(2, 2),
                    textcoords="offset points",
                )

        lo = min(both["x"].min(), both["y"].min())
        hi = max(both["x"].max(), both["y"].max())
        ax.plot([lo, hi], [lo, hi], color="grey", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

        p_str = f"{pval:.1e}" if pval < 0.01 else f"{pval:.2f}"
        ax.set_title(
            f"{ct[:35]}\n$r_s$={r:.2f}  p={p_str}",
            fontsize=7,
        )
        ax.set_xlabel(label_a, fontsize=6)
        ax.set_ylabel(label_b, fontsize=6)
        ax.tick_params(labelsize=5)

    for j in range(len(shared_cts), len(axes_flat)):
        axes_flat[j].set_visible(False)

    suptitle = title or f"{label_a}  vs  {label_b}"
    fig.suptitle(
        f"{suptitle}\n({len(shared_phenos)} phenotypes × {len(shared_cts)} cell types)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    if save:
        path = save if isinstance(save, str) else "method_comparison.pdf"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    elif show or (show is None and save is None):
        plt.show()

    return fig
