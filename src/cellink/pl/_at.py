import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from scipy.stats import beta

from cellink._core.data_fields import VAnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def qq(
    pvals_df: pd.DataFrame,
    pval_col: str = "pv_adj",
    group_col: str = "group",
    figsize: tuple = None,
    labelsize: int = None,
    titlesize: int = None,
    n_cols: int = None,
    show_ci: bool = True,
    null_line_color: str = "purple",
    title: str = None,
    show: bool | None = None,
    save: str = None,
    dpi: int = 300,
    ax=None,
) -> Figure:
    """
    Generate quantile-quantile (QQ) plots of p-values, optionally grouped by a categorical column.

    Parameters
    ----------
    pvals_df : pandas.DataFrame
        DataFrame containing p-values and group labels.
    pval_col : str, default 'pb_adj'
        Column name in `pvals_df` containing p-values.
    group_col : str, default 'group'
        Column name in `pvals_df` used to group p-values for faceted QQ plots.
    figsize : tuple, optional
        Size of the figure. Defaults to matplotlib's `figure.figsize` setting.
    labelsize : int, optional
        Font size for axis labels. Defaults to matplotlib's `axes.labelsize`.
    titlesize : int, optional
        Font size for plot titles. Defaults to matplotlib's `axes.titlesize`.
    n_cols : int, optional
        Number of columns in the subplot grid. Only used if multiple groups are present.
    show_ci : bool, default True
        Whether to display 95% confidence interval bands on the QQ plot.
    title : str, optional
        Title for the plot when only one group is plotted.
    show : bool or None, optional
        Whether to display the plot. If None, shows the plot only when `ax` is not provided.
    save : str or None, optional
        Path to save the figure. If provided, the figure is saved instead of shown.
    dpi : int, default 300
        Resolution of the saved figure in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting QQ plot figure.

    Raises
    ------
    ValueError
        If specified `pval_col` or `group_col` is not in the DataFrame,
        or if no valid p-values remain after filtering.
    """
    figsize = figsize or plt.rcParams.get("figure.figsize", (4.8, 6.4))
    labelsize = labelsize or plt.rcParams.get("axes.labelsize", 12)
    titlesize = titlesize or plt.rcParams.get("axes.titlesize", 14)

    if pval_col not in pvals_df.columns:
        raise ValueError(f"pval_col '{pval_col}' not found in DataFrame columns")
    if group_col not in pvals_df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame columns")

    filtered = pvals_df.dropna(subset=[pval_col, group_col])
    filtered = filtered[(filtered[pval_col] > 0) & (filtered[pval_col] <= 1)]

    if filtered.empty:
        raise ValueError("No valid p-values after filtering.")

    groups = filtered[group_col].unique()
    n_groups = len(groups)

    def qq_single(ax, pvals, title=None, show_ci=True):
        pvals_sorted = np.sort(pvals)
        n = len(pvals_sorted)
        i = np.arange(1, n + 1)

        expected = -np.log10(i / (n + 1))
        observed = -np.log10(pvals_sorted)

        ax.scatter(expected, observed, edgecolor="k", facecolor="none", s=20, alpha=0.2)
        ax.axline([0, 0], slope=1, color=null_line_color, linestyle="--")

        if show_ci:
            lower = -np.log10(beta.ppf(0.975, i, n + 1 - i))
            upper = -np.log10(beta.ppf(0.025, i, n + 1 - i))
            ax.fill_between(expected, lower, upper, color="gray", alpha=0.3)

        ax.set_xlabel("expected -log10(p)", fontsize=labelsize)
        ax.set_ylabel("observed -log10(p)", fontsize=labelsize)
        if title:
            ax.set_title(title, fontsize=titlesize)
        ax.grid(True)

    if n_groups == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        qq_single(ax, filtered[pval_col], title=title, show_ci=show_ci)
    else:
        n_cols = n_cols or min(3, n_groups)
        n_rows = int(np.ceil(n_groups / n_cols))

        if ax is None:
            fig, ax = fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(figsize[0] * n_cols / 2, figsize[1] * n_rows / 2)
            )
        else:
            fig, axes = ax.figure
        axes = axes.flatten() if n_groups > 1 else [axes]

        for i, group in enumerate(groups):
            ax = axes[i]
            group_pvals = filtered.loc[filtered[group_col] == group, pval_col].values
            qq_single(ax, group_pvals, title=str(group), show_ci=show_ci)

        for j in range(n_groups, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

    if save:
        save_path = save if isinstance(save, str) else "qq.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig


def manhattan(
    pvals_df: pd.DataFrame,
    pval_col: str = "pv_adj",
    chromosome_col: str = VAnn.chrom,
    position_col: str = VAnn.pos,
    point_size: int = 10,
    significance_threshold: int = None,
    label_column: str = None,
    figsize: tuple = None,
    labelsize: int = None,
    titlesize: int = None,
    title: str = None,
    palette_mode: str = "alternating",
    palette_alternating_colors: list | np.ndarray = ["#E24E42", "#008F95"],
    significant_hit_color: str = "#D62728",
    palette: str = None,
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 300,
    ax=None,
) -> Figure:
    """
    Generate a Manhattan plot for genome-wide association p-values.

    Parameters
    ----------
    pvals_df : pandas.DataFrame
        DataFrame containing p-values and variant genomic positions.
    pval_col : str, default "pv_adj"
        Column in `pvals_df` with p-values.
    chromosome_col : str
        Column name for chromosome numbers (can be numeric or 'X', 'Y').
    position_col : str
        Column name for base-pair positions.
    point_size : int
        Size of points.
    significance_threshold : float, default 5e-8
        Threshold for highlighting significant hits.
    label_column : str or None
        Column to use for labeling significant points. Labels are shown above points.
    figsize : tuple, optional
        Size of the figure. Defaults to matplotlib’s `figure.figsize`.
    labelsize : int, optional
        Font size for axis labels. Defaults to matplotlib’s `axes.labelsize`.
    titlesize : int, optional
        Font size for the title. Defaults to matplotlib’s `axes.titlesize`.
    title : str, optional
        Title for the plot.
    palette_mode : {'alternating', None}, default 'alternating'
        Color mode for chromosomes if no `palette` is specified.
    palette_alternating_colors : list or np.ndarray, default ['#E24E42', '#008F95']
        List of two colors used to alternate chromosome coloring.
    significant_hit_color : str, default "#D62728"
        Color for highlighting significant hits.
    palette : str or None
        Seaborn color palette name. Overrides `palette_mode` if provided.
    show : bool or None
        Whether to display the plot. If None, shows the plot only when `ax` is not provided.
    save : str or bool or None
        File path to save the plot. If True, saves as "manhattan.png".
    dpi : int, default 300
        Resolution (dots per inch) of the saved figure.
    ax : matplotlib.axes.Axes or None
        Optional matplotlib Axes object to plot into.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting Manhattan plot figure.

    Raises
    ------
    ValueError
        If both `palette` and `palette_mode` are specified simultaneously.
    """
    if palette_mode is not None and palette is not None:
        raise ValueError("Both palette and palette_mode are set.")

    figsize = figsize or plt.rcParams.get("figure.figsize", (14, 6))
    labelsize = labelsize or plt.rcParams.get("axes.labelsize", 8)
    titlesize = titlesize or plt.rcParams.get("axes.titlesize", 14)

    pvals_df[chromosome_col] = pvals_df[chromosome_col].replace({"X": 23, "Y": 24}).astype(int)
    pvals_df = pvals_df.sort_values([chromosome_col, position_col])

    pvals_df["-log10(P)"] = -np.log10(pvals_df[pval_col])
    pvals_df["ind"] = range(len(pvals_df))
    pvals_df[chromosome_col] = pd.Categorical(
        pvals_df[chromosome_col], categories=sorted(pvals_df[chromosome_col].unique()), ordered=True
    )

    chr_max_bp = pvals_df.groupby(chromosome_col)[position_col].max().cumsum()
    chr_offsets = chr_max_bp.shift(fill_value=0)
    pvals_df["cum_pos"] = pvals_df.apply(lambda row: row[position_col] + chr_offsets[row[chromosome_col]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if palette is not None or (palette is None and palette_mode != "alternating"):
        if palette is None:
            palette = "tab20"
        palette = sns.color_palette(palette, n_colors=22)
        colors = {chr_num: palette[i % 22] for i, chr_num in enumerate(sorted(pvals_df[chromosome_col].unique()))}
    else:
        colors = {
            chr_num: palette_alternating_colors[i % 2]
            for i, chr_num in enumerate(sorted(pvals_df[chromosome_col].unique()))
        }

    x_labels = []
    x_labels_pos = []

    for i, (chr_num, group) in enumerate(pvals_df.groupby(chromosome_col)):
        color = colors[chr_num]
        ax.scatter(group["cum_pos"], group["-log10(P)"], color=color, s=point_size, label=f"Chr {chr_num}", zorder=2)

        ax.axvline(x=group["cum_pos"].iloc[-1], color="lightgrey", linestyle="--", lw=0.5)

        x_labels.append(str(chr_num))
        x_labels_pos.append(int(group["cum_pos"].min() + (group["cum_pos"].max() - group["cum_pos"].min()) / 2))

    if significance_threshold:
        sig_hits = pvals_df[pvals_df[pval_col] < significance_threshold]
        ax.scatter(
            sig_hits["cum_pos"],
            sig_hits["-log10(P)"],
            color=significant_hit_color,
            s=int(point_size * 1.5),
            edgecolor="k",
            label="Significant",
            zorder=3,
        )

        if label_column and label_column in pvals_df.columns and not sig_hits.empty:
            logger.info(f"Significant hits: {', '.join(sig_hits[label_column])}")
            texts = []
            for _, row in sig_hits.iterrows():
                x = row["cum_pos"]
                y = row["-log10(P)"]
                if np.isfinite(x) and np.isfinite(y):
                    texts.append(
                        ax.text(
                            x, y + 0.3, str(row[label_column]), fontsize=labelsize, ha="center", va="center", zorder=4
                        )
                    )
            ax.axhline(y=-np.log10(significance_threshold), color="grey", linestyle="--", lw=1)
        else:
            texts = []
    else:
        texts = []

    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels, fontsize=labelsize)
    ax.set_xlabel("chromosome")
    ax.set_ylabel("-log10(p-value)")
    if title:
        ax.set_title(title)

    min_x, max_x = pvals_df["cum_pos"].min(), pvals_df["cum_pos"].max()
    pad = (max_x - min_x) * 0.005  # 0.5% padding
    ax.set_xlim(pvals_df["cum_pos"].min(), pvals_df["cum_pos"].max())
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if len(texts) > 0:
        adjust_text(
            texts, ax=ax, only_move={"points": "y", "text": "y"}, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5)
        )

    if save:
        save_path = save if isinstance(save, str) else "manhattan.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig


def locus(
    locus_df: pd.DataFrame,
    gene_df: pd.DataFrame = None,
    pval_col: str = "pval",
    position_col: str = "pos",
    label_column: str = None,
    significance_threshold: float = None,
    highlight_color: str = "#D62728",
    point_size: int = 20,
    figsize: tuple = (10, 5),
    labelsize: int = 10,
    titlesize: int = 14,
    title: str = None,
    gene_color: str = "#1f77b4",
    gene_style: str = "box",  # or "bracket"
    show_genes: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 300,
    ax=None,
) -> Figure:
    """
    Generate a locus plot with SNP p-values and optional gene annotations.

    Parameters
    ----------
    locus_df : pandas.DataFrame
        DataFrame containing SNP positions and p-values.
    gene_df : pandas.DataFrame, optional
        DataFrame of gene annotations with columns: "start", "end", "gene", and optionally "strand".
    pval_col : str, default "pval"
        Column name in `locus_df` for p-values.
    position_col : str, default "pos"
        Column name in `locus_df` for genomic positions.
    label_column : str or None, optional
        Column in `locus_df` used to label significant SNPs.
    significance_threshold : float, optional
        Threshold below which SNPs are highlighted and optionally labeled.
    highlight_color : str, default "#D62728"
        Color used to highlight significant SNPs.
    point_size : int, default 20
        Size of scatter plot points.
    figsize : tuple, default (10, 5)
        Size of the figure (width, height) in inches.
    labelsize : int, default 10
        Font size for axis and annotation labels.
    titlesize : int, default 14
        Font size for the plot title.
    title : str, optional
        Title for the plot.
    gene_color : str, default "#1f77b4"
        Color used for gene annotations.
    gene_style : {'box', 'bracket'}, default "box"
        Style used for gene annotations. "box" draws rectangles, "bracket" draws lines with caps.
    show_genes : bool, default True
        Whether to include gene annotations beneath the plot.
    show : bool or None, optional
        Whether to display the plot. If None, shows only when `ax` is not provided.
    save : str or bool or None, optional
        If a string, saves the plot to the specified file path.
        If True, saves as "locus_plot.png".
    dpi : int, default 300
        Resolution of the saved figure in dots per inch.
    ax : matplotlib.axes.Axes or None, optional
        Optional Matplotlib Axes object to plot into.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting locus plot figure.
    """
    locus_df = locus_df.copy()
    locus_df["-log10(P)"] = -np.log10(locus_df[pval_col])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(locus_df[position_col], locus_df["-log10(P)"], color="lightgrey", s=point_size, zorder=2)

    texts = []
    if significance_threshold:
        sig_df = locus_df[locus_df[pval_col] < significance_threshold]
        ax.scatter(
            sig_df[position_col],
            sig_df["-log10(P)"],
            color=highlight_color,
            s=point_size * 1.5,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        if label_column and label_column in locus_df.columns:
            for _, row in sig_df.iterrows():
                texts.append(
                    ax.text(
                        row[position_col],
                        row["-log10(P)"],
                        str(row[label_column]),
                        fontsize=labelsize,
                        ha="center",
                        va="bottom",
                        zorder=4,
                    )
                )

        ax.axhline(-np.log10(significance_threshold), color="grey", linestyle="--", lw=1)

    gene_track_y = -1
    gene_height = 0.3
    if show_genes and gene_df is not None and not gene_df.empty:
        for _, row in gene_df.iterrows():
            start, end = row["start"], row["end"]
            gene_name = row["gene"]
            strand = row.get("strand", "+")

            if gene_style == "box":
                rect = Rectangle(
                    (start, gene_track_y - gene_height / 2),
                    end - start,
                    gene_height,
                    facecolor=gene_color,
                    edgecolor="black",
                    lw=0.5,
                    zorder=1,
                )
                ax.add_patch(rect)

            elif gene_style == "bracket":
                ax.plot([start, end], [gene_track_y] * 2, color=gene_color, lw=1.8, zorder=1)
                cap_height = gene_height
                ax.plot([start, start], [gene_track_y, gene_track_y + cap_height], color=gene_color, lw=1.5)
                ax.plot([end, end], [gene_track_y, gene_track_y + cap_height], color=gene_color, lw=1.5)

            if strand in {"+", "-"} and end - start > 10_000:
                arrow_dir = 1 if strand == "+" else -1
                arrow_x = start + (end - start) * (0.75 if strand == "+" else 0.25)
                ax.arrow(
                    arrow_x,
                    gene_track_y,
                    (end - start) * 0.05 * arrow_dir,
                    0,
                    head_width=gene_height * 0.6,
                    head_length=(end - start) * 0.02,
                    fc=gene_color,
                    ec=gene_color,
                    linewidth=0,
                    zorder=1,
                )

            ax.text(
                (start + end) / 2,
                gene_track_y - 0.25,
                gene_name,
                fontsize=labelsize * 0.9,
                ha="center",
                va="top",
                zorder=2,
            )

    y_max = locus_df["-log10(P)"].max() + 1
    ax.set_ylim(gene_track_y - 0.6, max(1.0, y_max))

    y_ticks = ax.get_yticks()
    filtered_ticks = [yt for yt in y_ticks if 0 <= yt <= max(1.0, y_max)]

    if not filtered_ticks:
        filtered_ticks = [0, max(1.0, y_max)]

    ax.set_yticks(filtered_ticks)
    ax.spines["left"].set_bounds(0, max(1.0, y_max))
    ax.set_xlim(locus_df[position_col].min(), locus_df[position_col].max())

    ax.set_xlabel("Genomic Position", fontsize=labelsize)
    ax.set_ylabel("-log10(p-value)", fontsize=labelsize)
    if title:
        ax.set_title(title, fontsize=titlesize)

    ax.axhline(y=0, color="black", lw=0.8)
    ax.spines["bottom"].set_position(("data", 0))

    plt.tight_layout()
    if texts:
        adjust_text(
            texts, ax=ax, only_move={"points": "y", "text": "y"}, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5)
        )

    if save:
        path = save if isinstance(save, str) else "locus_plot.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig
