import logging
from typing import Literal

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.figure import Figure
from scipy.sparse import issparse

from cellink._core import DonorData
from cellink._core.data_fields import CAnn, VAnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def expression_by_genotype(
    dd: DonorData,
    snp: str | int = None,
    celltype_key: str = CAnn.celltype,
    gene_filter: list = None,
    celltype_filter: list = None,
    mode: Literal["cell_type", "gene"] = "cell_type",
    n_cols: int = None,
    a0: str = VAnn.a0,
    a1: str = VAnn.a1,
    figsize: tuple = None,
    labelsize: int = None,
    titlesize: int = None,
    plot_type: Literal["violin", "box"] = "violin",
    show_stripplot: bool = False,
    show_axis_ticks: bool = True,
    title: str = None,
    palette: str = "Set2",
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 300,
    ax=None,
) -> Figure:
    """
    Plot gene expression by donor genotype across cell types or genes.

    Parameters
    ----------
    dd : DonorData
        Donor-level expression and genotype data.
    snp : str or int
        SNP ID or index to stratify by genotype.
    celltype_key : str
        Column in `.obs` denoting cell type.
    gene_filter : list
        Genes to include in the plot.
    celltype_filter : list
        Optional list of cell types to include.
    mode : {'cell_type', 'gene'}
        Group plots by cell type or gene.
    n_cols : int
        Number of columns in FacetGrid.
    a0 : str
        Column name for allele 0.
    a1 : str
        Column name for allele 1.
    figsize : tuple
        Size of the figure.
    labelsize : int
        Size of axis labels.
    titlesize : int
        Title font size.
    plot_type : {'violin', 'box'}
        Plot type to use.
    show_stripplot : bool
        Whether to overlay a stripplot.
    show_axis_ticks : bool
        Whether to show axis ticks for subplots.
    title : str
        Title for the plot.
    palette : str
        Seaborn color palette.
    show : bool or None
        Whether to display the plot. Defaults to auto-show behavior.
    save : str or bool or None
        File path or boolean to save the plot.
    ax : matplotlib.axes.Axes or None
        Optional matplotlib Axes to plot into.
    dpi : int
        Resolution for saved image.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    figsize = figsize or plt.rcParams.get("figure.figsize", (6.4, 4.8))
    labelsize = labelsize or plt.rcParams.get("axes.labelsize", 12)
    titlesize = titlesize or plt.rcParams.get("axes.titlesize", 14)

    dd_C = dd.C
    dd_G = dd.G

    if celltype_filter is not None:
        dd_C = dd_C[np.where(dd_C.obs[celltype_key].isin(celltype_filter))[0], :]
    if gene_filter is not None:
        dd_C = dd_C[:, dd_C.var.index.get_indexer(gene_filter)]

    if snp in dd_G.var.index:
        dd_G = dd_G[:, dd_G.var.index.get_indexer([snp])]
    elif isinstance(snp, int):
        dd_G = dd_G[:, snp]
    else:
        raise ValueError(f"{snp} can't be found in DonorData.G.")

    geno_label_map = {
        0: dd_G.var[a0].iloc[0] * 2,
        1: dd_G.var[a0].iloc[0] + dd_G.var[a1].iloc[0],
        2: dd_G.var[a1].iloc[0] * 2,
    }
    X = dd_G.X
    if isinstance(X, da.Array):
        X = X.compute()
    donor_to_genotype = pd.Series(np.squeeze(X), index=dd_G.obs.index)
    donor_to_genotype = donor_to_genotype.map(geno_label_map)

    merged = []
    for gene in gene_filter:
        gene_expression = dd_C.X[:, dd_C.var.index.get_indexer([gene])]
        if issparse(gene_expression):
            gene_expression = gene_expression.toarray()
        gene_expression = np.squeeze(gene_expression)

        df = pd.DataFrame(
            {
                "expression": gene_expression,
                "donor_id": dd_C.obs["donor_id"],
                "cell_type": dd_C.obs[celltype_key],
                "gene": gene,
            }
        )

        df = df.groupby(["donor_id", "cell_type", "gene"]).mean().reset_index()
        df["genotype"] = df["donor_id"].map(donor_to_genotype)
        merged.append(df)

    merged = pd.concat(merged)

    genotype_order = [dd_G.var[a0].iloc[0] * 2, dd_G.var[a0].iloc[0] + dd_G.var[a1].iloc[0], dd_G.var[a1].iloc[0] * 2]
    merged["genotype"] = pd.Categorical(merged["genotype"], categories=genotype_order, ordered=True)

    facet_col = mode
    facet_values = merged[facet_col].unique()
    n_facets = len(facet_values)

    if n_facets == 0:
        raise ValueError(f"No unique values found for {facet_col}")

    plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
    plot_kwargs = {"x": "genotype", "y": "expression", "palette": palette, "order": genotype_order}
    if plot_type == "violin":
        plot_kwargs["inner"] = "box"

    if n_facets == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        plot_func(data=merged, ax=ax, **plot_kwargs)
        if show_stripplot:
            sns.stripplot(
                data=merged,
                x="genotype",
                y="expression",
                color="black",
                size=3,
                jitter=True,
                alpha=0.5,
                order=genotype_order,
                ax=ax,
            )
        if title:
            ax.set_title(f"{title}", fontsize=titlesize)
        ax.set_xlabel("genotype", fontsize=labelsize)
        ax.set_ylabel("expression", fontsize=labelsize)
    else:
        if n_cols is None:
            n_cols = min(n_facets, 3)

        g = sns.FacetGrid(merged, col=facet_col, col_wrap=n_cols, sharey=True, height=figsize[1])
        g.map_dataframe(plot_func, **plot_kwargs)
        if show_stripplot:
            g.map_dataframe(
                sns.stripplot,
                x="genotype",
                y="expression",
                color="black",
                size=2.5,
                jitter=True,
                alpha=0.5,
                order=genotype_order,
            )

        g.set_axis_labels("genotype", "expression", fontsize=labelsize)
        g.set_titles(col_template="{col_name}", size=titlesize)
        if title:
            g.fig.suptitle(f"{title}", fontsize=titlesize + 2)
        if show_axis_ticks:
            for ax in g.axes.flat:
                ax.tick_params(axis="x", which="both", labelbottom=True)
                ax.tick_params(axis="y", which="both", labelleft=True)
        fig = g.fig
        plt.subplots_adjust(top=0.88)

    plt.tight_layout()

    if save:
        save_path = save if isinstance(save, str) else "expression_by_genotype.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig


def volcano(
    df: pd.DataFrame,
    logfc_col: str = "logFC",
    pval_col: str = "pval",
    label_col: str = None,
    pval_thresh: float = 0.05,
    logfc_thresh: float = 1.0,
    highlight_color: str = "red",
    non_sig_color: str = "grey",
    figsize: tuple = None,
    labelsize: int = None,
    titlesize: int = None,
    title: str = None,
    show: bool | None = None,
    save: str = None,
    dpi: int = 300,
    ax=None,
) -> Figure:
    """
    Generate a volcano plot showing -log10(p-value) vs. log fold change.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing log fold changes and p-values.
    logfc_col : str, default 'logFC'
        Column name for log fold changes.
    pval_col : str, default 'pval'
        Column name for p-values.
    label_col : str
        Column name for gene/feature labels. Used for annotation.
    pval_thresh : float, default 0.05
        Significance threshold for p-values.
    logfc_thresh : float, default 1.0
        Threshold for absolute log fold change.
    highlight_color : str, default 'red'
        Color for significant points.
    non_sig_color : str, default 'grey'
        Color for non-significant points.
    figsize : tuple
        Figure size.
    labelsize : int
        Axis label font size.
    titlesize : int
        Title font size.
    title : str
        Plot title.
    show : bool or None
        Whether to display the plot.
    save : str or None
        Path to save the figure. If provided, the figure is saved instead of shown.
    dpi : int, default 300
        Figure resolution.
    ax : matplotlib.axes.Axes
        Axis to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting volcano plot figure.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """
    figsize = figsize or plt.rcParams.get("figure.figsize", (6, 6))
    labelsize = labelsize or plt.rcParams.get("axes.labelsize", 12)
    titlesize = titlesize or plt.rcParams.get("axes.titlesize", 14)

    if logfc_col not in df.columns or pval_col not in df.columns:
        raise ValueError(f"'{logfc_col}' or '{pval_col}' not found in DataFrame.")

    filtered = df.dropna(subset=[logfc_col, pval_col])
    if filtered.empty:
        raise ValueError("No valid values after filtering.")

    filtered["-log10p"] = -np.log10(filtered[pval_col])
    filtered["significant"] = (filtered[pval_col] < pval_thresh) & (np.abs(filtered[logfc_col]) >= logfc_thresh)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        filtered.loc[~filtered["significant"], logfc_col],
        filtered.loc[~filtered["significant"], "-log10p"],
        c=non_sig_color,
        alpha=0.5,
        label="Not Significant",
        s=20,
    )

    ax.scatter(
        filtered.loc[filtered["significant"], logfc_col],
        filtered.loc[filtered["significant"], "-log10p"],
        c=highlight_color,
        alpha=0.7,
        label="Significant",
        s=20,
    )

    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", lw=1)
    ax.axvline(-logfc_thresh, color="black", linestyle="--", lw=1)
    ax.axvline(logfc_thresh, color="black", linestyle="--", lw=1)

    ax.set_xlabel("log2 Fold Change", fontsize=labelsize)
    ax.set_ylabel("-log10(p-value)", fontsize=labelsize)
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    ax.legend()
    ax.grid(True)

    if label_col and label_col in filtered.columns:
        texts = []
        for _, row in filtered[filtered["significant"]].iterrows():
            txt = ax.text(row[logfc_col], row["-log10p"], str(row[label_col]), fontsize=8, ha="center", va="center")
            texts.append(txt)
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand_text=(1.05, 1.2),
            expand_points=(1.05, 1.2),
            force_text=0.5,
            force_points=0.2,
        )

    if save:
        fig.savefig(save, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig
