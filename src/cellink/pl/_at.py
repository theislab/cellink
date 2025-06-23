import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import logging
from scipy.stats import beta 
from cellink._core import DonorData
from scipy.sparse import issparse
from typing import Literal
from cellink._core.data_fields import VAnn, CAnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def expression_by_genotype(
    dd: DonorData,
    snp: str | int = None,
    celltype_key: str = CAnn.celltype,
    gene_filter: list = None,
    celltype_filter: list = None,
    mode: Literal['cell_type', 'gene'] = 'cell_type',
    n_cols: int = None,
    a0: str = VAnn.a0, 
    a1: str = VAnn.a1, 
    figsize: tuple = None,
    labelsize: int = None,
    titlesize: int = None,
    plot_type: Literal['violin', 'box'] = "violin",
    show_stripplot: bool = False,
    show_axis_ticks: bool = True,
    title: str = None,
    palette: str = 'Set2',
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 300,
    ax=None,
):
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

    figsize = figsize or plt.rcParams.get('figure.figsize', (6.4, 4.8))
    labelsize = labelsize or plt.rcParams.get('axes.labelsize', 12)
    titlesize = titlesize or plt.rcParams.get('axes.titlesize', 14)

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
        0: dd_G.var[a0].iloc[0]*2,
        1: dd_G.var[a0].iloc[0]+dd_G.var[a1].iloc[0],
        2: dd_G.var[a1].iloc[0]*2
    }
    donor_to_genotype = pd.Series(np.squeeze(dd_G.X.compute()), index=dd_G.obs.index)
    donor_to_genotype = donor_to_genotype.map(geno_label_map)

    merged = []
    for gene in gene_filter:
        gene_expression = dd_C.X[:, dd_C.var.index.get_indexer([gene])]
        if issparse(gene_expression):
            gene_expression = gene_expression.toarray()
        gene_expression = np.squeeze(gene_expression)

        df = pd.DataFrame({
            "expression": gene_expression,
            "donor_id": dd_C.obs["donor_id"],
            "cell_type": dd_C.obs[celltype_key],
            "gene": gene
        })

        df = df.groupby(["donor_id", "cell_type", "gene"]).mean().reset_index()
        df["genotype"] = df["donor_id"].map(donor_to_genotype)
        merged.append(df)

    merged = pd.concat(merged)

    genotype_order = [dd_G.var[a0].iloc[0]*2, dd_G.var[a0].iloc[0]+dd_G.var[a1].iloc[0], dd_G.var[a1].iloc[0]*2]
    merged['genotype'] = pd.Categorical(merged['genotype'], categories=genotype_order, ordered=True)

    facet_col = mode
    facet_values = merged[facet_col].unique()
    n_facets = len(facet_values)

    if n_facets == 0:
        raise ValueError(f"No unique values found for {facet_col}")
    
    plot_func = sns.violinplot if plot_type == 'violin' else sns.boxplot
    plot_kwargs = {
        'x': 'genotype',
        'y': 'expression',
        'palette': palette,
        'order': genotype_order
    }
    if plot_type == 'violin':
        plot_kwargs['inner'] = 'box'

    if n_facets == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        plot_func(data=merged, ax=ax, **plot_kwargs)
        if show_stripplot:
            sns.stripplot(
                data=merged, x='genotype', y='expression',
                color='black', size=3, jitter=True, alpha=0.5,
                order=genotype_order, ax=ax
            )
        if title:
            ax.set_title(f'{title}', fontsize=titlesize)
        ax.set_xlabel('genotype', fontsize=labelsize)
        ax.set_ylabel('expression', fontsize=labelsize)
    else:
        if n_cols is None:
            n_cols = min(n_facets, 3)

        g = sns.FacetGrid(merged, col=facet_col, col_wrap=n_cols, sharey=True, height=figsize[1])
        g.map_dataframe(plot_func, **plot_kwargs)
        if show_stripplot:
            g.map_dataframe(
                sns.stripplot, x='genotype', y='expression',
                color='black', size=2.5, jitter=True, alpha=0.5,
                order=genotype_order
            )

        g.set_axis_labels("genotype", "expression", fontsize=labelsize)
        g.set_titles(col_template="{col_name}", size=titlesize)
        if title:
            g.fig.suptitle(f"{title}", fontsize=titlesize + 2)
        if show_axis_ticks:
            for ax in g.axes.flat:
                ax.tick_params(axis='x', which='both', labelbottom=True)
                ax.tick_params(axis='y', which='both', labelleft=True)
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

def qq(
    pvals_df: pd.DataFrame,
    pval_col: str = 'pv_adj',
    group_col: str = 'group',
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
):
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

    figsize = figsize or plt.rcParams.get('figure.figsize', (4.8, 6.4))
    labelsize = labelsize or plt.rcParams.get('axes.labelsize', 12)
    titlesize = titlesize or plt.rcParams.get('axes.titlesize', 14)

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

        ax.scatter(expected, observed, edgecolor='k', facecolor='none', s=20, alpha=0.2)
        ax.axline([0, 0], slope=1, color=null_line_color, linestyle='--')

        if show_ci:
            lower = -np.log10(beta.ppf(0.975, i, n + 1 - i))
            upper = -np.log10(beta.ppf(0.025, i, n + 1 - i))
            ax.fill_between(expected, lower, upper, color='gray', alpha=0.3)

        ax.set_xlabel('expected -log10(p)', fontsize=labelsize)
        ax.set_ylabel('observed -log10(p)', fontsize=labelsize)
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
            fig, ax = fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows/2))
        else:
            fig, axes = ax.figure
        axes = axes.flatten() if n_groups > 1 else [axes]

        for i, group in enumerate(groups):
            ax = axes[i]
            group_pvals = filtered.loc[filtered[group_col] == group, pval_col].values
            qq_single(ax, group_pvals, title=str(group), show_ci=show_ci)

        for j in range(n_groups, len(axes)):
            axes[j].axis('off')

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
    palette_mode: str = 'alternating', 
    palette_alternating_colors: list | np.ndarray = ['#E24E42', '#008F95'],
    significant_hit_color: str = "#D62728",
    palette: str = None,
    show: bool | None = None,
    save: str | bool | None = None,
    dpi: int = 300,
    ax=None,
):
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
        File path to save the plot. If True, saves as "expression_by_genotype.png".
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
    
    figsize = figsize or plt.rcParams.get('figure.figsize', (14, 6))
    labelsize = labelsize or plt.rcParams.get('axes.labelsize', 8)
    titlesize = titlesize or plt.rcParams.get('axes.titlesize', 14)

    pvals_df[chromosome_col] = pvals_df[chromosome_col].replace({'X': 23, 'Y': 24}).astype(int)
    pvals_df = pvals_df.sort_values([chromosome_col, position_col])

    pvals_df['-log10(P)'] = -np.log10(pvals_df[pval_col])
    pvals_df['ind'] = range(len(pvals_df))
    pvals_df[chromosome_col] = pd.Categorical(pvals_df[chromosome_col], categories=sorted(pvals_df[chromosome_col].unique()), ordered=True)

    chr_max_bp = pvals_df.groupby(chromosome_col)[position_col].max().cumsum()
    chr_offsets = chr_max_bp.shift(fill_value=0)
    pvals_df['cum_pos'] = pvals_df.apply(lambda row: row[position_col] + chr_offsets[row[chromosome_col]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if palette is not None or (palette is None and palette_mode != "alternating"): 
        if palette is None:
            palette = "tab20"
        palette = sns.color_palette(palette, n_colors=22)
        colors = {chr_num: palette[i % 22] for i, chr_num in enumerate(sorted(pvals_df[chromosome_col].unique()))}
    else: 
        colors = {chr_num: palette_alternating_colors[i % 2] for i, chr_num in enumerate(sorted(pvals_df[chromosome_col].unique()))}

    x_labels = []
    x_labels_pos = []

    for i, (chr_num, group) in enumerate(pvals_df.groupby(chromosome_col)):
        color = colors[chr_num]
        ax.scatter(
            group['cum_pos'], group['-log10(P)'],
            color=color, s=point_size, label=f'Chr {chr_num}', zorder=2
        )

        ax.axvline(x=group['cum_pos'].iloc[-1], color='lightgrey', linestyle='--', lw=0.5)

        x_labels.append(str(chr_num))
        x_labels_pos.append(int(group['cum_pos'].min() + (group['cum_pos'].max()-group['cum_pos'].min())/2))

    if significance_threshold:
        sig_hits = pvals_df[pvals_df[pval_col] < significance_threshold]
        ax.scatter(
            sig_hits['cum_pos'], sig_hits['-log10(P)'],
            color=significant_hit_color, s=int(point_size*1.5), edgecolor='k', label='Significant', zorder=3
        )

        if label_column and label_column in pvals_df.columns and not sig_hits.empty:
            logger.info(f"Significant hits: {', '.join(sig_hits[label_column])}")
            texts = []
            for _, row in sig_hits.iterrows():
                x = row['cum_pos']
                y = row['-log10(P)']
                if np.isfinite(x) and np.isfinite(y):
                    texts.append(ax.text(x, y + 0.3, str(row[label_column]),
                                        fontsize=labelsize, ha='center', va='center', zorder=4)) 
            ax.axhline(y=-np.log10(significance_threshold), color='grey', linestyle='--', lw=1)
        else:
            texts = []
    else:
        texts = []

    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels, fontsize=labelsize)
    ax.set_xlabel('chromosome')
    ax.set_ylabel('-log10(p-value)')
    if title:
        ax.set_title(title)

    min_x, max_x = pvals_df['cum_pos'].min(), pvals_df['cum_pos'].max()
    pad = (max_x - min_x) * 0.005  # 0.5% padding
    ax.set_xlim(pvals_df['cum_pos'].min(), pvals_df['cum_pos'].max())
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if len(texts) > 0:
        adjust_text(texts, ax=ax, only_move={'points': 'y', 'text': 'y'}, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5)) 

    if save:
        save_path = save if isinstance(save, str) else "manhattan.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    elif show or (show is None and ax is None):
        plt.show()

    return fig
