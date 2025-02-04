from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from plotnine import *
from cellink.tl._eqtl import _get_pb_data
import pandas as pd
import numpy as np
from itertools import islice
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from upsetplot import UpSet, from_indicators

# FDR correction on association results #############


def FDR_correction(association_results):
    """
    association_results: pd.DataFrame, results from burden_test/ output written in run_burden_ass.py

    returns: - df_final: pd.DataFrame, association_results, sorted by celltype and burden type added column "FDR corrected" values and "significant"
             - df_to_plot: pd.DataFrame, summary of # significant egenes per cell type and burden type
    """

    df = association_results.copy()
    df_to_plot = pd.DataFrame(columns=['celltype', 'annotation', 'n'])

    df_final = []
    for celltype in df["cell_type"].unique():
        df_annotations = {}
        for annotation in df["burden_type"].unique():
            df_annotations[annotation] = df[(df["cell_type"] == celltype) & (df["burden_type"] == annotation)]
            df_annotations[annotation] = df_annotations[annotation].dropna(how='all')
            # FDR correction
            df_annotations[annotation]["significant"], df_annotations[annotation]["FDR_corrected"], _, _ = multipletests(df_annotations[annotation]["pvalue"], alpha=0.05, method='fdr_bh')

            df_to_plot = pd.concat([df_to_plot, pd.DataFrame({
                    'celltype': [celltype],
                    'annotation': [annotation],
                    'n': [len(df_annotations[annotation][df_annotations[annotation]["significant"]])]
                })], ignore_index=True)
            df_annotations[annotation]["celltype-annotation"]= f"{celltype}-{annotation}"
            df_final.append(df_annotations[annotation]) 

    df_final = pd.concat(df_final)

    df_to_plot['celltype_annotation'] = df_to_plot['celltype'] + "_" + df_to_plot['annotation']
    df_to_plot
    return df_final, df_to_plot


# grouped bar plot of egenes per chromosome, cell_type and burden #############


def plot_egenes(df_to_plot):
    """
    df_to_plot: pd.DataFrame, as output[1] of  FDR_correction()
    returns: plot, grouped barplot # egenes per cell type and burden
    """
    zero_types = list(df_to_plot.query("n==0")["celltype_annotation"])
    #Filter out the rows where `n` is 0 and create the plot
    plot = (
        ggplot(df_to_plot.query("celltype_annotation not in @zero_types"), 
               aes(x='celltype', y='n', fill='annotation'))  # Specify x, y, and fill variables
        + geom_bar(stat="identity", position="dodge")  # Use position="dodge" for grouped bars
        + theme_classic()  # Optional: use a clean theme
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, vjust=1),
            figure_size=(7, 4)
        )
        + labs(
            y="Number of E-Genes",
            x="",
            fill="Burden Annotation"
        )
    )

    return plot


def plot_egenes_seaborn(df_to_plot):
    """
    df_to_plot: pd.DataFrame, as output[1] of FDR_correction()
    returns: seaborn grouped barplot # egenes per cell type and burden
    """
    # Filter out rows where `n` is 0
    df_filtered = df_to_plot[df_to_plot["n"] > 0]

    # Set figure size
    plt.figure(figsize=(10, 5))

    # Create grouped barplot
    sns.barplot(
        data=df_filtered,
        x="celltype",
        y="n",
        hue="annotation",
        dodge=True  # Ensures bars are grouped correctly
    )

    # Adjust labels and title
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.ylabel("Number of E-Genes")
    plt.title("E-Genes per Cell Type and Burden")
    plt.legend(title="Burden Annotation")

    # Show the plot
    plt.show()


def plot_egenes_with_broken_axis(df_to_plot):
    #plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 16})
    # Filter out rows where `n` is 0
    df_filtered = df_to_plot[df_to_plot["n"] > 0]

    df_filtered["celltype"] = pd.Categorical(df_filtered["celltype"], categories=sorted(df_filtered["celltype"].unique()), ordered=True)
    df_filtered["annotation"] = pd.Categorical(df_filtered["annotation"], categories=sorted(df_filtered["annotation"].unique()), ordered=True)

    # Sort the DataFrame by the specified order
    df_filtered = df_filtered.sort_values(by=["celltype", "annotation"])


    # Create figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 9]})
    
    # Set y-axis limits for the two axes
    upper_limit = 105
    lower_limit = 180

    # Upper part of the plot
    sns.barplot(data=df_filtered, x="celltype", y="n", hue="annotation", dodge=True, ax=ax1)
    ax1.set_ylim(lower_limit, 190)  # Set upper y-axis range
    ax1.spines["bottom"].set_visible(False)  # Hide bottom spine
    ax1.tick_params(bottom=False)  # Remove bottom ticks

    # Lower part of the plot
    sns.barplot(data=df_filtered, x="celltype", y="n", hue="annotation", dodge=True, ax=ax2)
    ax2.set_ylim(0, upper_limit)  # Set lower y-axis range
    ax2.spines["top"].set_visible(False)  # Hide top spine

    # Add diagonal lines to indicate the break in the y-axis
    d = 0.005  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

    # ax2_ticks = ax2.get_yticks()
    ax1.set_yticks([180, 190])
    ax2.set_yticks(range(0, 105, 10))
    #ax2.plot([], [], ' ', label="cut y-axis, GENE_TSS_DIST_SAIGE in CD4 NC: 186 eGenes")

    handles, labels = ax2.get_legend_handles_labels()

    # Find the index of "GENE_TSS_DISTANCE_SAIGE" and update its label
    index_to_update = labels.index("GENE_TSS_DISTANCE_SAIGE")
    labels[index_to_update] = "GENE_TSS_DISTANCE_SAIGE, \ncut y axis, 186 eGenes in CD4 NC"  # Replace with the new label text
    # Move "GENE_TSS_DISTANCE_SAIGE" to the bottom
    handles.append(handles.pop(index_to_update))
    labels.append(labels.pop(index_to_update))

    legend = ax2.legend(
        title="Burden Annotation", 
        handles=handles, 
        labels=labels, 
        #handles=[extra_text] + ax2.legend.legendHandles,  # Add extra text at the top
        loc='upper right',  # Place legend within the box
        fontsize=15,  # Increase font size
        title_fontsize=15  # Increase title font size
    )
    legend.set_frame_on(True)  # Add a box around the legend
    #legend.get_texts()[-1].set_fontsize(12)   # Set a smaller font size for the extra text
    #legend.get_texts()[-1].set_color("purple")  # Change text color to gray (or any color)

    # Add labels and legend
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of significant E-Genes", fontsize=18)
    ax1.set_ylabel("")
    #ax1.legend(title="Burden Annotation", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.legend_.remove()
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax1.set_title("E-Genes per Cell Type and Burden Type")
    ax1.set_title("")

    # Adjust layout
    plt.tight_layout()
    plt.show()




# QQ plots ##############


def QQ_plot_sig_egenes(FDR_corrected_results, burdentype):
    """
    FDR_corrected_results: pd.DataFrame, output[0] of FDR_correction()
    burdentype: Str

    returns: QQ plot for given burden. Red line: uniform distributed pvalues.
                                        in red: significant egens
    """
    df = FDR_corrected_results.copy().query("burden_type == @burdentype")

    df = df.sort_values("pvalue")
    df["-log10pval_expected"] = -np.log10(np.arange(1, len(df) + 1) / len(df))

    if "-log10pval" not in df.columns:
        df["-log10pval"] = -np.log10(df["pvalue"])

    aes_kwargs = dict(x="-log10pval_expected", y="-log10pval")
    plot = (
        ggplot(df, aes(**aes_kwargs, color="significant"))
        + labs(title=f"{burdentype}")
        + scale_color_manual(values = ["black", "red"])
        + theme_classic()
        + geom_abline(intercept=0, slope=1, color="red")
        + geom_point()
        + theme(legend_position="top", figure_size = (5,5))
    )
    return plot


# Plotnine grids #######


def _check_plotnine_grid(plots_list, figsize):
    if not isinstance(plots_list, list):
        raise ValueError('Input plots_list must be a list')
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError('Input figsize should be a tuple of length 2')


def plotnine_grid(plots_list, row=2, col=3, figsize=(12, 12)):
    _check_plotnine_grid(plots_list, figsize)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for ax, plot in zip(axes, plots_list):
        # Convert plotnine plot to matplotlib figure
        plot_fig = plot.draw()
        canvas = FigureCanvas(plot_fig)
        canvas.draw()

        # Convert figure to numpy array
        img_array = np.array(canvas.renderer.buffer_rgba())

        # Display in matplotlib subplot
        ax.imshow(img_array, aspect='auto')
        ax.axis('off')

    fig.tight_layout()
    return fig


def plotnine_grid2(plots_list, row=2, col=3, figsize=(30, 30)):
    _check_plotnine_grid(plots_list, figsize)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for ax, plot in zip(axes, plots_list):
        # Convert plotnine plot to matplotlib figure
        plot_fig = plot.draw()
        canvas = FigureCanvas(plot_fig)
        canvas.draw()

        # Convert figure to numpy array
        img_array = np.array(canvas.renderer.buffer_rgba())

        # Display in matplotlib subplot without stretching
        ax.imshow(img_array)
        ax.set_xticks([])  # Remove ticks
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.tight_layout()

    #return fig


# Pseudobulk grid scatter plots (with density coloring)##########


def get_pb_data_for_ct_per_gene(data, all_burdens, eigenvec, celltype, target_chrom):
    """
    data: DonorData object, annotated (output from run_data_annotation.py)
    all_burdens: pd.DataFrame, computed burden scores (output from run_compute_burdens.py)
    eigenvec: pd.DataFrame
                (eigenvec = pd.read_csv("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/input_data/pcdir/wgs.dose.filtered.R2_0.8.filtered.pruned.eigenvec", sep=' ', header=None)
    celltype: Str, cell type for which pb are computed
    target_chrom: Str

    returns: pb_by_gene_df, pd.DataFrame: sample id, pb_geneexpressions
    """
    eigenvec.index = eigenvec[1]
    eigenvec = eigenvec.iloc[:, 2:]
    eigenvec = eigenvec[eigenvec.index.isin(all_burdens.index.unique())]

    pb_data = _get_pb_data(
        scdata=data.adata,
        gdata=data.gdata,
        cell_type=celltype,
        target_chromosome=target_chrom,
        donor_key_in_scdata="individual",
        sex_key_in_scdata="sex",
        age_key_in_scdata="age",
        pseudobulk_aggregation_type="mean",
        min_individuals_threshold=10,
        n_top_genes=5_000,
        n_sc_comps=15,
        n_genetic_pcs=20,
        n_cellstate_comps=50,
        eigenvector_df=eigenvec
    )

    pb_data_all_genes = pb_data.adata.layers["mean"]
    pb_by_gene_df = pd.DataFrame(pb_data_all_genes, columns=pb_data.adata.var_names, index=pb_data.adata.obs_names)

    return pb_by_gene_df


def burdenscore_with_pb_expression(pb_by_gene_df, geneid, all_burdens, burden_type, celltype):
    """
    for a given gene, burdentype and celltype: combined info in one dataframe

    pb_by_gene_df: pd.DataFrame (output, get_pb_data_for_ct_per_gene)
    geneid: str
    all_burdens: pd.DataFrame, computed burden scores (output from run_compute_burdens.py)
    burden_type: Str
    celltype: Str (musst be the same as used in get_pb_data_for_ct_per_gene)

    returns: pd.DataFrame, cols: burdenscores, pb_expr, celltype
    """
    burden_type_gene = pd.DataFrame(all_burdens[all_burdens["Geneid"] == geneid][burden_type])
    # add pb expr
    burden_type_gene["pb_expr"] = pb_by_gene_df[geneid]
    burden_type_gene["cell_type"] = celltype
    return burden_type_gene


def plot_burden_expre_corr_originally(burden_type_gene, burden_type, geneid):
    """
    plots the relation between pb expression and the burden score
    Version from student notebook but slightly adapted.

    burden_type_gene:  pd.DataFrame (output, burdenscore_with_pb_expression, 
                            can be concatinated for multiple cell types)
    burden_type: Str
    """
    plot = (
        ggplot(burden_type_gene, aes(x=burden_type, y="pb_expr"))
        + labs(x = "Gene score", title=f"Gene: {geneid}, annotation: {burden_type}")
        + theme_classic()
        + geom_smooth()
        + geom_point()
        + facet_wrap("cell_type", scales = "free")
        + theme(legend_position="top", figure_size=(5,5), plot_title=element_text(size=10))
    )

    plot
    return plot


def plot_burden_expre_corr(burden_type_gene, burden_type, geneid, chrom,save=""):
    """
    plots the relation between pb expression and the burden score

    burden_type_gene:  pd.DataFrame (output, burdenscore_with_pb_expression, 
                            can be concatinated for multiple cell types)
    burden_type: Str
    """
    fig_height = 7
    fig_length = 4
    title = 12
    axis_title = 10
    axis_text = 8
    facet_text = 8
    n_celltypes = burden_type_gene["cell_type"].unique()
    if len(n_celltypes) > 11:
        fig_height = 20
        fig_length = 18
        title = 23
        axis_title = 18
        axis_text = 15
        facet_text = 15
    elif len(n_celltypes) > 6:
        fig_height = 12
        fig_length = 8
        title = 15
        axis_title = 14
        axis_text = 12
        facet_text = 12

    if burden_type == "GENE_TSS_DISTANCE":
        axis_text = axis_text/3

    def compute_2d_density(df):
        """
        Compute KDE density for Gene Score (burden_type) and PB Expression (pb_expr) within each cell_type group.
        """
        df = df.dropna(subset=["pb_expr"])
        if len(df) < 2:  # Skip if too few values
            return pd.Series(np.nan, index=df.index)

        if df["pb_expr"].nunique() == 1 or df[burden_type].nunique() == 1:
            # If all values are the same, assign uniform density
            return pd.Series(1.0, index=df.index)

        # Fit 2D KDE using both pb_expr and burden_type
        kde = gaussian_kde(np.vstack([df[burden_type], df["pb_expr"]]))

        # Evaluate KDE at each observed point
        density_values = kde(np.vstack([df[burden_type], df["pb_expr"]]))

        return pd.Series(density_values, index=df.index)  # Ensure row-wise mapping

    # Apply KDE computation per group
    burden_type_gene["density"] = burden_type_gene.groupby("cell_type", group_keys=False).apply(compute_2d_density)

    # Normalize density values for better visualization
    burden_type_gene["density"] = burden_type_gene["density"] / burden_type_gene["density"].max()

    #print(burden_type_gene.head(20))

    plot = (
    ggplot(burden_type_gene, aes(x = burden_type, y = "pb_expr", color="density"))
    + labs(x = "Gene score", title=f"Gene: {geneid}, Annotation: {burden_type}")
    + theme_classic()
    + geom_point()
    + geom_smooth()
    + facet_wrap("cell_type")
    + theme(legend_position="top",
            figure_size=(fig_height,fig_length),
            plot_title=element_text(size=title),  # Increase title font size
            axis_title_x=element_text(size=axis_title),  # Increase x-axis title font size
            axis_title_y=element_text(size=axis_title),  # Increase y-axis title font size
            axis_text_x=element_text(size=axis_text),   # Increase x-axis tick labels font size
            axis_text_y=element_text(size=axis_text),    # Increase y-axis tick labels font size)
            strip_text=element_text(size=facet_text)
            )
    )
    plot.show()
    if save != "":
        plot_name = f"{save}/dotplot_grid_{burden_type}_{geneid}_chr{chrom}.png"
        plot.save(plot_name, dpi=300, width=fig_length, height=fig_height)


def plot_burden_expre_corr_old_mai(burden_type_gene, burden_type, geneid, chrom, save=""):
    """
    plots the relation between pb expression and the burden score (without density coloring)

    burden_type_gene:  pd.DataFrame (output, burdenscore_with_pb_expression, 
                            can be concatinated for multiple cell types)
    burden_type: Str 
    """
    fig_height = 7
    fig_length = 4
    title = 12
    axis_title = 10
    axis_text = 8
    facet_text = 8
    n_celltypes = burden_type_gene["cell_type"].unique()
    if len(n_celltypes) > 11:
        fig_height = 20
        fig_length = 18
        title = 23
        axis_title = 18
        axis_text = 15
        facet_text = 15
    elif len(n_celltypes) > 6:
        fig_height = 12
        fig_length = 8
        title = 15
        axis_title = 14
        axis_text = 12
        facet_text = 12

    if burden_type == "GENE_TSS_DISTANCE":
        axis_text = axis_text/3
    print(burden_type_gene)
    plot = (
    ggplot(burden_type_gene, aes(x = burden_type, y = "pb_expr"))
    + labs(x = "Gene score", title=f"Gene: {geneid}, Annotation: {burden_type}")
    + theme_classic()
    + geom_smooth()
    + geom_point()
    + facet_wrap("cell_type")
    + theme(legend_position="top",
            figure_size=(fig_height,fig_length),
            plot_title=element_text(size=title),  # Increase title font size
            axis_title_x=element_text(size=axis_title),  # Increase x-axis title font size
            axis_title_y=element_text(size=axis_title),  # Increase y-axis title font size
            axis_text_x=element_text(size=axis_text),   # Increase x-axis tick labels font size
            axis_text_y=element_text(size=axis_text),    # Increase y-axis tick labels font size)
            strip_text=element_text(size=facet_text)
        )
    )
    plot.show()
    if save != "":
        plot_name = f"{save}/dotplot_grid_{burden_type}_{geneid}_chr{chrom}.png"
        plot.save(plot_name, dpi=300, width=fig_length, height=fig_height)


def plot_burden_expre_corr_mai(burden_type_gene, burden_type, geneid, chrom, save=""):
    """
    plots the relation between pb expression and the burden score
    Note: i adated the code so it would work perfectly for our gene of interest, might need adaptations again for code to work for all genes

    burden_type_gene:  pd.DataFrame (output, burdenscore_with_pb_expression, 
                            can be concatinated for multiple cell types)
    burden_type: Str
    geneid: str target gene id
    chrom: str of target chromosome
    ax: where to plot this
    """
    # Create the FacetGrid
    g = sns.FacetGrid(burden_type_gene, col="cell_type", col_wrap=3, sharex=True, sharey=True,height=4, aspect=1.5)

    # Function to plot scatter plot with density-based colors
    def scatter_with_density(data, x, y, **kwargs):
        # Remove NaNs/Infs
        data = data.dropna(subset=[x, y])

        # Compute KDE values for color mapping
        if len(data) > 1:  # Ensure enough data points for KDE
            values = np.vstack([data[x], data[y]])
            kernel = gaussian_kde(values)(values)
        else:
            kernel = np.zeros(len(data))  # Avoid errors when data is too small

        # Plot scatter with KDE-based color
        scatter = plt.scatter(data[x], data[y], c=kernel,cmap="viridis")
        plt.colorbar(scatter, ax=plt.gca(), label="Density")

    g.map_dataframe(scatter_with_density, data=burden_type_gene, x=burden_type, y="pb_expr")

    def plot_regression(data, x, y, **kwargs):
        sns.regplot(data=data, x=x, y=y, scatter=False, color="blue", line_kws={"color": "black", "lw": 1})

    # Apply regression line to each facet
    g.map_dataframe(plot_regression, data=burden_type_gene, x=burden_type, y="pb_expr")

    # Adjust layout
    g.set_axis_labels("Gene Impairment Score", "Pseudobulk expression", fontweight="bold", fontsize=12)
    g.set_titles(col_template="{col_name}", fontweight="bold", size=14)

    # Add a title for the entire plot
    plt.suptitle(f"{burden_type}", fontsize=16, y=0.98, x=0.33)
    g.fig.subplots_adjust(top=0.85)  # Adjust the top spacing for the title

    # Adjust layout to make room for the main title
    #plt.subplots_adjust(top=1)  # Adjust top space for the title to avoid overlap

    plt.show()
    
    if save != "":
        plot_name = f"{save}/densityscatterplot_grid_{burden_type}_{gene}_chr{chrom}.png"
        plt.save(plot_name, dpi=300)


def get_pb_plots(data,
                 df_burdens_chrom,
                 eigenvec,
                 chrom,
                 cell_types=['CD8 ET', 'CD8 NC','CD8 ET', 'CD8 NC', 'CD4 NC', 'Mono C', 'NK', 'B IN', 'CD4 ET', 'Mono NC', 'CD8 S100B', 'B Mem', 'Plasma', 'DC', 'NK R','CD4 SOX4'],
                 burden_types=['CADD_PHRED','CADD_PHRED', 'DNA_LM_up', 'DNA_LM_down', 'MAF_beta_1.25', 'DNA_LM_combined', 'GENE_TSS_DISTANCE_SAIGE'],
                 gene_limit=3,
                 gene_list=[],
                 save_dir=""):
    """
    Plots the relation between pb expression and the burden score per chromosome & cell_type for set number of genes

    data: GeneAnnoData Object to get pb from
    df_burdens_chrom (pd.DataFrame): burden results
    eigenvec (pd.DataFrame): read eigenvector object without processing
    chrom (str): chromosome number
    cell_types (list of string): list of cell types
    burden_types (list of string): list of burden annotations
    gene_limit (int): max genes that should be taken for plotting
    gene_list (list of str): alternatively to gene_limit set a list of genes
    save_dir (str): dir path where to save the plots (plot name will be generated)
    """
    all_pb_X = []
    burdens_expr_per_gene = {}
    for cell_type in cell_types:
        print(f"Getting PB {cell_type}")
        pb_X = get_pb_data_for_ct_per_gene(data, df_burdens_chrom, eigenvec, cell_type, chrom)
        all_pb_X.append(pb_X)
        # define genes to create plots for
        for_genes = pb_X.columns[:gene_limit]  # first gene_limit genes from pb_X
        if len(gene_list) != 0:  # take input genes
            for_genes = gene_list
        for gene in for_genes:
            if gene in pb_X.keys():
                for burden in burden_types:
                    # get burden scores and save per burden annotation and gene the pb expression results
                    burdens_expr = burdenscore_with_pb_expression(pb_X, gene, df_burdens_chrom, burden, cell_type)
                    if burden not in burdens_expr_per_gene.keys():
                        burdens_expr_per_gene[burden] = {gene: burdens_expr}
                    else:
                        if gene not in burdens_expr_per_gene[burden].keys():
                            burdens_expr_per_gene[burden][gene] = burdens_expr
                        else:
                            burdens_expr_per_gene[burden][gene] = pd.concat([burdens_expr_per_gene[burden][gene],burdens_expr], ignore_index=True)
            else:
                print(f"pb X not calculated for genen {gene} in cell type {cell_type}")

    # create plots
    for annotation, genes in burdens_expr_per_gene.items():
        for gene, expression_data in islice(genes.items(), 3):
            plot_burden_expre_corr_mai(expression_data, annotation, gene, chrom, save_dir)


# upset plots ##########


def plot_upset(burden_dict, cell_type, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})

    # Create a DataFrame indicating gene presence in each burden_type
    all_genes = set(gene for genes in burden_dict.values() for gene in genes)
    data = {gene: [gene in burden_dict[bt] for bt in burden_dict] for gene in all_genes}
    df = pd.DataFrame.from_dict(data, orient='index', columns=burden_dict.keys())

    # Ensure all values are boolean
    df = df.astype(bool)

    # Convert to UpSet format
    upset_data = from_indicators(df.columns, df)
    #import ipdb; ipdb.set_trace()

    # Create and show UpSet plot
    UpSet(upset_data).plot()
    plt.title(cell_type)
    plt.show()


# Beta distribution plots #####################


def grouped_boxplot(df):

    sorted_df = df.sort_values(by=['cell_type', 'burden_type'])

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=sorted_df, x='cell_type', y='beta', hue='burden_type')

    #plt.title('Distribution of Beta Values for Cell Type and Burden Type')
    plt.ylabel('Beta Values', fontsize = 16)

    plt.xticks(rotation=45, fontsize=14)


    plt.yticks(fontsize=14)


    plt.legend(title='Burden Annotation',title_fontsize=16, fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/output/data_plots/beta_distribution.png")
    plt.show()


def calculate_std(df):

    std_df = df.groupby(['cell_type', 'burden_gene'])['beta'].std().reset_index(name='std_beta')

    df = df.merge(std_df, on=['cell_type', 'burden_gene'])

    return df


def grouped_boxplot_sd(df):
    # Sortieren des DataFrame nach 'cell_type'
    sorted_df = df.sort_values(by=['cell_type', 'burden_type'])

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=sorted_df, x='cell_type', y='std_beta', hue='burden_type')

    #plt.title('Distribution of Standard Deviation of Beta Values for Cell Type and Burden Type')
    plt.ylabel('Standard Deviation of Beta Value', fontsize = 16)

    plt.xticks(rotation=45, fontsize= 14)
    plt.yticks(fontsize= 14)
    plt.legend(title='Burden Annotation', title_fontsize=16, fontsize = 14,bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/output/data_plots/std_beta_distribution.png")
    plt.show()
