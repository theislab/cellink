import pandas as pd
import gc
from pathlib import Path
import warnings

import anndata as ad
import scanpy as sc
import dask.array as da
import numpy as np
from tqdm.auto import tqdm

import cellink as cl
from cellink._core import DAnn, GAnn
from cellink.tl._rvat import run_burden_test, run_skat_test, beta_weighting
from cellink.utils import column_normalize, gaussianize

from cellink.at.acat import acat_test
from cellink.resources import get_onek1k

DATA = Path(cl.__file__).parent.parent.parent / "docs/tutorials/data"

n_gpcs = 20
n_epcs = 15
batch_e_pcs_n_top_genes = 2000
chrom = 22
cis_window = 500_000
#cell_type = "CD8 Naive"
celltype_key = "predicted.celltype.l2"
#pb_gex_key = f"PB_{cell_type}"  # pseudobulk expression in dd.G.obsm[key_added]
original_donor_col = "donor_id"
min_percent_donors_expressed = 0.1
do_debug = False

dd = get_onek1k(config_path='./cellink/resources/config/onek1k.yaml')
dd


###
aggregation_map = {
    # CD4 T cell group
    'CD4 TCM': 'CD4 T cells',
    'CD4 Naive': 'CD4 T cells',
    'CD4 TEM': 'CD4 T cells',
    'CD4 CTL': 'CD4 T cells',
    'CD4 Proliferating': 'CD4 T cells',
    'Treg': 'CD4 T cells',

    # CD8 T cell group
    'CD8 TEM': 'CD8 T cells',
    'CD8 Naive': 'CD8 T cells',
    'CD8 TCM': 'CD8 T cells',
    'CD8 Proliferating': 'CD8 T cells',

    # NK cells
    'NK': 'NK cells',
    'NK_CD56bright': 'NK cells',
    'NK Proliferating': 'NK cells',

    # B cells
    'B naive': 'B cells',
    'B memory': 'B cells',
    'B intermediate': 'B cells',
    'Plasmablast': 'B cells',

    # Monocytes
    'CD14 Mono': 'Monocytes',
    'CD16 Mono': 'Monocytes',

    # Dendritic cells
    'cDC1': 'Conventional DCs',
    'cDC2': 'Conventional DCs',
    'ASDC': 'Conventional DCs',
}
all_celltypes = ["CD4 T cells", 'CD8 T cells', 'NK cells', 'B cells', 'Monocytes', 'Conventional DCs']
# Replace labels with coarse-grained ones where applicable
#dd.C.obs[celltype_key] = dd.C.obs[celltype_key].replace(aggregation_map)
###

def _get_ensembl_gene_id_start_end_chr():
    from pybiomart import Server
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    ensembl_gene_id_start_end_chr = dataset.query(attributes=['ensembl_gene_id', 'start_position', 'end_position', 'chromosome_name'])
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.set_index("Gene stable ID")
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.rename(columns={
        "Gene start (bp)": GAnn.start,
        "Gene end (bp)": GAnn.end,
        "Chromosome/scaffold name": GAnn.chrom,
    })
    return ensembl_gene_id_start_end_chr

ensembl_gene_id_start_end_chr = _get_ensembl_gene_id_start_end_chr()
ensembl_gene_id_start_end_chr

dd.C.var = dd.C.var.join(ensembl_gene_id_start_end_chr)
dd.C.obs[DAnn.donor] = dd.C.obs[original_donor_col]
dd.G.obsm["gPCs"] = dd.G.obsm["gPCs"][dd.G.obsm["gPCs"].columns[:n_gpcs]]

sc.pp.normalize_total(dd.C)
sc.pp.log1p(dd.C)
sc.pp.normalize_total(dd.C)

# are the expression pcs computed by pseudobulking across all cell types?
mdata = sc.get.aggregate(dd.C, by=DAnn.donor, func="mean")
mdata.X = mdata.layers.pop("mean")

sc.pp.highly_variable_genes(mdata, n_top_genes=batch_e_pcs_n_top_genes)
sc.tl.pca(mdata, n_comps=n_epcs)

dd.G.obsm["ePCs"] = mdata[dd.G.obs_names].obsm["X_pca"]

#######
for cell_type in tqdm(np.unique(dd.C.obs[celltype_key])): #tqdm(all_celltypes):  
    print(cell_type)
    dd_celltype = dd.copy()
    dd_celltype = dd_celltype[..., dd_celltype.C.obs[celltype_key] == cell_type, :].copy()
    dd_celltype
    pb_gex_key = f"PB_{cell_type}" 

    gc.collect()

    dd_celltype.aggregate(key_added=pb_gex_key, sync_var=True, verbose=True)
    dd_celltype.aggregate(obs=["sex", "age"], func="first", add_to_obs=True)
    dd_celltype

    print(f"{pb_gex_key} shape:", dd_celltype.G.obsm[pb_gex_key].shape)
    print("dd_celltype.shape:", dd_celltype.shape)

    keep_genes = ((dd_celltype.G.obsm[pb_gex_key] > 0).mean(axis=0) >= min_percent_donors_expressed).values
    dd_celltype = dd_celltype[..., keep_genes]
    print("after filtering")
    print(f"{pb_gex_key} shape:", dd_celltype.G.obsm[pb_gex_key].shape)
    print("dd_celltype.shape:", dd_celltype.shape)

    for chrom in tqdm(list(range(1, 23))[::-1]):
        print(chrom)
        # alternative to dd[:, dd.G.var.chrom == str(chrom), :, dd.C.var.chrom == str(chrom)]
        dd_chrom = dd_celltype.copy()
        dd_chrom = dd_chrom.sel(G_var=dd_chrom.G.var.chrom == str(chrom), C_var=dd_chrom.C.var.chrom == str(chrom)).copy()
        dd_chrom

        vep_annotation_file = DATA / f"variant_annotation/variants_vep_annotated_chr{chrom}.txt"

        cl.tl.add_vep_annos_to_gdata(vep_anno_file=vep_annotation_file, gdata=dd_chrom.G, dummy_consequence=True)
        dd_chrom.G.uns["variant_annotation_vep"]

        cl.tl.aggregate_annotations_for_varm(
            dd_chrom.G, "variant_annotation_vep", agg_type="first", return_data=True
        )  # TODO change agg type

        burden_agg_fct = "sum"
        run_lrt = True
        annotation_cols = ["maf_beta"] #["CADD_RAW", "maf_beta", "tss_distance", "tss_distance_exp"]

        rare_maf_threshold = 0.05

        dd_chrom = dd_chrom.sel(G_var=dd_chrom.G.var.maf < rare_maf_threshold).copy()
        dd_chrom

        dd_chrom.G.varm["variant_annotation"]["maf_beta"] = beta_weighting(dd_chrom.G.var["maf"])

        # This specifies covariates/fixed effects
        F = np.concatenate(
            [
                np.ones((dd_chrom.shape[0], 1)),
                dd_chrom.G.obs[["sex"]].values - 1,
                dd_chrom.G.obs[["age"]].values,
                dd_chrom.G.obsm["gPCs"].values,
                dd_chrom.G.obsm["ePCs"],
            ],
            axis=1,
        )
        F[:, 2:] = column_normalize(F[:, 2:].astype("float"))

        F = F.astype("float")

        results = []
        if isinstance(dd_chrom.G.X, da.Array | ad._core.views.DaskArrayView):
            if dd_chrom.G.is_view:
                dd_chrom._G = dd_chrom._G.copy()
            dd_chrom.G.X = dd_chrom.G.X.compute()

        if do_debug:
            warnings.filterwarnings("ignore", category=RuntimeWarning)


        for gene, row in tqdm(dd_chrom.C.var.iterrows(), total=dd_chrom.shape[3]):
            Y = gaussianize(dd_chrom.G.obsm[pb_gex_key][[gene]].values.astype("float") + 1e-5 * np.random.randn(dd_chrom.shape[0], 1))

            start = max(0, row.start - cis_window)
            end = row.end + cis_window
            _G = dd_chrom.G[:, (dd_chrom.G.var.pos < end)]
            _G = _G[:, (_G.var.pos > start)]
            _G = _G[:, (_G.X.std(0) != 0)]
            _G = _G.copy()

            # TODO make strand aware
            #_G.varm["variant_annotation"]["tss_distance"] = np.abs(row.start - _G.var["pos"])
            #_G.varm["variant_annotation"]["tss_distance_exp"] = np.exp(-1e-5 * _G.varm["variant_annotation"]["tss_distance"])

            rdf = run_burden_test(
                _G, Y, F, gene, annotation_cols=annotation_cols, burden_agg_fct=burden_agg_fct, run_lrt=run_lrt
            )
            results.append(rdf)

        rdf = pd.concat(results)
        rdf

        #print((rdf.pv < 0.05).sum())

        rdf.to_csv(f"burden_test_maf_beta_{cell_type}_chr{chrom}.csv")
        

##############################

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
# Only keep valid p-values
rdf_clean = rdf.dropna(subset=["pv"])

# Sort by significance
top_genes = rdf_clean.sort_values("pv").head(20)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_genes, x=-np.log10(top_genes["pv"]), y="egene", palette="viridis")
plt.xlabel("-log10(p-value)")
plt.ylabel("Gene")
plt.title("Top Burden Test Hits")
plt.tight_layout()
plt.show()

###

plt.figure(figsize=(8, 6))
sns.scatterplot(data=rdf_clean, x="beta", y=-np.log10(rdf_clean["pv"]), alpha=0.7)
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
plt.xlabel("Effect size (beta)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot of Burden Test Results")
plt.legend()
plt.tight_layout()
plt.show()

###

rdf["cell_type"] = current_cell_type  # e.g. "T-cell", "B-cell", etc.
all_rdfs.append(rdf)
df_all = pd.concat(all_rdfs, ignore_index=True)

# Filter for significant hits
df_sig = df_all[df_all["pv"] < 0.05]

plt.figure(figsize=(10, 8))
sns.stripplot(data=df_sig, x="cell_type", y=-np.log10(df_sig["pv"]), jitter=True, alpha=0.5)
plt.ylabel("-log10(p-value)")
plt.title("Burden Test Significance Across Cell Types")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###

pivot = df_all.pivot_table(index="egene", columns="cell_type", values="pv", aggfunc="min")
heatmap_data = -np.log10(pivot)

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap="viridis", linewidths=0.5, linecolor='gray')
plt.title("-log10(p-value) for Burden Test per Gene and Cell Type")
plt.xlabel("Cell Type")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

"""