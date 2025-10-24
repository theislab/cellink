import gc
import warnings

import anndata as ad
import scanpy as sc
import dask.array as da
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from cellink._core import DAnn, GAnn
from cellink.at.gwas import GWAS
from cellink.utils import column_normalize, gaussianize
from cellink.resources import get_onek1k

import requests
import os
import tarfile
from tqdm import tqdm

n_gpcs = 20
n_epcs = 15
batch_e_pcs_n_top_genes = 2000
chrom = 22
cis_window = 500_000
# cell_type = "CD8 Naive"
celltype_key = "predicted.celltype.l2"
# pseudobulk expression in dd.G.obsm[key_added]
original_donor_col = "donor_id"
min_percent_donors_expressed = 0.1
do_debug = False

dd = get_onek1k(config_path="./cellink/datasets/config/onek1k.yaml")
dd

###
aggregation_map = {
    # CD4 T cell group
    "CD4 TCM": "CD4 T cells",
    "CD4 Naive": "CD4 T cells",
    "CD4 TEM": "CD4 T cells",
    "CD4 CTL": "CD4 T cells",
    "CD4 Proliferating": "CD4 T cells",
    "Treg": "CD4 T cells",
    # CD8 T cell group
    "CD8 TEM": "CD8 T cells",
    "CD8 Naive": "CD8 T cells",
    "CD8 TCM": "CD8 T cells",
    "CD8 Proliferating": "CD8 T cells",
    # NK cells
    "NK": "NK cells",
    "NK_CD56bright": "NK cells",
    "NK Proliferating": "NK cells",
    # B cells
    "B naive": "B cells",
    "B memory": "B cells",
    "B intermediate": "B cells",
    "Plasmablast": "B cells",
    # Monocytes
    "CD14 Mono": "Monocytes",
    "CD16 Mono": "Monocytes",
    # Dendritic cells
    "cDC1": "Conventional DCs",
    "cDC2": "Conventional DCs",
    "ASDC": "Conventional DCs",
}
all_celltypes = ["CD4 T cells", "CD8 T cells", "NK cells", "B cells", "Monocytes", "Conventional DCs"]
# Replace labels with coarse-grained ones where applicable
dd.C.obs[celltype_key] = dd.C.obs[celltype_key].replace(aggregation_map)
###


def _get_ensembl_gene_id_start_end_chr():
    from pybiomart import Server

    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    ensembl_gene_id_start_end_chr = dataset.query(
        attributes=["ensembl_gene_id", "start_position", "end_position", "chromosome_name"]
    )
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.set_index("Gene stable ID")
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.rename(
        columns={
            "Gene start (bp)": GAnn.start,
            "Gene end (bp)": GAnn.end,
            "Chromosome/scaffold name": GAnn.chrom,
        }
    )
    return ensembl_gene_id_start_end_chr


ensembl_gene_id_start_end_chr = _get_ensembl_gene_id_start_end_chr()
ensembl_gene_id_start_end_chr

dd.C.var = dd.C.var.join(ensembl_gene_id_start_end_chr)
dd.C.obs[DAnn.donor] = dd.C.obs[original_donor_col]
dd.G.obsm["gPCs"] = dd.G.obsm["gPCs"][dd.G.obsm["gPCs"].columns[:n_gpcs]]

# gene_mask = np.array((dd.C.X > 0).sum(axis=0) / dd.C.X.shape[0] >= 0.01)[0]
# dd.C = dd.C[:, gene_mask]

sc.pp.normalize_total(dd.C)
sc.pp.log1p(dd.C)
sc.pp.normalize_total(dd.C)

mdata = sc.get.aggregate(dd.C, by=DAnn.donor, func="mean")
mdata.X = mdata.layers.pop("mean")

sc.pp.highly_variable_genes(mdata, n_top_genes=batch_e_pcs_n_top_genes)
sc.tl.pca(mdata, n_comps=n_epcs)

dd.G.obsm["ePCs"] = mdata[dd.G.obs_names].obsm["X_pca"]


def run_pseudobulk_eqtl():
    for cell_type in tqdm(
        all_celltypes
    ):  # Last: CD4 Proliferating. Next: 9 tqdm(np.unique(dd.C.obs[celltype_key])[9:]): #
        count = np.sum(dd.C.obs[celltype_key] == cell_type)

        if count >= 20:
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

            F = np.concatenate(
                [
                    np.ones((dd_celltype.shape[0], 1)),
                    dd_celltype.G.obs[["sex"]].astype("int").values - 1,
                    dd_celltype.G.obs[["age"]].astype("int").values,
                    dd_celltype.G.obsm["gPCs"].values,
                    dd_celltype.G.obsm["ePCs"],
                ],
                axis=1,
            )
            F[:, 2:] = column_normalize(F[:, 2:])
            F = F.astype(np.float64)

            for chrom in tqdm(list(range(1, 23))[::-1]):
                print(chrom)
                # alternative to dd[:, dd.G.var.chrom == str(chrom), :, dd.C.var.chrom == str(chrom)]
                dd_chrom = dd_celltype.copy()
                dd_chrom = dd_chrom.sel(
                    G_var=dd_chrom.G.var.chrom == str(chrom), C_var=dd_chrom.C.var.chrom == str(chrom)
                ).copy()
                dd_chrom

                results = []
                if isinstance(dd_chrom.G.X, da.Array | ad._core.views.DaskArrayView):
                    if dd_chrom.G.is_view:
                        dd_chrom._G = dd_chrom._G.copy()  # TODO: discuss with SWEs
                    dd_chrom.G.X = dd_chrom.G.X.compute()

                if do_debug:
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                for gene, row in tqdm(dd_chrom.C.var.iterrows(), total=dd_chrom.shape[3]):
                    Y = gaussianize(dd_chrom.G.obsm[pb_gex_key][[gene]].values.astype(np.float64))

                    start = max(0, row.start - cis_window)
                    end = row.end + cis_window
                    _G = dd_chrom.G[:, (dd_chrom.G.var.pos < end)]
                    _G = _G[:, (_G.var.pos > start)]
                    _G = _G[:, (_G.X.std(0) != 0)]
                    G = _G.X.astype(np.float64)

                    if G.shape[1] > 0:
                        gwas = GWAS(Y, F)
                        gwas.process(G)

                        snp_idx = gwas.getPv().argmin()

                        def _get_top_snp(arr, snp_idx=snp_idx):
                            return arr.ravel()[snp_idx].item()

                        rdict = {
                            "snp": _G.var.iloc[snp_idx].name,
                            "egene": gene,
                            "n_cis_snps": G.shape[1],
                            "pv": _get_top_snp(gwas.getPv()),
                            "beta": _get_top_snp(gwas.getBetaSNP()),
                            "betaste": _get_top_snp(gwas.getBetaSNPste()),
                            "lrt": _get_top_snp(gwas.getLRT()),
                        }
                        Gt0 = np.random.permutation(G[:, [snp_idx]])
                        gwas.process(Gt0)
                        rdict["pv0"] = gwas.getPv().item()

                        results.append(rdict)

                rdf = pd.DataFrame(results)
                rdf

                rdf["pv_adj"] = np.clip(rdf["pv"] * rdf["n_cis_snps"], 0, 1)  # gene-wise Bonferroni
                rdf["qv"] = fdrcorrection(rdf["pv_adj"])[1]

                print((rdf.qv < 0.05).sum())

                rdf.to_csv(f"pseudobulk_eqtl_{cell_type}_chr{chrom}.csv")


def analysis_yazar_eqtl():
    if not os.path.isfile("eqtl_table.tsv"):
        r = requests.get("https://onek1k.s3.ap-southeast-2.amazonaws.com/eqtl/eqtl_table.tsv.gz", allow_redirects=True)
        open("eqtl_table.tsv.gz", "wb").write(r.content)
        file = tarfile.open("eqtl_table.tsv.gz")
        file.extractall("eqtl_table.tsv")
        file.close()
    if not os.path.isfile("esnp_table.tsv"):
        r = requests.get("https://onek1k.s3.ap-southeast-2.amazonaws.com/eqtl/esnp_table.tsv.gz", allow_redirects=True)
        open("esnp_table.tsv.gz", "wb").write(r.content)
        file = tarfile.open("esnp_table.tsv.gz")
        file.extractall("esnp_table.tsv")
        file.close()


def load_data():
    results = []
    all_celltypes = ["CD4 T cells", "CD8 T cells", "NK cells", "B cells", "Monocytes", "Conventional DCs"]
    for cell_type in all_celltypes:  # glob.glob("pseudobulk_eqtl_*_chr1.csv"):
        # cell_type = cell_type.removeprefix("pseudobulk_eqtl_").removesuffix("_chr1.csv")
        results_cell_type = []
        for chrom in list(range(1, 23)):
            results_cell_type.append(pd.read_csv(f"./association/pseudobulk_eqtl_{cell_type}_chr{chrom}.csv"))
        results_cell_type = pd.concat(results_cell_type)
        results_cell_type["cell_type"] = cell_type
        results.append(results_cell_type)
    results = pd.concat(results)
    results["chrom"] = results["snp"].apply(lambda x: x.split("_")[0])
    results["pos"] = results["snp"].apply(lambda x: int(x.split("_")[1]))

    return results


if __name__ == "__main__":
    # run_pseudobulk_eqtl()
    # analysis_yazar_eqtl()

    from cellink.pl import expression_by_genotype, manhattan, qq

    all_celltypes = ["CD4 T cells", "CD8 T cells", "NK cells", "B cells", "Monocytes", "Conventional DCs"]

    dd.C = dd.C[dd.C.obs[celltype_key].isin(all_celltypes), :]

    results = load_data()
    results = results.join(dd.G.var["id_hg19"], on="snp")

    """
    for index, row in tqdm(results.iterrows()):

        dd_copy = dd.copy()

        fig = expression_by_genotype(
            dd=dd_copy,
            snp=row["snp"],
            gene_filter=[row["egene"]],
            celltype_key = "predicted.celltype.l2",
            #celltype_filter=['T cell', 'B cell'],
            mode='cell_type',
            n_cols=2,
            figsize=(8, 6),
            labelsize=11,
            titlesize=14,
            palette='pastel',
            plot_type="boxplot",
            show_stripplot=False,
            save=f"./plots/{row['snp']}.png"
            save=f"./plots/{row['snp']}.png"
        )
    """

    fig = qq(results, pval_col="pv_adj", group_col="cell_type", save="./plots_new/qq_pv_adj_2.png")
    fig = qq(results, pval_col="pv0", group_col="cell_type", save="./plots_new/qq_pv0_adj_2.png")

    fig = manhattan(
        results,
        pval_col="pv_adj",
        significance_threshold=2.655989298359627e-253,
        label_column="snp",
        figsize=(14, 6),
        labelsize=8,
        save="./plots_new/manhattan_pv_adj_with_threshold_FINAL.png",
    )
    fig = manhattan(
        results,
        pval_col="pv_adj",
        significance_threshold=2.655989298359627e-150,
        label_column="snp",
        figsize=(14, 6),
        labelsize=8,
        save="./plots_new/manhattan_pv_adj_with_threshold_LO WER_FINAL.png",
    )
    fig = manhattan(
        results,
        pval_col="pv_adj",
        label_column="snp",
        figsize=(14, 6),
        labelsize=8,
        save="./plots_new/manhattan_pv_adj_FINAL.png",
    )
    fig = manhattan(
        results,
        pval_col="pv0",
        significance_threshold=5e-5,
        label_column="snp",
        figsize=(14, 6),
        labelsize=8,
        save="./plots_new/manhattan_pv0_adj_FINAL.png",
    )

    for snp in ["1_1215852_T_C", "1_6502724_A_G", "1_16441466_G_A", "1_23436186_A_G", "1_25563181_G_A"]:
        print(results[results["snp"] == snp].iloc[0]["id_hg19"])

    esnps = [
        "4_17614591_C_A",
        "4_118279119_C_T",
        "5_65606264_C_T",
        "5_96916728_G_A",
        "6_32594017_C_G",
        "6_32619681_AGT_A",
        "7_22812880_C_T",
        "17_7306067_G_A",
        "17_7306067_G_A",
        "22_42027106_G_C",
        "22_42078134_C_G",
    ]
    esnps = [
        "1_26287295_GCCT_G",
        "1_109740350_G_T",
        "1_173551078_GC_G",
        "1_205824729_ATT_A",
        "2_264227_G_A",
        "2_10880776_G_A",
        "2_10880776_G_A",
        "2_196163068_T_G",
        "2_197307424_T_G",
        "2_200889340_T_C",
        "2_230174862_G_A",
        "4_17614331_G_C",
        "4_17614591_C_A",
        "4_17636041_AT_A",
        "4_118279119_C_T",
        "4_152097104_G_C",
        "5_65606264_C_T",
        "5_65606264_C_T",
        "5_96916728_G_A",
        "5_96916728_G_A",
        "5_96916728_G_A",
        "5_177448955_C_A",
        "6_26353872_G_T",
        "6_26353872_G_T",
        "6_29675201_C_CAACTTGAA",
        "6_29942211_C_T",
        "6_31186843_T_C",
        "6_31186843_T_C",
        "6_31197235_G_T",
        "6_31268539_G_T",
        "6_31268562_T_C",
        "6_31357458_CA_C",
        "6_31357458_CA_C",
        "6_31357458_CA_C",
        "6_32594017_C_G",
        "6_32594017_C_G",
        "6_32594017_C_G",
        "6_32594017_C_G",
        "6_32619681_AGT_A",
        "6_32643736_A_G",
        "6_32647644_A_G",
        "6_32648073_A_G",
        "6_32652622_G_T",
        "6_32655686_A_G",
        "6_32667083_G_GAC",
        "6_33272092_C_T",
        "6_37517194_A_C",
        "6_135497759_C_A",
        "6_166961685_G_C",
        "7_6845133_A_C",
        "7_22812880_C_T",
        "7_22812880_C_T",
        "7_107934339_G_A",
        "7_130366967_G_C",
        "7_150779462_A_G",
        "7_150779462_A_G",
        "8_2134028_G_C",
        "8_29051898_TAAAAAAAA_T",
        "8_144792397_C_T",
        "8_144792397_C_T",
        "9_34353881_G_T",
        "9_69818165_G_A",
        "9_133251979_C_T",
        "9_136757877_G_C",
        "10_70233338_T_G",
        "10_124921078_G_T",
        "10_124921078_G_T",
        "11_324163_C_T",
        "11_9430253_G_A",
        "11_18407090_A_AT",
        "11_57516515_C_T",
        "11_65877883_A_G",
        "11_74788779_G_A",
        "11_74892136_G_A",
        "11_118304724_G_A",
        "12_27695410_A_G",
        "12_27695410_A_G",
        "12_31096184_A_G",
        "12_52677598_A_C",
        "12_68451756_A_G",
        "13_43023570_C_T",
        "14_49633965_C_G",
        "14_91873506_G_A",
        "15_33103578_G_C",
        "15_50337206_T_G",
        "15_82538000_C_G",
        "16_28826194_T_C",
        "17_7304645_A_C",
        "17_7304645_A_C",
        "17_7306067_G_A",
        "17_7306067_G_A",
        "17_45693629_G_A",
        "17_45693629_G_A",
        "17_46678363_A_C",
        "19_35742022_T_C",
        "19_55385959_A_G",
        "19_57428251_C_T",
        "20_47560968_C_CT",
        "20_47560968_C_CT",
        "20_63975807_C_T",
        "21_29067909_C_A",
        "21_44049878_C_CTT",
        "21_44056665_A_G",
        "21_44072034_G_T",
        "21_44908184_T_C",
        "21_46540580_T_C",
        "21_46540580_T_C",
        "21_46541798_G_A",
        "22_42027106_G_C",
        "22_42078134_C_G",
        "22_42094877_T_TAA",
        "22_45333097_G_C",
        "22_45335849_T_C",
        "22_46760806_T_C",
    ]

    results_filt = results[results["snp"].isin(esnps)]

    for index, row in tqdm(results_filt.iterrows()):
        dd_copy = dd.copy()

        fig = expression_by_genotype(
            dd=dd_copy,
            snp=row["snp"],
            gene_filter=[row["egene"]],
            celltype_key="predicted.celltype.l2",
            # celltype_filter=['T cell', 'B cell'],
            mode="cell_type",
            n_cols=2,
            figsize=(8, 6),
            labelsize=11,
            titlesize=14,
            palette="pastel",
            plot_type="boxplot",
            show_stripplot=False,
            save=f"./plots_new/{row['snp']}_{row['egene']}.png",
        )
        print(results[results["snp"] == row["snp"]].iloc[0]["id_hg19"])

    for index, row in tqdm(results_filt.iterrows()):
        print(row["snp"])
        print(results[results["snp"] == row["snp"]].iloc[0]["id_hg19"])
        print("###")
