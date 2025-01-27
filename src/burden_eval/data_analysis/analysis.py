import pandas as pd
from pathlib import Path
import os
import pandas as pd
import scanpy as sc
import cellink as cl
from cellink.tl._burden_testing import *
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

def preprocess_scdata(scdata):
    scdata = scdata.copy()  # don't mess with view changes just in case
    scdata.layers["raw"] = scdata.X.copy()
    sc.pp.normalize_total(scdata, target_sum=1e4)  # Normalize total counts per cell
    sc.pp.log1p(scdata)  # Apply log-transform
    scdata.layers["log1p"] = scdata.X.copy()
    return scdata

def create_umap(scdata):
    scdata = scdata.copy()

    sc.pp.highly_variable_genes(scdata, n_top_genes=2000)
    sc.pp.scale(scdata, max_value=10)
    sc.tl.pca(scdata, svd_solver='arpack', use_highly_variable=True)
    #sc.pp.neighbors(scdata, n_neighbors=10, n_pcs=5)  
    sc.pp.neighbors(scdata, n_neighbors=10, n_pcs=50)  
    sc.tl.umap(scdata)
    return scdata

def compute_marker_genes(scdata):
    scdata = scdata.copy()
    sc.tl.rank_genes_groups(scdata, 'cell_label', method='wilcoxon', layer="log1p", use_raw=False) 
    marker_genes = scdata.uns['rank_genes_groups']
    return scdata

def plots(scdata, out_path):
    sc.pl.umap(scdata, color='cell_label', save=f"umap_by_cell_type.png")
    sc.pl.rank_genes_groups(scdata, n_genes=20, sharey=False, save=f"rank_genes.png")
    sc.pl.rank_genes_groups_dotplot(scdata, n_genes=5, save=f"dotplot_rank_genes.png")
    sc.pl.highest_expr_genes(scdata, n_top=20,save=f"highest_expr_genes.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='dataAnnotation',
                    description='Run data annotation for provided chromosome')
    parser.add_argument('-i', '--input_path', help='Enter path to target chromosome data file')
    parser.add_argument('-o', '--output_path', help='Enter path to target chromosome data file')

    args = parser.parse_args()
    scdata_path = args.input_path

    print("reading scdata")
    scdata = sc.read_h5ad(scdata_path)
    
    #print("preparing scdata")
    # for testing: subset on chr
    #scdata = scdata[:, scdata.var["chromosome"] == "22"]
    print("preprocessing scdata")
    scdata = preprocess_scdata(scdata)
    print("create umap")
    scdata = create_umap(scdata)
    print("compute marke genes")
    scdata = compute_marker_genes(scdata)
    
    print("saving scdata")
    scdata.write_h5ad(f"{args.output_path}/analysis_adata.h5ad")

    sc.settings.figdir = args.output_path
    print("create plots")
    plots(scdata)

        

    





    
