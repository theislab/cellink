import argparse
from pathlib import Path
import os
import pandas as pd
import scanpy as sc
import cellink as cl
from cellink.tl._burden_testing import *
from scipy.stats import beta
import pickle


def get_gene_location(ensembl_id):
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        #print(data)
        if data.get('strand') == 1: 
            # forward: this means start ----> end (start < end)
            chrom = data.get('seq_region_name')
            start = data.get('start')
            end = data.get('end')
            return chrom, start, end
        else:
            # reverse: this means end <---- start (start > end, on the scale of the forward strand)
            chrom = data.get('seq_region_name')
            start = data.get('end')
            end = data.get('start')
            return chrom, start, end
    else:
        chrom = np.nan
        start = np.nan
        end = np.nan
        return chrom, start, end

base_data_dir = Path("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/")
scdata_path = base_data_dir / "input_data/OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad.gz"
    
scdata = sc.read_h5ad(scdata_path)


scdata.var[['chromosome', 'start', 'end']] = scdata.var.index.to_series().apply(
    lambda x: pd.Series(get_gene_location(x))
)
res_path =  base_data_dir / "input_data/OneK1K_cohort_gene_expression_matrix_14_celltypes_w_gene_locations.h5ad.gz"

scdata.write_h5ad(res_path)

# with open(res_path, "wb") as file:
#         all_res = pickle.dump(scdata, file)



    
