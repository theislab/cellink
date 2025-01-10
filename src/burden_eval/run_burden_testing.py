import argparse
from pathlib import Path
import os
import pandas as pd
import scanpy as sc
import cellink as cl
from cellink.tl._burden_testing import *

def preprocess_scdata(scdata):
    scdata = scdata.copy() # don't mess with view changes just in case
    sc.pp.normalize_total(scdata, target_sum=1e4)  # Normalize total counts per cell
    sc.pp.log1p(scdata)  # Apply log-transform
    return scdata

if __name__ == '__main___':

    parser = argparse.ArgumentParser(
                    prog='burdenTesting',
                    description='Run burden testing for provided chromosome')
    parser.add_argument('-c', '--chromosome', help='Enter target chromosome')
    #parser.add_argument('-z', '--zarrfile', help='Path to zarr file')
    
    args = parser.parse_args()

    cell_type_col = "cell_label"

    # set paths
    base_data_dir = Path("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/")
    scdata_path = base_data_dir / "input_data/OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad.gz"
    gdata_dir = "/data/ceph/hdd/project/node_09/sys_gen_students/2024_2025/project04_rare_variant_sc/input_data/filter_vcf_r08/"
    zarr_file = os.path.join(gdata_dir, f"chr{str(args.chromosome)}.dose.filtered.R2_0.8.vcz")
    eigenvec = pd.read_csv(base_data_dir / "input_data/pcdir/wgs.dose.filtered.R2_0.8.filtered.pruned.eigenvec", sep = ' ')

    # read objects
    scdata = sc.read_h5ad(scdata_path)
    gdata = cl.io.read_sgkit_zarr(args.zarrfile)

    # perform normalization and log transformation
    scdata = preprocess_scdata(scdata)

    # create data object
    data = cl.DonorData(adata=scdata, gdata=gdata, donor_key_in_sc_adata="individual")

    # run burden testing for specified chromosome
    results = compute_burdens(data, max_af=0.05, weight_cols=["DISTANCE", "CADD_PHRED", "DNA_LM_influence_score", "MAF_beta_1.25"], window_size=100000)
    
    # write results
    res_path = output_dir/f"chr{args.chromosome}_all_results_DNA_LM_and_MAF_100k.pkl"
    with open(res_path, "wb") as file:
        all_res = pickle.dump(results, file)



