from pathlib import Path
import os
import pandas as pd
import scanpy as sc
import cellink as cl
import pickle

from cellink.tl._burden_testing import *


base_data_dir = Path("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/")
output_dir = base_data_dir / "output"

# data
data = pd.read_pickle(output_dir/"data_preprocessed.pkl")

results = compute_burdens(data, max_af=0.05, weight_cols=["DISTANCE", "CADD_PHRED", "DNA_LM_influence_score", "MAF_beta_1.25"], window_size=100000)

# write results
res_path = output_dir/"all_results_DNA_LM_and_MAF_100k.pkl"
with open(res_path, "wb") as file:
    all_res = pickle.dump(results, file)