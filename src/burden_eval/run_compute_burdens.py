import argparse
from pathlib import Path
import pandas as pd
import scanpy as sc
import cellink as cl
from cellink.tl._burden_testing import *
from scipy.stats import beta
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='computeBurdens',
                    description='Run burden computing for provided chromosome')
    parser.add_argument('-c', '--chromosome', help='Enter target chromosome')
    parser.add_argument('-i', '--input_path', help='Enter path to target chromosome input file')
    parser.add_argument('-o', '--output_path', help='Enter path to target chromosome output file')
    args = parser.parse_args()

    #print(f"-c:{args.chromosome}, -i: {args.input_path}, -o: {args.output_path}")
    #sys.exit()

    # READ FILES
    data_file = args.input_path
    data = pd.read_pickle(data_file)

    # RUN BURDEN TESTING
    print(f"START burden computing for chr{args.chromosome}...")
    #results = compute_burdens(data, max_af=0.05, weight_cols=["DISTANCE", "CADD_PHRED", "DNA_LM_up", "DNA_LM_down", "MAF_beta_1.25"], window_size=100000, DNA_LM_up="DNA_LM_up", DNA_LM_down="DNA_LM_down")
    all_burdens = compute_burdens(data, max_af=0.05, weight_cols=["CADD_PHRED", "DNA_LM_up", "DNA_LM_down", "MAF_beta_1.25"], window_size=100000, DNA_LM_up="DNA_LM_up", DNA_LM_down="DNA_LM_down")
    print(f"DONE with burden computing for chr{args.chromosome}...")

    # WRITE RESULTS
    print(f"SAVE ALL BURDENS OF CHR{args.chromosome} TO {args.output_path}")
    res_path = args.output_path
    all_burdens.to_parquet(res_path)



