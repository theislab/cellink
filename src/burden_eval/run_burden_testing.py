import argparse
from pathlib import Path
import os
import pandas as pd
import scanpy as sc
import cellink as cl
from cellink.tl._burden_testing import *
from scipy.stats import beta
import sys
import pickle

def preprocess_scdata(scdata):
    scdata = scdata.copy() # don't mess with view changes just in case
    sc.pp.normalize_total(scdata, target_sum=1e4)  # Normalize total counts per cell
    sc.pp.log1p(scdata)  # Apply log-transform
    return scdata
    

def add_snp_id(DNA_LM):
    DNA_LM['snp_id'] = DNA_LM['Chromosome'] + "_" + DNA_LM['pos'].astype(str) + "_" + DNA_LM['ref'] + "_" + DNA_LM['alt']
    DNA_LM['snp_id'] = DNA_LM['snp_id'].str.replace('chr', '')
    
    # Set 'snap_id' as the index
    DNA_LM.set_index('snp_id', inplace=True)
    return DNA_LM

def reverse_and_update_snp_ids(gdata_df, dna_df):
    updated_index = []
    
    for snp_id in dna_df.index:
        if snp_id in gdata_df.index:
            updated_index.append(snp_id)
        else:
            chrom, pos, ref, alt = snp_id.split("_")
            reversed_snp_id = f"{chrom}_{pos}_{alt}_{ref}"  # Reverse ref and alt
            
            # Check if reversed_snp_id exists in data_df
            if reversed_snp_id in gdata_df.index:
                #print(f"Reversing {snp_id} to {reversed_snp_id}")
                updated_index.append(reversed_snp_id)
            else:
                print(f"Error, unknown snp_id {snp_id}")
                updated_index.append(snp_id)
    
    # Update DNA_LM's index
    dna_df.index = updated_index
    print("\nUpdated DNA_LM index:")
    print(dna_df.index)
    return dna_df


def add_maf_annotation(gdata):
    weighted_snp_maf = beta.pdf(gdata.var["maf"], 1, 25)
    gdata.varm["annotations_0"]["MAF_beta_1.25"] = weighted_snp_maf
    return gdata

def add_DNA_LM(gdata, file, chromosome, colname):

    DNA_LM = pd.read_csv(file,
                sep = '\t')
    DNA_LM = add_snp_id(DNA_LM)
    DNA_LM = reverse_and_update_snp_ids(gdata.varm["annotations_0"],
                                        DNA_LM[DNA_LM["Chromosome"]==f"chr{chromosome}"])
    
    gdata.varm["annotations_0"][colname] = DNA_LM["influence_score"].reindex(gdata.varm["annotations_0"].index)

    # Rename the merged column if needed
    gdata.varm["annotations_0"].rename(columns={"influence_score": colname}, inplace=True)
    return gdata



if __name__ == "__main__":
    #print("hi")

    parser = argparse.ArgumentParser(
                    prog='burdenTesting',
                    description='Run burden testing for provided chromosome')
    parser.add_argument('-c', '--chromosome', help='Enter target chromosome')
    parser.add_argument('-i', '--input_path', help='Enter path to target chromosome input file')
    parser.add_argument('-o', '--output_path', help='Enter path to target chromosome output file')
    args = parser.parse_args()

    #print(f"-c:{args.chromosome}, -i: {args.input_path}, -o: {args.output_path}")
    #sys.exit()

    # SET PATHS
    base_data_dir = Path("/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/")
    scdata_path = base_data_dir / "input_data/OneK1K_cohort_gene_expression_matrix_14_celltypes_w_gene_locations.h5ad.gz"
    gdata_dir = "/data/ceph/hdd/project/node_09/sys_gen_students/2024_2025/project04_rare_variant_sc/input_data/filter_vcf_r08/"

    # READ FILES
    zarr_file = args.input_path
    #zarr_file = os.path.join(gdata_dir, f"chr{str(args.chromosome)}.dose.filtered.R2_0.8.vcz")
    eigenvec = pd.read_csv(base_data_dir / "input_data/pcdir/wgs.dose.filtered.R2_0.8.filtered.pruned.eigenvec", sep = ' ')
    DNA_LM_upstream = base_data_dir/ "input_data/annotations/onek1k_inf_scores_upstream_model.tsv"
    DNA_LM_downstream = base_data_dir/ "input_data/annotations/onek1k_inf_scores_downstream_model.tsv"
    # ---------------------------
    # TODO REMOVE TEST WITH CHR22
    # ---------------------------
    #vep_scores = base_data_dir/ "input_data/annotations/onek1k1_all_variants_annotated_vep.txt"
    vep_scores = base_data_dir/ "input_data/annotations/onek1k1_chr22_variants_annotated_vep.txt"

    print(f"reading scdata")
    scdata = sc.read_h5ad(scdata_path)
    print(f"reading gdata")
    gdata = cl.io.read_sgkit_zarr(zarr_file)

    print(f"preparing scdata")
    # PERFORM NORMALIZATION AND LOG TRANSFORMATION
    scdata = scdata[:,scdata.var["chromosome"] == str(args.chromosome)]
    # ----------
    #scdata.var["chrom"] = args.chromosome #current fix before chromosome is added to object
    scdata = preprocess_scdata(scdata)

    # ANNOTATIONS
    print(f"add vep annotation to gdata ")
    # add vep annotation to gdata 
    cl.tl.add_vep_annos_to_gdata(vep_scores, gdata,
                             cols_to_explode=["Consequence"],
                             cols_to_dummy=["Consequence"])

    # add maf annotaion to gdata
    print(f"add maf annotation to gdata ")
    gdata = add_maf_annotation(gdata)

    # add DNA_LM annotations (downstream and upstream models) to gdata 
    print(f"add DNA_LM annotation to gdata ")
    gdata = add_DNA_LM(gdata, file=DNA_LM_upstream, chromosome=args.chromosome, colname='DNA_LM_up')
    gdata = add_DNA_LM(gdata, file=DNA_LM_downstream, chromosome=args.chromosome, colname='DNA_LM_down')
    
    # CREATE DATA OBJ
    print(f"CREATE DATA OBJ")
    data = cl.DonorData(adata=scdata, gdata=gdata, donor_key_in_sc_adata="individual")

    # RUN BURDEN TESTING
    print(f"start burden computing for chr{args.chromosome}...")
    #results = compute_burdens(data, max_af=0.05, weight_cols=["DISTANCE", "CADD_PHRED", "DNA_LM_up", "DNA_LM_down", "MAF_beta_1.25"], window_size=100000, DNA_LM_up="DNA_LM_up", DNA_LM_down="DNA_LM_down")
    results = compute_burdens(data, max_af=0.05, weight_cols=["CADD_PHRED", "DNA_LM_up", "DNA_LM_down", "MAF_beta_1.25"], window_size=100000, DNA_LM_up="DNA_LM_up", DNA_LM_down="DNA_LM_down")
    print(f"done with burden computing for chr{args.chromosome}...")
    
    # WRITE RESULTS
    res_path = args.output_path
    with open(res_path, "wb") as file:
        all_res = pickle.dump(results, file)



