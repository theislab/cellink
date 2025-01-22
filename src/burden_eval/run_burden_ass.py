import pandas as pd
import cellink as cl
from cellink.tl._burden_testing import *
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='burdenTesting',
                    description='Run burden association testing for provided chromosome')
    parser.add_argument('-c', '--chromosome', help='Enter target chromosome')
    parser.add_argument('-d', '--data', help='Enter path to data pickle for target chromosome')
    parser.add_argument('-b', '--burdens', help='Enter path to burdens for target chromosome')
    parser.add_argument('-o', '--output', help='Enter path to save output for target chromosome')
    parser.add_argument('-e', '--eigenvector', help='Enter path to eigenvector file.')
    parser.add_argument('--dump', help='Enter path to dump folder.')
    args = parser.parse_args()

    # read files
    data = pd.read_pickle(args.data)

    all_burdens = pd.read_parquet(args.burdens)
    eigenvec = pd.read_csv(args.eigenvector, sep=' ', header=None)
    eigenvec.index = eigenvec[1]
    eigenvec = eigenvec.iloc[:, 2:]
    print(f'eigenvec: {eigenvec.index}')
    print(f'allburdens 0: {all_burdens[0].index}')
    eigenvec = eigenvec[eigenvec.index.isin(all_burdens[0].index)]
    
    all_res = []
    # get cell_types
    cell_types = data.adata.obs["cell_label"].unique()
    for target_cell_type in cell_types[0:2]:
        print(target_cell_type)
        this_res = burden_test(
                donor_data=data,
                gene_burdens=all_burdens,
                target_cell_type=target_cell_type,
                target_chromosome=args.chromosome,
                eigenvector_df=eigenvec,
                dump_dir=args.dump,
                # target_genes = target_genes,
                #transforms_seq=None #  TODO comment this back in to quantile transform phenotype. Commented out to make testing faster
                )
        all_res.append(this_res[0])
    all_res = pd.concat(all_res)

    with open(args.output, "wb") as file:
        pickle.dump(all_res, file)
