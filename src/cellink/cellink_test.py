import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np

if __name__ == "__main__":

    a = 5
    import sys
    import os
    sys.path.append('/Users/larnoldt/sc-genetics/src')
    sys.path.append('/Users/larnoldt/sc-genetics/src/cellink')
    
    import cellink

    ######

    from cellink._core.donordata import DonorData
    from cellink.io import read_sgkit_zarr
    from pathlib import Path
    from anndata import read_h5ad

    DATA = Path("../tests/data")

    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    # gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")

    ######

    from cellink.tl._simulate_genotype_data import simulate_genotype_data_msprime, simulate_genotype_data_numpy

    simulate_genotype_data_msprime(10, 20)
    geno = simulate_genotype_data_numpy(10, 20)

    from cellink.pp._variant_qc import variant_qc

    variant_qc(geno, inplace=False)

    from cellink.tl._encode_genotype_data import one_hot_encode_genotypes, dosage_per_strand

    one_hot_encode_genotypes(geno)
    dosage_per_strand(geno)
