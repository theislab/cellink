import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import random

## simulated sc anndata 
n_cells = 5000  
n_genes = 500  
n_samples = 1000
n_variants = 2000


## genotype data  # still has to be fixed!! 
adata = sgkit.simulate_genotype_call_dataset(n_variants, n_samples)
# os.makedirs("../tests/data_new/")
sgkit.io.plink.write_plink(adata, path="simulated_genotype_calls")
adata.to_zarr("simulated_genotype_calls.zarr")
sgkit.io.vcf.read_vcf("simulated_genotype_calls")
plink --bfile simulated_genotype_calls --recode vcf --out simulated_genotype_calls
bgzip simulated_genotype_calls.vcf
tabix -p vcf simulated_genotype_calls.vcf.gz
vcf2zarr explode simulated_genotype_calls.vcf.gz simulated_genotype_calls.icf
vcf2zarr encode simulated_genotype_calls.icf simulated_genotype_calls.vcz



## sc data 

individual = [f"0_S{random.randint(0, n_samples-1)}" for i in range(n_cells)]  #ids in the same format is in the genotypes 
age = {key: random.randint(50, 80) for key in set(individual)}
age = [age[key] for key in individual]
indvidual = pd.Categorical(individual)


expression_matrix = np.random.poisson(lam=5, size=(n_cells, n_genes))

cell_types = np.random.choice(['CD4 NC', 'CD8 NC', 'NK'], size=n_cells)
cell_metadata = pd.DataFrame({'cell_label': cell_types, "age": age, "individual": pd.Categorical(individual)}, index=[f'Cell_{i}' for i in range(n_cells)])

gene_metadata = pd.DataFrame(index=[f'Gene_{i}' for i in range(n_genes)])

adata = ad.AnnData(X=expression_matrix, obs=cell_metadata, var=gene_metadata)
adata.write_h5ad("simulated_gene_expression.h5ad")
