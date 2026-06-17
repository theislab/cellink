# %% [markdown]
# # Tutorial: Donor Effect Decomposition with LIVI using `cellink`
# 
# This tutorial demonstrates how to use **LIVI** (Latent Interaction Variational Inference) through the `cellink` package to decompose single-cell gene expression into cell-state-specific and persistent donor effects.
# 
# LIVI is a Variational Autoencoder (VAE)-based model that jointly learns:
# - **Cell-state latent factors** (`z`): shared across donors, capturing transcriptional programs.
# - **Donor × cell-state interaction factors** (`DxC`): how each donor's gene expression changes depending on the cell state they are in.
# - **Persistent donor factors** (`V`): cell-state-independent donor effects (e.g. driven by population structure or persistent eQTLs).
# 
# After training, the learned donor embeddings can be tested for association with genotype data to identify trans-genetic effects on cell state. The `cellink` wrapper handles data preparation, training, inference and association testing directly from `DonorData`.
# 
# For installation of the LIVI repository, please refer to [https://github.com/PMBio/LIVI](https://github.com/PMBio/LIVI). The `cellink` wrapper requires the LIVI repository to be cloned locally; it does **not** need to be installed as a package.

# %% [markdown]
# ## Environment Setup
# 
# We import the necessary libraries and define key analysis parameters. The only configuration LIVI requires beyond standard `cellink` setup is the path to the cloned LIVI repository.

# %%
import numpy as np
import pandas as pd
import scanpy as sc

import cellink as cl
from cellink.resources import get_dummy_onek1k
from cellink.tl.external import (
    configure_livi_runner,
    infer_livi,
    load_livi_results,
    run_livi_association_testing,
    save_livi_results,
    train_livi,
)

# Path to the cloned LIVI repository
LIVI_ROOT = "LIVI"

# Analysis parameters
n_gpcs = 20
celltype_key = "predicted.celltype.l2"
individual_col = "donor_id"

# %% [markdown]
# ## Configure the LIVI Runner
# 
# The `LIVIRunner` manages the connection to the LIVI repository and the compute device. It needs to be configured once per session. The `execution_mode` controls how LIVI is invoked:
# 
# - `"python_api"` *(recommended)*: imports LIVI's PyTorch Lightning classes directly into the current Python process — no Hydra, no subprocess, results are Python objects.
# - `"subprocess"`: runs LIVI's `src/train.py` CLI as a child process using Hydra configs; better suited for isolated HPC job submission.
# 
# The `device` parameter is set to `"auto"`, which detects a GPU if available and falls back to CPU otherwise.

# %%
runner = configure_livi_runner(
    livi_root=LIVI_ROOT,
    execution_mode="python_api",
    device="auto",
)
print(f"LIVI root: {runner.livi_root}")
print(f"Compute device: {runner.resolve_device()}")

# %% [markdown]
# ## Load and Prepare Data
# 
# We load the dummy OneK1K dataset, which contains both genotype and single-cell expression data for 100 donors. LIVI operates on the **cell-level AnnData** (`dd.C`) and requires raw counts. When a `DonorData` object is passed to LIVI functions, `dd.C` is used automatically and the `donor_id` key is inferred from `dd.donor_id`.

# %%
dd = get_dummy_onek1k(
    config_path="../../src/cellink/resources/config/dummy_onek1k.yaml",
    verify_checksum=False,
)
print(f"Dataset shape: {dd.shape}")
print(f"Cells: {dd.C.shape}, Donors: {dd.G.shape[0]}")

# Restrict genotype PCs to the first n_gpcs components
dd.G.obsm["gPCs"] = dd.G.obsm["gPCs"][dd.G.obsm["gPCs"].columns[:n_gpcs]]

# %% [markdown]
# ## Data Preprocessing for LIVI
# 
# LIVI expects **raw integer counts** as input. We verify the data and optionally subset to a cell type of interest. LIVI handles all normalisation internally via learnable per-cell library-size factors.

# %%
# Propagate donor-level metadata (sex, age, pool batch) down to the cell level
dd.aggregate(obs=["donor_id", "sex", "age"], func="first", add_to_obs=True)

# Confirm that adata.X contains raw counts (LIVI requirement)
import scipy.sparse as sp
X_sample = dd.C.X[:100]
if sp.issparse(X_sample):
    X_sample = X_sample.toarray()
assert np.all(np.mod(X_sample, 1) == 0), "Non-integer values found — LIVI requires raw counts!"
print("Raw count check passed.")
print(f"Cells: {dd.C.shape[0]}, Genes: {dd.C.shape[1]}, Donors: {dd.G.shape[0]}")

# %% [markdown]
# We optionally filter to a single cell type. Fitting LIVI per cell type focuses the model on cell-state variation within a single lineage.

# %%
cell_type = "CD8 Naive"
dd_ct = dd[..., dd.C.obs[celltype_key] == cell_type, :].copy()
print(f"After filtering to '{cell_type}': {dd_ct.C.shape[0]} cells, {dd_ct.G.shape[0]} donors")

# %% [markdown]
# ## Training LIVI
# 
# ### Key Architecture Parameters
# 
# | Parameter | Meaning |
# |---|---|
# | `z_dim` | Number of cell-state latent factors (the VAE bottleneck) |
# | `n_dxc_factors` | Number of donor × cell-state interaction (DxC) factors |
# | `n_persistent_factors` | Number of persistent (cell-state-independent) donor factors |
# | `warmup_epochs_vae` | Epochs to train only the VAE before activating donor effects |
# | `warmup_epochs_G` | Additional epochs to train only `V` before activating `DxC` |
# 
# ### Training Schedule
# 
# LIVI uses a three-phase warm-up to stabilise training:
# 1. **VAE pre-training** (`warmup_epochs_vae` epochs): only the encoder and base decoder are updated — the model learns cell-state factors without interference from donor effects.
# 2. **V training** (`warmup_epochs_G` epochs): the VAE is frozen; only persistent donor effects `V` are trained.
# 3. **Full model** (remaining epochs): DxC interaction factors are activated alongside `V`.
# 
# ### Basic Training
# 
# The parameters below are scaled down for demonstration. In practice, use `z_dim=15`, `n_dxc_factors=300–700`, `max_epochs=400–600`, and `warmup_epochs_vae=60–90`.

# %%
checkpoint_path = train_livi(
    dd_ct,
    output_dir="livi_basic_run",
    # Architecture
    z_dim=5,               # cell-state latent dimensions (use 15 in practice)
    n_dxc_factors=20,      # DxC factors (use 300–700 in practice)
    n_persistent_factors=3, # persistent donor factors (use 5 in practice)
    encoder_hidden_dims=[256, 128, 64],  # encoder MLP widths
    # Data
    layer_key=None,        # None → uses dd.C.X (must be raw counts)
    use_size_factor=True,  # normalise by per-cell library size
    # Training schedule
    warmup_epochs_vae=5,
    warmup_epochs_G=0,
    max_epochs=15,
    min_epochs=5,
    # Optimiser
    learning_rate=8e-4,
    l1_weight=1e-3,
    A_weight=1e-3,
    # DataLoader
    batch_size=512,
    num_workers=0,
    seed=42,
)

print(f"Best checkpoint: {checkpoint_path}")

# %% [markdown]
# ### Training with Batch Covariates
# 
# Technical covariates such as experimental pool or sequencing batch can be corrected for within LIVI via the `covariates_keys` parameter. Each key must be a categorical column in `dd.C.obs`. LIVI learns a per-gene additive correction for each category.

# %%
# Verify that the covariate columns exist in dd.C.obs
print("Available obs columns:", dd_ct.C.obs.columns.tolist())

# 'sex' is available in the dummy dataset
covariates = ["sex"]

checkpoint_with_covariates = train_livi(
    dd_ct,
    output_dir="livi_covariate_run",
    z_dim=5,
    n_dxc_factors=20,
    n_persistent_factors=3,
    covariates_keys=covariates,  # corrects for sex
    # covariates_dims is inferred from data if not provided
    layer_key=None,
    warmup_epochs_vae=5,
    max_epochs=15,
    batch_size=512,
    seed=42,
)

print(f"Best checkpoint (with covariates): {checkpoint_with_covariates}")

# %% [markdown]
# ### Training with Known cis-eQTLs (using `dd.G` genotypes)
# 
# This mode jointly uses `dd.C` (expression) and `dd.G` (genotypes) to correct for local genetic effects during training, preventing cis-genetic variance from leaking into the donor embeddings.
# 
# Required inputs:
# 
# | Input | Shape | Source |
# |---|---|---|
# | `eqtl_genotypes` | donors × cis-SNPs | `dd.G.X` — auto-extracted when DonorData is passed **and `dd.G` has been pre-filtered to cis-SNPs** |
# | `known_cis_eqtls` | cis-SNPs × genes (binary 0/1) | External annotation (GTEx, eQTL Catalogue); only genes with known cis-eQTLs needed |
# | `n_cis_snps` | int | Number of cis-SNPs; must equal `dd.G.n_vars` when auto-extracting |
# 
# **Auto-extraction pattern (recommended for real analyses):**
# Pre-filter `dd.G` to cis-SNPs only, then pass DonorData — `eqtl_genotypes` is auto-extracted:
# ```python
# dd_cis = dd_ct[:, :, cis_snp_mask, :]          # subset dd.G to cis-SNPs
# train_livi(dd_cis, ..., n_cis_snps=dd_cis.G.n_vars, known_cis_eqtls=annotation_df)
# ```
# 
# Below we build a small synthetic `known_cis_eqtls` and extract `eqtl_genotypes` from `dd.G.X` explicitly (since we are using a SNP subset for speed). The cell runs end-to-end on the dummy dataset.

# %%
from anndata.utils import asarray

# Use a small subset of SNPs from dd.G for the demo (50 SNPs for speed)
# In a real analysis, dd.G would be pre-filtered to cis-SNPs and the
# eqtl_genotypes auto-extraction would be used instead.
n_demo_snps = 50
demo_snps = dd_ct.G.var_names[:n_demo_snps]

# eqtl_genotypes: donors × cis-SNPs, extracted from dd.G.X
eqtl_genotypes_demo = pd.DataFrame(
    asarray(dd_ct.G.X[:, :n_demo_snps]),
    index=dd_ct.G.obs_names,
    columns=demo_snps,
)

# known_cis_eqtls: cis-SNPs × genes, binary (0/1)
# Rows = SNP IDs (must match eqtl_genotypes columns)
# Columns = gene IDs with known cis-associations (LIVI fills 0 for all other genes)
# In practice: derive from GTEx, eQTL Catalogue, or similar reference
rng = np.random.default_rng(42)
target_genes = dd_ct.C.var_names[:200]   # only a subset of genes needed
known_cis_eqtls_demo = pd.DataFrame(
    0, index=demo_snps, columns=target_genes, dtype=int
)
for snp in demo_snps:                    # assign each SNP to ~5 random genes
    hits = rng.choice(target_genes, size=5, replace=False)
    known_cis_eqtls_demo.loc[snp, hits] = 1

print(f"eqtl_genotypes:   {eqtl_genotypes_demo.shape}  (donors × cis-SNPs)")
print(f"known_cis_eqtls:  {known_cis_eqtls_demo.shape}  (cis-SNPs × genes with known associations)")

checkpoint_cis = train_livi(
    dd_ct,
    output_dir="livi_cis_run",
    z_dim=5,
    n_dxc_factors=20,
    n_persistent_factors=3,
    n_cis_snps=n_demo_snps,
    known_cis_eqtls=known_cis_eqtls_demo,   # SNPs × genes binary annotation
    eqtl_genotypes=eqtl_genotypes_demo,      # donors × SNPs from dd.G.X
    layer_key=None,
    covariates_keys=["sex"],
    warmup_epochs_vae=5,
    warmup_epochs_G=0,
    max_epochs=15,
    min_epochs=5,
    batch_size=512,
    cell_state_cis=True,   # learn cell-state-specific cis corrections per SNP
    seed=42,
)

print(f"Best checkpoint (cis-eQTL mode): {checkpoint_cis}")

# %% [markdown]
# ### Dry Run: Inspect Configuration Without Training
# 
# Pass `run=False` to print the resolved model dimensions and return `None` without running any training. This is useful for verifying that `x_dim` and `y_dim` are inferred correctly before launching a long job.

# %%
train_livi(
    dd_ct,
    output_dir="livi_dryrun",
    z_dim=15,
    n_dxc_factors=300,
    n_persistent_factors=5,
    covariates_keys=["sex"],
    max_epochs=400,
    run=False,  # inspect config only
)

# %% [markdown]
# ## Inference: Extracting Latent Factors
# 
# `infer_livi` loads a trained checkpoint and runs batch-wise inference to extract all latent representations. It returns a dictionary of DataFrames:
# 
# | Key | Shape | Content |
# |---|---|---|
# | `cell_state_latent` | n_cells × z_dim | Cell-state factor scores per cell |
# | `cell_state_decoder` | n_genes × z_dim | Gene loadings of the cell-state decoder |
# | `D_embedding` | n_donors × n_DxC | Donor × cell-state interaction embeddings |
# | `DxC_decoder` | n_genes × n_DxC | Gene loadings of the DxC decoder |
# | `V_embedding` | n_donors × n_V | Persistent donor factor embeddings |
# | `V_decoder` | n_genes × n_V | Gene loadings of the V decoder |
# | `assignment_matrix` | z_dim × n_DxC | Assignment matrix *A* linking cell-state to DxC factors |
# 
# > **Important**: inference should be run on the **same AnnData** (same donors, same cell order) as used for training so that the donor embedding indices align correctly.

# %%
results = infer_livi(
    dd_ct,
    checkpoint_path=checkpoint_path,
    batch_size=50_000,   # cells per inference batch
)

print("Inference outputs:")
for key, df in results.items():
    print(f"  {key}: {df.shape}")

# %% [markdown]
# ### Cell-State Latent Factors
# 
# Each row is a cell; each column is a learned cell-state factor. These capture transcriptional programs that are shared across donors and can be used as input for downstream analyses such as UMAP or differential state testing.

# %%
cell_state = results["cell_state_latent"]
print(f"Cell-state factors: {cell_state.shape}")
cell_state.head()

# %% [markdown]
# We can add the cell-state factors directly to the AnnData for downstream analysis and UMAP visualisation.

# %%
# Add cell-state factors to adata as an obsm embedding
dd_ct.C.obsm["X_livi"] = cell_state.loc[dd_ct.C.obs_names].values

# Compute neighbours and UMAP from LIVI factors
sc.pp.neighbors(dd_ct.C, use_rep="X_livi", n_neighbors=15)
sc.tl.umap(dd_ct.C)
sc.pl.umap(dd_ct.C, color=[celltype_key, "donor_id"], frameon=False)

# %% [markdown]
# ### Donor Embeddings
# 
# The **D embedding** captures how each donor's gene expression is modulated by their cell state. Each row is a donor; each column is a learned DxC interaction factor.

# %%
D_embedding = results["D_embedding"]
print(f"D embedding (donor × DxC): {D_embedding.shape}")
D_embedding.head()

# %% [markdown]
# The `variance_threshold` argument retains only D factors with sufficient variability across donors, reducing noise in downstream association testing.

# %%
results_filtered = infer_livi(
    dd_ct,
    checkpoint_path=checkpoint_path,
    variance_threshold=0.01,  # keep only variable D factors
)

print("All D factors:     ", results["D_embedding"].shape)
print("Variable D factors:", results_filtered["D_embedding"].shape)

# %% [markdown]
# ## Association Testing: Linking Donor Embeddings to Genotype
# 
# LIVI's donor embeddings can be treated as quantitative traits and tested for association with genetic variants. `run_livi_association_testing` tests each D and V factor against all SNPs using either:
# 
# - `"LMM"` (LIMIX): a linear mixed model that accounts for sample relatedness via a kinship matrix — recommended for cohorts with population structure.
# - `"TensorQTL"`: fast GPU-accelerated testing without relatedness correction.
# 
# **When a `DonorData` object is passed as `genotype_matrix`, the wrapper automatically extracts:**
# - `dd.G.X` → donors × SNPs genotype matrix
# - `dd.G.uns["kinship"]` → kinship / GRM matrix (used for `method="LMM"`)
# - `dd.G.obsm["gPCs"]` → genotype PCs (used as covariates)
# 
# You can always override any of these by passing `kinship=` or `genotype_pcs=` explicitly.

# %%
# Pass DonorData directly — dd.G.X, dd.G.uns["kinship"], and dd.G.obsm["gPCs"]
# are all extracted automatically.
associations = run_livi_association_testing(
    inference_results=results,
    genotype_matrix=dd,          # DonorData: dd.G is used for genotypes + kinship + gPCs
    output_dir="livi_associations",
    method="LMM",
    fdr_threshold=0.05,
    fdr_method="Benjamini-Hochberg",
    quantile_norm=True,
    output_file_prefix="demo",
)

# Returns a tuple (DxC_results, V_results) when both D and V embeddings are present
DxC_assoc, V_assoc = associations if isinstance(associations, tuple) else (associations, None)
print(f"DxC associations: {DxC_assoc.shape if DxC_assoc is not None else 'None'}")
print(f"V associations:   {V_assoc.shape if V_assoc is not None else 'None'}")

# %%
if DxC_assoc is not None:
    DxC_assoc.head()

# %% [markdown]
# ### Explicit Component Override
# 
# When you need more control — e.g. a subset of SNPs or a different kinship — you can pass components individually instead of using `DonorData`. All three (genotype matrix, kinship, genotype PCs) can be overridden independently while still passing `dd` for the rest.

# %%
# Example: override genotype PCs but still pass dd for genotype matrix and kinship
# gPCs = dd.G.obsm["gPCs"]   # already extracted automatically when passing dd,
#                              # but you could subset or replace it here

# --- TensorQTL: pass components explicitly (no kinship needed) ---
#
# from anndata.utils import asarray
# GT_matrix = pd.DataFrame(
#     asarray(dd.G.X), index=dd.G.obs_names, columns=dd.G.var_names
# )
# gPCs = dd.G.obsm["gPCs"]
#
# associations_tensorqtl = run_livi_association_testing(
#     inference_results=results,
#     genotype_matrix=GT_matrix,   # explicit DataFrame
#     output_dir="livi_associations_tensorqtl",
#     method="TensorQTL",
#     genotype_pcs=gPCs,
#     fdr_threshold=0.05,
#     quantile_norm=True,
# )

# %% [markdown]
# ## Saving and Loading Results
# 
# `save_livi_results` writes all inference DataFrames to TSV files in a single directory. `load_livi_results` reads them back, allowing you to restart downstream analyses without re-running inference.

# %%
# Save
saved_paths = save_livi_results(
    results,
    output_dir="livi_results",
    prefix="cd8_naive",
)
print("Saved files:")
for key, path in saved_paths.items():
    print(f"  {key}: {path}")

# %%
# Load back
results_loaded = load_livi_results("livi_results", prefix="cd8_naive")
print("Loaded DataFrames:", list(results_loaded.keys()))

# %%



