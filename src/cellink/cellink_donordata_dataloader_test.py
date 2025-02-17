import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import sys
import os

if __name__ == "__main__":

    from cellink.io import read_sgkit_zarr
    from cellink import DonorData, DonorDataBaseModel
    from anndata import read_h5ad
    from pathlib import Path

    DATA = Path("../tests/data")

    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")
    dd.aggregate("X", "Gex")

    dd.gdata.obs["patient_label"] = np.random.randint(0, 2, 997)
    dd.mod = {"adata": dd.adata, "gdata": dd.gdata}
    dd.uns = {}
    dd.isbacked = True
    dd.n_obs = dd.gdata.n_obs
    dd.gdata.obs["donor_patient"] = dd.gdata.obs.index
    DonorDataBaseModel.setup_anndata(dd, cell_batch_key="age", cell_patient_key="individual", donor_labels_key="patient_label", cell_labels_key="cell_label", donor_patient_key="donor_patient")

    model = DonorDataBaseModel(
        dd,
        n_input_snps=len(dd.gdata.var),
        n_input_genes=len(dd.adata.var),
        n_hidden=10,
    )

    model.train(max_epochs=1, shuffle_set_split=False)

    ############################

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools/src/")
    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/serendipity/")
    import scvi
    from benchmarking import datasets

    paired_rate = 1.0
    utilize_highly_variable = True
    os.chdir("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/serendipity/")
    mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, n_patient_covariates, protein_expression_obsm_key = datasets.load_neurips2021_cite_BMMC(
        paired_rate, utilize_highly_variable)
    os.chdir('/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/sc-genetics/src')

    adata_mvi = sc.concat([mdata_counts.mod[mod] for mod in mdata_counts.mod if mod != "prot"], join="outer",
                          merge="first", axis=1)
    adata_mvi.X = np.array(adata_mvi.X)
    if "prot" in mdata_counts.mod:
        adata_mvi.obsm["protein_expression"] = mdata_counts.mod[
            "prot"].X  # pd.DataFrame(mdata_counts.mod["prot"].X, index=mdata_counts.mod["prot"].obs.index, columns=mdata_counts.mod["prot"].var["gene_stable_id"])
        adata_mvi.uns['protein_expression'] = {
            'var': mdata_counts.mod["prot"].var,
            'obs': mdata_counts.mod["prot"].obs
        }

    inject_batch_covariates = True
    inject_patient_covariates = True
    categorical_covariate_keys = ([batch_key] if inject_batch_covariates else []) + ([patient_key] if inject_patient_covariates else []) if inject_batch_covariates or inject_patient_covariates else None

    from cellink import DonorDataBaseModel

    DonorDataBaseModel.setup_anndata(adata_mvi, batch_key="modality",
                                      categorical_covariate_keys=categorical_covariate_keys,
                                      patient_key=patient_key, labels_key=labels_key,
                                      protein_expression_obsm_key=protein_expression_obsm_key)

    model = DonorDataBaseModel(
        adata_mvi,
    )


    ### #max_epochs=1,
    model.train(shuffle_set_split=False,
                adversarial_mixing=True if (paired_rate != 1.0 and paired_rate != 1) else False) #train_idx=train_indices, validation_indices=val_indices
