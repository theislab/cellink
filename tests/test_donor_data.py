from pathlib import Path

import pytest
from anndata import read_h5ad

from cellink._core.donordata import DonorData
from cellink.io import read_sgkit_zarr

DATA = Path("tests/data")


@pytest.mark.slow
def test_donordata_init():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    # gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")
    print(dd)


@pytest.mark.slow
def test_donordata_aggregate():
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    # gdata.obs = gdata.obs.set_index("id")
    adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")
    dd.aggregate("X", "Gex")
    assert "Gex" in dd.gdata.obsm

    previous_adata_shape = dd.adata.shape
    dd.aggregate("X", "Gex_CD4NC", filter_key="cell_label", filter_value="CD4 NC")
    assert "Gex_CD4NC" in dd.gdata.obsm
    assert dd.adata.shape == previous_adata_shape

    dd.aggregate("age", "age")
    assert "age" in dd.gdata.obs

@pytest.mark.slow
def test_donordata_aggregate():

    from cellink.io import read_sgkit_zarr
    from cellink import DonorData, DonorDataBaseModel
    from anndata import read_h5ad
    from pathlib import Path
    import numpy as np

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

