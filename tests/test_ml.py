from pathlib import Path

import pytest
import os
from sgkit.io.plink import read_plink as sg_read_plink

from cellink._core.donordata import DonorData
from cellink.io import read_sgkit_zarr
from cellink import DonorData, DonorDataBaseModel
from anndata import read_h5ad
import numpy as np
from cellink._core.data_fields import CAnn, DAnn

DATA = Path("tests/data") #TODO ARNOLDT

@pytest.mark.slow
def test_dataloader(adata, gdata):

    dd = DonorData(G=gdata, C=adata).copy()

    DonorDataBaseModel.setup_anndata(dd, cell_batch_key=CAnn.CELL_BATCH_KEY, cell_donor_key=CAnn.CELL_DONOR_KEY, donor_labels_key=DAnn.DONOR_LABELS_KEY, cell_labels_key=CAnn.CELL_LABELS_KEY, donor_id_key=DAnn.DONOR_ID_KEY)

    model = DonorDataBaseModel(
        dd,
        n_input_snps=len(dd.G.var),
        n_input_genes=len(dd.C.var),
        n_hidden=10,
    )

    model.train(max_epochs=1, shuffle_set_split=False)

    """
    #gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    #adata = read_h5ad(DATA / "simulated_gene_expression.h5ad")
    dd = DonorData(adata, gdata, "individual")
    dd.aggregate("X", "Gex")

    dd.gdata.obs["patient_label"] = np.random.randint(0, 2, 997)
    dd.mod = {"adata": dd.adata, "gdata": dd.gdata}
    dd.uns = {}
    dd.isbacked = True
    dd.n_obs = dd.gdata.n_obs
    dd.gdata.obs["donor_patient"] = dd.gdata.obs.index
    DonorDataBaseModel.setup_anndata(dd, cell_batch_key="age", cell_donor_key="individual", donor_labels_key="patient_label", cell_labels_key="cell_label", donor_id_key="donor_patient")

    model = DonorDataBaseModel(
        dd,
        n_input_snps=len(dd.gdata.var),
        n_input_genes=len(dd.adata.var),
        n_hidden=10,
    )

    model.train(max_epochs=1, shuffle_set_split=False)
    """