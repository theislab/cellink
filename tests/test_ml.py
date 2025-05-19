import os
import shutil
from pathlib import Path

import pytest
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from cellink import DonorData
from cellink.ml.dataset import MILDataset, mil_collate_fn 
from cellink.ml.model import DonorMILModel


DATA = Path("tests/data")

@pytest.mark.slow
def test_dataloader(adata, gdata):

    dd = DonorData(G=gdata, C=adata)

    dd.G.obs["donor_id"] = dd.G.obs.index
    dd.G.obs["donor_labels"] = np.random.randint(2, size=len(dd.G.obs))
    split_indices = [0, 1, 2]
    split_donors = None
    batch_size = 2
    shuffle = True
    
    dataset = MILDataset(dd, cell_batch_key="rna:cov1", split_donors=split_donors, split_indices=split_indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mil_collate_fn, 
    )

    model = DonorMILModel(n_input_donor=dd.G.n_vars, n_input_cell=dd.C.n_vars)

    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(model, dataloader)