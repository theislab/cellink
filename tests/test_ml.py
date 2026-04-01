from pathlib import Path

import pytest

# skip the entire module if torch or lightning aren't installed
pytest.importorskip("torch", minversion="1.10", reason="need torch for ML tests")
pytest.importorskip("pytorch_lightning", minversion="1.7", reason="need pytorch_lightning for ML tests")


import numpy as np

from cellink import DonorData
from cellink.ml.dataset import MILDataset, _get_contiguous_bag_slices, mil_collate_fn
from cellink.ml.model import DonorMILModel

DATA = Path("tests/data")


@pytest.mark.slow
def test_dataloader(adata, gdata):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    dd = DonorData(G=gdata, C=adata)

    dd.G.obs["donor_id"] = dd.G.obs.index
    dd.G.obs["donor_labels"] = np.random.randint(2, size=len(dd.G.obs))
    split_indices = [0, 1, 2]
    split_donors = None
    batch_size = 2
    shuffle = True

    dataset = MILDataset(
        dd,
        cell_batch_key="rna:cov1",
        split_donors=split_donors,
        split_indices=split_indices,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mil_collate_fn,
    )

    model = DonorMILModel(n_input_donor=dd.G.n_vars, n_input_cell=dd.C.n_vars)

    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(model, dataloader)


def test_cell_only_mil_forward(adata, gdata):
    from torch.utils.data import DataLoader

    dd = DonorData(G=gdata, C=adata)

    dd.G.obs["donor_id"] = dd.G.obs.index
    dd.G.obs["donor_labels"] = np.random.randint(2, size=len(dd.G.obs))

    dataset = MILDataset(
        dd,
        include_donor_x=False,
        split_indices=[0, 1],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=mil_collate_fn,
    )

    batch = next(iter(dataloader))
    assert "donor_x" not in batch
    assert len(batch["cell_x"]) == 2

    model = DonorMILModel(n_input_donor=None, n_input_cell=dd.C.n_vars)
    logits = model(batch)
    assert logits.shape == (2, 1)


def test_get_contiguous_bag_slices_filters_selected_donors():
    ids = np.array(["d1", "d1", "d2", "d2", "d2", "d3"])
    bag_slices = _get_contiguous_bag_slices(ids, np.array(["d1", "d3"]))

    assert bag_slices == [
        ("d1", slice(0, 2)),
        ("d3", slice(5, 6)),
    ]
