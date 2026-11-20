from pathlib import Path

import pytest

# skip the entire module if torch or lightning aren't installed
pytest.importorskip("torch", minversion="1.10", reason="need torch for ML tests")
pytest.importorskip("pytorch_lightning", minversion="1.7", reason="need pytorch_lightning for ML tests")


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cellink import DonorData
from cellink.ml import DonorMILModel, MILDataset, mil_collate_fn

DATA = Path("tests/data")


@pytest.mark.slow
@pytest.mark.parametrize("n_cells", [None, 8])
def test_dataloader(adata, gdata, n_cells):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    dd = DonorData(D=gdata, C=adata)

    dd.D.obs["donor_labels"] = np.random.randint(2, size=len(dd.D.obs))
    split_indices = [0, 1, 2]
    split_donors = None
    batch_size = 2
    shuffle = True

    dataset = MILDataset(
        dd,
        cell_batch_key="cov1",
        split_donors=split_donors,
        split_indices=split_indices,
        n_cells=n_cells,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mil_collate_fn,
    )

    model = DonorMILModel(n_input_donor=dd.D.n_vars, n_input_cell=dd.C.n_vars)

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu")  # MPS aborts torch on macOS 14

    trainer.fit(model, dataloader)


def _batch(n_donors=2, n_cells=4, n_donor_x=3, n_cell_x=5, y=None):
    gen = torch.Generator().manual_seed(0)
    return {
        "donor_x": torch.randn(n_donors, n_donor_x, generator=gen),
        "cell_x": [torch.randn(n_cells, n_cell_x, generator=gen) for _ in range(n_donors)],
        "donor_y": torch.zeros(n_donors, 1) if y is None else y,
    }


def test_forward_ignores_padded_cells():
    torch.manual_seed(0)
    model = DonorMILModel(n_input_donor=3, n_input_cell=5).eval()
    batch = _batch(n_donors=1, n_cells=3)
    padded = dict(batch)
    padded["cell_x"] = [torch.cat([batch["cell_x"][0], torch.zeros(2, 5)])]
    padded["cell_mask"] = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(padded))


def test_classification_uses_cross_entropy():
    torch.manual_seed(0)
    model = DonorMILModel(n_input_donor=3, n_input_cell=5, n_output=3).eval()
    batch = _batch(y=torch.tensor([[0.0], [2.0]]))  # (B, 1), as MILDataset + mil_collate_fn stack it
    with torch.no_grad():
        expected = F.cross_entropy(model(batch), torch.tensor([0, 2]))
        torch.testing.assert_close(model._loss(batch), expected)


def test_regression_loss_does_not_broadcast():
    torch.manual_seed(0)
    model = DonorMILModel(n_input_donor=3, n_input_cell=5).eval()
    y = torch.tensor([[1.0], [-1.0]])
    batch = _batch(y=y)
    with torch.no_grad():
        expected = F.mse_loss(model(batch).squeeze(-1), y.squeeze(-1))
        torch.testing.assert_close(model._loss(batch), expected)


@pytest.mark.parametrize("n_output", [1, 3])
def test_training_runs_for_regression_and_classification(n_output):
    gen = torch.Generator().manual_seed(0)
    items = [
        {
            "donor_id": f"D{i}",
            "donor_x": torch.randn(3, generator=gen),
            "donor_y": torch.tensor([float(i % max(n_output, 2))]),
            "cell_x": torch.randn(4 + i, 5, generator=gen),
        }
        for i in range(6)
    ]
    loader = DataLoader(items, batch_size=2, collate_fn=mil_collate_fn)
    model = DonorMILModel(n_input_donor=3, n_input_cell=5, n_output=n_output)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
