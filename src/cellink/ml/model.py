import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class DonorMILModel(pl.LightningModule):
    """A PyTorch Lightning model for multiple instance learning (MIL) using donor-level and cell-level features.

    This model encodes donor-level features and aggregates
    variable-length cell-level features using an attention mechanism before concatenating
    them for final regression or classification.

    Parameters
    ----------
    n_input_donor : int
        Number of input features for the donor-level encoder.
    n_input_cell : int
        Number of input features for the cell-level encoder.
    n_hidden : int, default=128
        Dimension of the hidden representation for donor and cell encoders.
    n_output : int, default=1
        Number of output units: 1 for regression (mean squared error loss), more than 1 for
        multi-class classification with ``n_output`` classes (cross-entropy loss on integer
        class labels ``0 .. n_output - 1`` in ``donor_y``).
    lr : float, default=1e-3
        Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        n_input_donor: int = None,
        n_input_cell: int = None,
        n_hidden: int = 128,
        n_output: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.donor_encoder = nn.Linear(n_input_donor, n_hidden)
        self.cell_encoder = nn.Sequential(nn.Linear(n_input_cell, n_hidden), nn.ReLU(), nn.Dropout(0.2))

        self.attention = nn.Sequential(nn.Linear(n_hidden, 64), nn.Tanh(), nn.Linear(64, 1))

        self.classifier = nn.Sequential(nn.Linear(n_hidden * 2, n_output))

    def forward(self, batch):
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing input features:

            - ``"donor_x"``: tensor of shape ``(B, D_donor)`` with donor-level features.
            - ``"cell_x"``: list of tensors ``[(N_i, D_cell), ...]`` with cell-level features for
              each donor in the batch, where ``N_i`` is the number of cells of donor ``i``.
            - ``"cell_mask"`` (optional): per donor, a boolean tensor of length ``N_i`` that is
              True for real cells and False for padding rows; padding rows are left out of the
              attention pooling.

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, n_output), representing the predicted target
            for each donor.
        """
        donor_x = batch["donor_x"].float()  # shape: (B, D_donor)
        cell_x_list = batch["cell_x"]  # list of (N_i, D_cell)

        h_donor = self.donor_encoder(donor_x)  # shape: (B, H)

        cell_mask = batch.get("cell_mask")  # present when bags are padded to a fixed size
        h_cell_list = []
        for i in range(len(cell_x_list)):
            cells = cell_x_list[i] if cell_mask is None else cell_x_list[i][cell_mask[i].bool()]
            h_cells = self.cell_encoder(cells)  # shape: (N_i, H), real cells only
            attn_scores = self.attention(h_cells)  # shape: (N_i, 1)
            attn_weights = F.softmax(attn_scores.squeeze(-1), dim=0)  # shape: (N_i,)
            h_cell = torch.sum(attn_weights.unsqueeze(-1) * h_cells, dim=0)  # shape: (H,)
            h_cell_list.append(h_cell)

        h_cell = torch.stack(h_cell_list)  # shape: (B, H)

        combined = torch.cat([h_donor, h_cell], dim=-1)  # shape: (B, H*2)
        logits = self.classifier(combined)  # shape: (B, n_output)
        return logits

    def _loss(self, batch):
        """Mean squared error for regression (``n_output == 1``), cross-entropy for classification."""
        logits = self(batch)  # shape: (B, n_output)
        y = batch["donor_y"].reshape(logits.shape[0], -1)  # one target per donor
        if self.hparams.n_output > 1:
            return F.cross_entropy(logits, y[:, 0].long())
        return F.mse_loss(logits.squeeze(-1), y[:, 0].float())

    def training_step(self, batch, batch_idx):
        """
        Training step used by PyTorch Lightning.

        Parameters
        ----------
        batch : dict
            Dictionary containing input features and target labels.
        batch_idx : int
            Index of the batch (unused).

        Returns
        -------
        torch.Tensor
            The loss for the batch (MSE for regression, cross-entropy for classification).
        """
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step used by PyTorch Lightning.

        Parameters
        ----------
        batch : dict
            Dictionary containing input features and target labels.
        batch_idx : int
            Index of the batch (unused).

        Returns
        -------
        None
            Logs validation loss to PyTorch Lightning without returning a value.
        """
        loss = self._loss(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer with learning rate specified in `self.hparams.lr`.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
