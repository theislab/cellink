
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DonorMILModel(pl.LightningModule):
    """
    A PyTorch Lightning model for MIL using donor-level and cell-level features.
    """

    def __init__(self, n_input_donor, n_input_cell, n_hidden=128, n_output=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.donor_encoder = nn.Linear(n_input_donor, n_hidden)
        self.cell_encoder = nn.Sequential(
            nn.Linear(n_input_cell, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden * 2, n_output)
        )

    def forward(self, batch):
        donor_x = batch["donor_x"].float()                 # shape: (B, D_donor)
        cell_x_list = batch["cell_x"]                      # list of (N_i, D_cell)

        h_donor = self.donor_encoder(donor_x)              # shape: (B, H)

        h_cell_list = []
        for i in range(len(cell_x_list)):
            h_cells = self.cell_encoder(cell_x_list[i])    # shape: (N_i, H)
            attn_scores = self.attention(h_cells)          # shape: (N_i, 1)
            attn_weights = F.softmax(attn_scores.squeeze(-1), dim=0)  # shape: (N_i,)
            h_cell = torch.sum(attn_weights.unsqueeze(-1) * h_cells, dim=0)  # shape: (H,)
            h_cell_list.append(h_cell)

        h_cell = torch.stack(h_cell_list)                  # shape: (B, H)

        combined = torch.cat([h_donor, h_cell], dim=-1)    # shape: (B, H*2)
        logits = self.classifier(combined)                 # shape: (B, n_output)
        return logits

    def training_step(self, batch, batch_idx):
        y = batch["donor_y"].float()
        y_hat = self(batch).squeeze(-1)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["donor_y"].float()
        y_hat = self(batch).squeeze(-1)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)