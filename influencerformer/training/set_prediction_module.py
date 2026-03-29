"""Lightning module for set prediction loss comparison.

Generic training module that takes any model outputting a set of elements
and any loss function operating on a pairwise distance matrix.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class SetPredictionModule(pl.LightningModule):
    """Generic Lightning module for training set prediction with any loss.

    The model outputs a set of elements (B, M, D).
    The target is a set of elements (B, N, D).
    The loss operates on the pairwise distance matrix D (B, M, N).

    Evaluation always uses Hungarian matching as ground truth metric,
    regardless of training loss.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        distance_fn: str = "l2",  # "l2" or "ce"
        match_threshold: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.distance_fn = distance_fn
        self.match_threshold = match_threshold
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x):
        return self.model(x)

    def _compute_distance_matrix(self, preds, targets):
        if self.distance_fn == "l2":
            return torch.cdist(preds, targets)
        elif self.distance_fn == "ce":
            raise NotImplementedError("CE distance for token prediction")
        else:
            raise ValueError(f"Unknown distance function: {self.distance_fn}")

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.model(inputs)
        D = self._compute_distance_matrix(preds, targets)
        loss = self.loss_fn(D)

        if not torch.isfinite(loss):
            return None

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.model(inputs)
        D = self._compute_distance_matrix(preds, targets)

        # Compute training loss for comparison
        train_loss = self.loss_fn(D)
        self.log("val_loss", train_loss)

        # Hungarian matching evaluation (ground truth metric)
        B, M, N = D.shape
        D_np = D.detach().cpu().numpy()
        matched_dists = []
        n_correct = 0
        n_total = 0
        n_dup = 0

        for b in range(B):
            row, col = linear_sum_assignment(D_np[b])
            matched = D_np[b][row, col]
            matched_dists.append(matched.mean())
            n_correct += (matched < self.match_threshold).sum()
            n_total += len(row)
            for j in range(N):
                if (D_np[b, :, j] < self.match_threshold).sum() > 1:
                    n_dup += 1

        self.log("val_matched_dist", np.mean(matched_dists), prog_bar=True)
        self.log("val_match_acc", n_correct / max(n_total, 1))
        self.log("val_dup_rate", n_dup / max(B * N, 1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
