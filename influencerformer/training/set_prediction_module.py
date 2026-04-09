"""Lightning module for set prediction loss comparison.

Generic training module that takes any model outputting a set of elements
and any loss function operating on a pairwise distance matrix.
"""

import inspect
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class SetPredictionModule(pl.LightningModule):
    """Generic Lightning module for training set prediction with any loss.

    The model outputs a set of elements (B, M, D).
    The target is a set of elements (B, N, D).
    The loss operates on the pairwise distance matrix D (B, M, N).

    Evaluation always uses Hungarian matching as ground truth metric,
    regardless of training loss.

    Logs cumulative timing for model forward, distance matrix, loss,
    and backward passes separately.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        distance_fn: str = "l2",
        match_threshold: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.distance_fn = distance_fn
        self.match_threshold = match_threshold
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Auto-detect if model accepts mask kwarg
        sig = inspect.signature(self.model.forward)
        self._model_accepts_mask = "mask" in sig.parameters

        # Timing accumulators
        self._t_model = 0.0
        self._t_dist = 0.0
        self._t_loss = 0.0
        self._t_backward = 0.0
        self._t_total = 0.0
        self._n_steps = 0

    def forward(self, x, mask=None):
        if mask is not None and self._model_accepts_mask:
            return self.model(x, mask=mask)
        return self.model(x)

    def _compute_distance_matrix(self, preds, targets):
        if self.distance_fn == "l2":
            return torch.cdist(preds, targets)
        elif self.distance_fn == "smooth_l1":
            # Pairwise Smooth L1 (Huber) distance, matching DSPN convention
            diff = preds.unsqueeze(2) - targets.unsqueeze(1)  # (B, M, N, D)
            return F.smooth_l1_loss(
                diff, torch.zeros_like(diff), reduction="none",
            ).mean(dim=-1)  # (B, M, N)
        else:
            raise ValueError(f"Unknown distance function: {self.distance_fn}")

    def _unpack_batch(self, batch):
        """Unpack batch, handling optional mask (3rd element)."""
        if len(batch) == 3:
            inputs, targets, mask = batch
        else:
            inputs, targets = batch
            mask = None
        return inputs, targets, mask

    def training_step(self, batch, batch_idx):
        t0 = time.perf_counter()

        inputs, targets, mask = self._unpack_batch(batch)

        t1 = time.perf_counter()
        if mask is not None and self._model_accepts_mask:
            preds = self.model(inputs, mask=mask)
        else:
            preds = self.model(inputs)
        t2 = time.perf_counter()

        D = self._compute_distance_matrix(preds, targets)
        t3 = time.perf_counter()

        loss = self.loss_fn(D, mask=mask) if mask is not None else self.loss_fn(D)
        t4 = time.perf_counter()

        if not torch.isfinite(loss):
            return None

        self.log("train_loss", loss, prog_bar=True)

        # Timing (backward is measured via on_before/after_backward hooks)
        self._t_model += t2 - t1
        self._t_dist += t3 - t2
        self._t_loss += t4 - t3
        self._t_total += t4 - t0
        self._n_steps += 1

        return loss

    def on_before_backward(self, loss):
        self._t_bwd_start = time.perf_counter()

    def on_after_backward(self):
        self._t_backward += time.perf_counter() - self._t_bwd_start

    def on_train_epoch_end(self):
        n = max(self._n_steps, 1)
        self.log("time/model_ms", 1000 * self._t_model / n)
        self.log("time/dist_ms", 1000 * self._t_dist / n)
        self.log("time/loss_ms", 1000 * self._t_loss / n)
        self.log("time/backward_ms", 1000 * self._t_backward / n)
        self.log("time/total_ms", 1000 * self._t_total / n)
        self.log("time/steps", float(self._n_steps))

    def get_timing_summary(self):
        """Return timing summary as a dict (call after training)."""
        n = max(self._n_steps, 1)
        return {
            "model_ms": 1000 * self._t_model / n,
            "dist_ms": 1000 * self._t_dist / n,
            "loss_ms": 1000 * self._t_loss / n,
            "backward_ms": 1000 * self._t_backward / n,
            "total_ms": 1000 * self._t_total / n,
            "n_steps": self._n_steps,
            "total_train_s": self._t_total + self._t_backward,
        }

    def validation_step(self, batch, batch_idx):
        inputs, targets, mask = self._unpack_batch(batch)
        if mask is not None and self._model_accepts_mask:
            preds = self.model(inputs, mask=mask)
        else:
            preds = self.model(inputs)
        D = self._compute_distance_matrix(preds, targets)

        train_loss = self.loss_fn(D, mask=mask) if mask is not None else self.loss_fn(D)
        self.log("val_loss", train_loss)

        # Metrics use submatrix for correct Hungarian matching
        B, M, N = D.shape
        D_np = D.detach().cpu().numpy()
        matched_dists = []
        n_correct = 0
        n_total = 0
        n_dup = 0
        total_real = 0

        for b in range(B):
            n_real = int(mask[b].sum().item()) if mask is not None else N
            total_real += n_real

            # Hungarian on real entries only
            D_sub = D_np[b, :n_real, :n_real]
            row, col = linear_sum_assignment(D_sub)
            matched = D_sub[row, col]
            matched_dists.append(matched.mean())
            n_correct += (matched < self.match_threshold).sum()
            n_total += len(row)

            # Duplicate detection over real targets only
            for j in range(n_real):
                if (D_np[b, :n_real, j] < self.match_threshold).sum() > 1:
                    n_dup += 1

        self.log("val_matched_dist", np.mean(matched_dists), prog_bar=True)
        self.log("val_match_acc", n_correct / max(n_total, 1))
        self.log("val_dup_rate", n_dup / max(total_real, 1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
