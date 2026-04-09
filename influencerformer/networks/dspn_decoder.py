"""DSPN: Deep Set Prediction Networks decoder.

Gradient-descent based decoder that iteratively refines a learned starting
set by backpropagating through the encoder.

Reference: Zhang et al., "Deep Set Prediction Networks", NeurIPS 2019.
Adapted from github.com/Cyanogenoid/dspn (MIT license).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSPNDecoder(nn.Module):
    """DSPN gradient-descent decoder.

    Starting from a learned initial set, iteratively refines predictions
    by computing gradients of the representation loss w.r.t. the set
    elements and mask.

    Args:
        encoder: Set encoder mapping (set, mask) → representation.
                 Must be the SAME encoder used for target encoding.
        obj_dim: Dimension of each object (default 18 for CLEVR state).
        max_objects: Maximum set size.
        inner_lr: Learning rate for inner gradient descent (default 800).
        inner_steps: Number of inner optimization steps (default 10).
    """

    def __init__(self, encoder, obj_dim=18, max_objects=10,
                 inner_lr=800.0, inner_steps=10):
        super().__init__()
        self.encoder = encoder
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.starting_set = nn.Parameter(torch.rand(1, max_objects, obj_dim))
        self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_objects))

    def forward(self, target_repr):
        """
        Args:
            target_repr: (B, latent_dim) target set representation from encoder.
        Returns:
            final_set: (B, max_objects, obj_dim)
            final_mask: (B, max_objects) in [0, 1]
            intermediates: list of (set, mask) at each step
        """
        B = target_repr.shape[0]
        current_set = self.starting_set.expand(B, -1, -1)
        current_mask = self.starting_mask.expand(B, -1).clamp(0, 1)

        intermediates = []

        for _ in range(self.inner_steps):
            # Enable gradients for current set/mask
            if not self.training:
                current_set = current_set.detach().requires_grad_(True)
                current_mask = current_mask.detach().requires_grad_(True)

            with torch.enable_grad():
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    predicted_repr = self.encoder(current_set, current_mask)
                repr_loss = F.smooth_l1_loss(
                    predicted_repr, target_repr, reduction="mean",
                )
                set_grad, mask_grad = torch.autograd.grad(
                    outputs=repr_loss,
                    inputs=[current_set, current_mask],
                    create_graph=self.training,
                    allow_unused=True,
                )
                # If mask gradient didn't flow, use zeros
                if mask_grad is None:
                    mask_grad = torch.zeros_like(current_mask)

            current_set = current_set - self.inner_lr * set_grad
            current_mask = (current_mask - self.inner_lr * mask_grad).clamp(0, 1)

            if not self.training:
                current_set = current_set.detach()
                current_mask = current_mask.detach()

            intermediates.append((current_set, current_mask))

        return current_set, current_mask, intermediates


class DSPNModel(nn.Module):
    """Full DSPN model: encoder + gradient-descent decoder.

    Wraps DSPNDecoder to match the SetAutoencoder interface:
    forward(x, mask) → (B, M, D).
    """

    def __init__(self, encoder, obj_dim=18, max_objects=10,
                 inner_lr=800.0, inner_steps=10):
        super().__init__()
        self.encoder = encoder
        self.dspn = DSPNDecoder(
            encoder=encoder, obj_dim=obj_dim, max_objects=max_objects,
            inner_lr=inner_lr, inner_steps=inner_steps,
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) target set (used to compute target representation)
            mask: (B, N) binary mask
        Returns:
            (B, max_objects, obj_dim) predicted set
        """
        target_repr = self.encoder(x, mask)
        final_set, _, _ = self.dspn(target_repr)
        return final_set
