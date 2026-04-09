"""Set-equivariant encoders for set prediction.

Both encoders map (B, N, D) sets with optional masks to (B, latent_dim)
representations, and are permutation-invariant by construction.
"""

import torch
import torch.nn as nn


class DeepSetsEncoder(nn.Module):
    """DeepSets encoder: per-element MLP → masked mean pool → post-pool MLP.

    Reference: Zaheer et al., "Deep Sets", NeurIPS 2017.
    """

    def __init__(self, input_dim, hidden_dim=512, latent_dim=512):
        super().__init__()
        self.element_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.post_pool = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) set elements
            mask: (B, N) binary mask, 1=real, 0=padding. None means all real.
        Returns:
            (B, latent_dim) set representation
        """
        h = self.element_mlp(x)  # (B, N, hidden)

        if mask is not None:
            h = h * mask.unsqueeze(-1)  # zero out padding
            n_real = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled = h.sum(dim=1) / n_real  # (B, hidden)
        else:
            pooled = h.mean(dim=1)

        return self.post_pool(pooled)


class SetTransformerEncoder(nn.Module):
    """Set Transformer encoder: self-attention + PMA pooling.

    Reference: Lee et al., "Set Transformer", ICML 2019.
    Uses standard PyTorch TransformerEncoder + Pooling by Multihead Attention.
    """

    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=2,
                 latent_dim=512, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # PMA: Pooling by Multihead Attention (1 seed vector)
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.pma_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pma_norm = nn.LayerNorm(d_model)

        self.project = nn.Linear(d_model, latent_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) set elements
            mask: (B, N) binary mask, 1=real, 0=padding. None means all real.
        Returns:
            (B, latent_dim) set representation
        """
        B = x.shape[0]
        x = self.embed(x)  # (B, N, d_model)

        # PyTorch expects key_padding_mask where True = ignore
        key_padding_mask = (mask == 0) if mask is not None else None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # PMA: seed attends to transformer output
        seed = self.seed.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pma_attn(
            seed, x, x, key_padding_mask=key_padding_mask,
        )  # (B, 1, d_model)
        pooled = self.pma_norm(pooled + seed)  # residual + norm

        return self.project(pooled.squeeze(1))  # (B, latent_dim)
