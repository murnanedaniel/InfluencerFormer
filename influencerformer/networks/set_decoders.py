"""Set decoders for set prediction.

Decoders map (B, latent_dim) representations to (B, max_objects, obj_dim) sets.
"""

import torch
import torch.nn as nn


class MLPSetDecoder(nn.Module):
    """Standard MLP decoder: latent → hidden → set."""

    def __init__(self, latent_dim=512, hidden_dim=512, max_objects=10, obj_dim=18):
        super().__init__()
        self.max_objects = max_objects
        self.obj_dim = obj_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_objects * obj_dim),
        )

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            (B, max_objects, obj_dim)
        """
        return self.mlp(z).reshape(-1, self.max_objects, self.obj_dim)


class SlottedMLPSetDecoder(nn.Module):
    """MLP decoder with per-slot learnable embeddings.

    Each output slot gets a unique learned offset added to the shared latent
    vector z before the MLP. This guarantees prediction diversity from
    initialisation and gives each slot a distinct identity.

    Architecture:
        slot_embeddings: (max_objects, latent_dim) — learnable per-slot offsets
        z_slotted[b, i] = z[b] + slot_embeddings[i]   (B, M, latent_dim)
        output = mlp(z_slotted)                         (B, M, obj_dim)

    The MLP maps latent_dim → obj_dim (applied independently per slot),
    unlike MLPSetDecoder which maps latent_dim → max_objects * obj_dim in one shot.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 512,
                 max_objects: int = 10, obj_dim: int = 18):
        super().__init__()
        self.max_objects = max_objects
        self.obj_dim = obj_dim
        self.slot_embeddings = nn.Embedding(max_objects, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) scene latent from encoder
        Returns:
            (B, max_objects, obj_dim)
        """
        slots = self.slot_embeddings.weight          # (M, latent_dim)
        z_slotted = z.unsqueeze(1) + slots.unsqueeze(0)  # (B, M, latent_dim)
        return self.mlp(z_slotted)                   # (B, M, obj_dim)


class SetAutoencoder(nn.Module):
    """Encoder-decoder wrapper for set prediction.

    Composes a set-equivariant encoder with any decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) input set
            mask: (B, N) binary mask
        Returns:
            (B, M, D) predicted set
        """
        z = self.encoder(x, mask)
        return self.decoder(z)
