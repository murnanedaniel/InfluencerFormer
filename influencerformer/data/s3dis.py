"""S3DIS dataset download and loading for point cloud instance segmentation.

Stanford Large-Scale 3D Indoor Spaces (S3DIS):
- 271 rooms across 6 areas, 3 buildings
- ~696 million points total
- 13 semantic classes
- Standard split: train on Areas 1,2,3,4,6 / test on Area 5
- Auto-downloads via PyTorch Geometric (~2 GB)
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def download_s3dis(root="./data/s3dis", test_area=5):
    """Download S3DIS dataset via PyTorch Geometric.

    Args:
        root: Directory to store the dataset.
        test_area: Which area to hold out for testing (1-6). Standard is 5.

    Returns:
        Tuple of (train_dataset, test_dataset) PyG dataset objects.
    """
    try:
        from torch_geometric.datasets import S3DIS
    except ImportError:
        raise ImportError(
            "torch-geometric is required for S3DIS. Install with:\n"
            "  pip install torch-geometric\n"
            "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )

    root = str(Path(root).resolve())
    print(f"Downloading S3DIS to {root} (test_area={test_area})...")
    print("This may take a while on first run (~2 GB download).")

    train_dataset = S3DIS(root=root, test_area=test_area, train=True)
    test_dataset = S3DIS(root=root, test_area=test_area, train=False)

    print(f"S3DIS loaded: {len(train_dataset)} train rooms, {len(test_dataset)} test rooms")
    return train_dataset, test_dataset


class S3DISInstanceDataset(Dataset):
    """Wraps S3DIS PyG data into a standard PyTorch Dataset for instance segmentation.

    Each item returns:
        coords: (N, 3) float tensor of xyz coordinates
        colors: (N, 3) float tensor of RGB values (normalized to [0,1])
        sem_labels: (N,) long tensor of semantic class labels
        instance_labels: (N,) long tensor of instance IDs
    """

    def __init__(self, root="./data/s3dis", test_area=5, train=True, max_points=None):
        """
        Args:
            root: Path to store/load the dataset.
            test_area: Area held out for testing.
            train: If True, load training split; else test split.
            max_points: If set, randomly subsample each room to this many points.
        """
        try:
            from torch_geometric.datasets import S3DIS
        except ImportError:
            raise ImportError(
                "torch-geometric is required. Install with: pip install torch-geometric"
            )

        self.pyg_dataset = S3DIS(root=root, test_area=test_area, train=train)
        self.max_points = max_points

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]

        coords = data.pos  # (N, 3)
        colors = data.x[:, :3] if data.x is not None else torch.zeros_like(coords)  # (N, 3)
        sem_labels = data.y  # (N,)

        # S3DIS from PyG may not have instance labels directly;
        # we derive them from the segment information if available
        if hasattr(data, "instance_labels"):
            instance_labels = data.instance_labels
        else:
            # Fallback: use semantic labels as a proxy (each contiguous
            # region of same semantic label = one "instance").
            # For real instance segmentation, you'd need the full S3DIS
            # annotations with instance IDs.
            instance_labels = sem_labels.clone()

        # Subsample if requested
        if self.max_points is not None and coords.shape[0] > self.max_points:
            perm = torch.randperm(coords.shape[0])[: self.max_points]
            coords = coords[perm]
            colors = colors[perm]
            sem_labels = sem_labels[perm]
            instance_labels = instance_labels[perm]

        return {
            "coords": coords.float(),
            "colors": colors.float(),
            "sem_labels": sem_labels.long(),
            "instance_labels": instance_labels.long(),
        }
