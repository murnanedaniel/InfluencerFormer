"""MS-COCO dataset download and loading for image instance segmentation.

MS-COCO (Common Objects in Context):
- 118K train, 5K val images
- 80 instance classes, ~860K instance annotations
- Primary metric: AP (mAP at IoU 0.50:0.95)
- Download via fiftyone or manual download + pycocotools
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def download_coco(root="./data/coco", split="val", max_samples=None):
    """Download COCO dataset via fiftyone (easiest auto-download).

    Args:
        root: Base directory. fiftyone manages its own storage but we
              return the dataset object for access.
        split: "train" or "val" (default "val" for quick experiments).
        max_samples: If set, only download this many images.

    Returns:
        A fiftyone Dataset object.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        raise ImportError(
            "fiftyone is required for auto-downloading COCO. Install with:\n"
            "  pip install fiftyone\n"
            "Alternatively, manually download from https://cocodataset.org/"
        )

    split_map = {"train": "train", "val": "validation", "test": "test"}
    fo_split = split_map.get(split, split)

    kwargs = {
        "split": fo_split,
        "label_types": ["segmentations"],
    }
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    print(f"Downloading COCO-2017 {split} split via fiftyone...")
    if max_samples:
        print(f"  (limited to {max_samples} samples)")

    dataset = foz.load_zoo_dataset("coco-2017", **kwargs)
    print(f"COCO loaded: {len(dataset)} images")
    return dataset


def download_coco_torchvision(root="./data/coco", split="val"):
    """Download COCO via torchvision (requires manual image download first).

    This function provides instructions and returns a torchvision dataset
    if the data is already present.

    Args:
        root: Directory containing coco/val2017/ and coco/annotations/.
        split: "train" or "val".

    Returns:
        A torchvision CocoDetection dataset.
    """
    try:
        from torchvision.datasets import CocoDetection
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")

    year = "2017"
    img_dir = Path(root) / f"{split}{year}"
    ann_file = Path(root) / "annotations" / f"instances_{split}{year}.json"

    if not img_dir.exists() or not ann_file.exists():
        print("COCO data not found. To download manually:")
        print(f"  1. Download images: http://images.cocodataset.org/zips/{split}{year}.zip")
        print(f"  2. Download annotations: http://images.cocodataset.org/annotations/annotations_trainval{year}.zip")
        print(f"  3. Extract to: {root}/")
        print(f"     Expected structure:")
        print(f"       {root}/{split}{year}/  (images)")
        print(f"       {root}/annotations/instances_{split}{year}.json")
        print()
        print("Or use download_coco() with fiftyone for automatic download.")
        raise FileNotFoundError(f"COCO data not found at {root}")

    dataset = CocoDetection(root=str(img_dir), annFile=str(ann_file))
    print(f"COCO loaded via torchvision: {len(dataset)} images")
    return dataset


class COCOInstanceDataset(Dataset):
    """COCO instance segmentation dataset using pycocotools.

    Loads images and their instance masks. Each item returns:
        image: (3, H, W) float tensor normalized to [0,1]
        masks: (K, H, W) bool tensor of K instance masks
        labels: (K,) long tensor of category IDs
        boxes: (K, 4) float tensor of bounding boxes [x1, y1, x2, y2]
    """

    def __init__(self, root="./data/coco", split="val", max_size=640):
        """
        Args:
            root: Directory containing coco images and annotations.
            split: "train" or "val".
            max_size: Resize longest edge to this (for memory efficiency).
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required. Install with: pip install pycocotools"
            )

        self.root = Path(root)
        self.split = split
        self.max_size = max_size

        ann_file = self.root / "annotations" / f"instances_{split}2017.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotations not found at {ann_file}. "
                "Use download_coco() or download_coco_torchvision() first."
            )

        self.coco = COCO(str(ann_file))
        self.img_ids = sorted(self.coco.getImgIds())
        self.img_dir = self.root / f"{split}2017"

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Resize if needed
        if max(w, h) > self.max_size:
            scale = self.max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            new_w, new_h = w, h
            scale = 1.0

        image_tensor = TF.to_tensor(image)  # (3, H, W), [0, 1]

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        labels = []
        boxes = []

        for ann in anns:
            mask = self.coco.annToMask(ann)  # (h, w) binary
            if scale != 1.0:
                from PIL import Image as PILImage

                mask_pil = PILImage.fromarray(mask.astype(np.uint8))
                mask_pil = mask_pil.resize((new_w, new_h), PILImage.NEAREST)
                mask = np.array(mask_pil)

            masks.append(torch.from_numpy(mask).bool())
            labels.append(ann["category_id"])

            # Scale bounding box
            x, y, bw, bh = ann["bbox"]
            boxes.append([x * scale, y * scale, (x + bw) * scale, (y + bh) * scale])

        if len(masks) > 0:
            masks = torch.stack(masks)  # (K, H, W)
            labels = torch.tensor(labels, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            masks = torch.zeros((0, new_h, new_w), dtype=torch.bool)
            labels = torch.zeros((0,), dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image": image_tensor,
            "masks": masks,
            "labels": labels,
            "boxes": boxes,
            "image_id": img_id,
        }
