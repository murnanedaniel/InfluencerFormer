"""Create a tiny COCO-style dataset for benchmark smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


CATEGORIES = [
    {'id': 1, 'name': 'person', 'supercategory': 'person'},
]


SPECS = {
    'train2017': [
        {'image_id': 1, 'file_name': '000000000001.jpg', 'bbox': [20, 18, 54, 70]},
        {'image_id': 2, 'file_name': '000000000002.jpg', 'bbox': [36, 30, 48, 56]},
    ],
    'val2017': [
        {'image_id': 3, 'file_name': '000000000003.jpg', 'bbox': [22, 24, 52, 68]},
        {'image_id': 4, 'file_name': '000000000004.jpg', 'bbox': [30, 22, 58, 62]},
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', type=str, default='benchmarks/coco_detr/data/tiny_coco', help='Tiny COCO root.')
    return parser.parse_args()


def build_split(root: Path, split: str, specs: list[dict]) -> None:
    """Create one COCO split."""
    image_dir = root / split
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for idx, spec in enumerate(specs, start=1):
        width, height = 128, 128
        image = Image.new('RGB', (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(image)
        x, y, w, h = spec['bbox']
        draw.rectangle([x, y, x + w, y + h], outline=(240, 80, 80), fill=(220, 100, 100))
        image.save(image_dir / spec['file_name'])
        images.append({'id': spec['image_id'], 'file_name': spec['file_name'], 'width': width, 'height': height})
        annotations.append(
            {
                'id': idx,
                'image_id': spec['image_id'],
                'category_id': 1,
                'bbox': spec['bbox'],
                'area': spec['bbox'][2] * spec['bbox'][3],
                'iscrowd': 0,
            }
        )
    ann_dir = root / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    payload = {'images': images, 'annotations': annotations, 'categories': CATEGORIES}
    (ann_dir / f'instances_{split}.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    """Create the tiny COCO dataset."""
    args = parse_args()
    root = Path(args.output_root).resolve()
    for split, specs in SPECS.items():
        build_split(root, split, specs)
    print(root)


if __name__ == '__main__':
    main()
