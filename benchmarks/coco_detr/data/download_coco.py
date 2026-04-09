"""Download COCO images and annotations into the benchmark layout."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


RESOURCE_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'test2017': 'http://images.cocodataset.org/zips/test2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco_root', type=str, default='data/coco', help='COCO download root.')
    parser.add_argument(
        '--resources',
        nargs='+',
        default=['train2017', 'val2017', 'annotations'],
        choices=sorted(RESOURCE_URLS),
        help='COCO resources to download.',
    )
    parser.add_argument('--dry_run', action='store_true', help='Print planned downloads only.')
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Download one file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, 'wb') as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(target_dir)


def main() -> None:
    """Download and extract requested COCO resources."""
    args = parse_args()
    coco_root = Path(args.coco_root).resolve()
    for resource in args.resources:
        url = RESOURCE_URLS[resource]
        zip_path = coco_root / f'{resource}.zip'
        print(f'{resource}: {url}')
        print(f'  zip -> {zip_path}')
        if args.dry_run:
            continue
        if not zip_path.exists():
            download_file(url, zip_path)
        extract_zip(zip_path, coco_root)
        print(f'  extracted into {coco_root}')


if __name__ == '__main__':
    main()
