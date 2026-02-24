#!/usr/bin/env python3
"""register_criterion.py — helper for integrating InfluencerCriterion with mmdet3d.

Three modes:

  --verify
      Import mmdet3d and influencerformer, then check MODELS.get('InfluencerCriterion').
      Prints PASS or FAIL with a fix suggestion. Use this to confirm the integration
      is working before running training.

  --custom-imports
      Print the custom_imports snippet to add to any mmdet3d config file.
      This is the preferred (non-invasive) integration approach — no source
      modifications to OneFormer3D are needed.

  --patch
      Write a one-line import sentinel into OneFormer3D's package __init__.py.
      Use this only if custom_imports is not available (e.g., very old mmdet3d).
      The patch is idempotent: re-running does nothing if already applied.

Usage:
    python scripts/register_criterion.py --verify
    python scripts/register_criterion.py --custom-imports
    python scripts/register_criterion.py --patch
"""

import argparse
import importlib.util
import sys


SENTINEL = '# [influencerformer-patch]'
PATCH_LINE = (
    f'{SENTINEL} -- added by register_criterion.py\n'
    'import influencerformer  # registers InfluencerCriterion in MODELS registry\n'
)


def _find_oneformer3d_init():
    """Return path to oneformer3d package __init__.py, or None."""
    spec = importlib.util.find_spec('oneformer3d')
    if spec is None:
        return None
    return spec.origin  # e.g. /path/to/oneformer3d/__init__.py


def cmd_verify(args):
    """Check that InfluencerCriterion is reachable in the MODELS registry."""
    # Step 1: ensure mmdet3d is importable
    try:
        from mmdet3d.registry import MODELS
    except ImportError:
        print('FAIL: mmdet3d is not installed or not importable.')
        print('      Install OneFormer3D and its dependencies first.')
        sys.exit(1)

    # Step 2: ensure influencerformer is importable
    try:
        import influencerformer  # noqa: F401  (triggers MODELS registration)
    except ImportError:
        print('FAIL: influencerformer is not importable.')
        print('      Add the repo root to PYTHONPATH:')
        print('          export PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH')
        sys.exit(1)

    # Step 3: check registry
    cls = MODELS.get('InfluencerCriterion')
    if cls is None:
        print('FAIL: InfluencerCriterion not found in MODELS registry.')
        print('      Possible causes:')
        print('      1. influencerformer was imported but mmdet3d was not importable')
        print('         at that time (check import order).')
        print('      2. The registration block in criterion.py was not reached.')
        print('      Try: python -c "import mmdet3d; import influencerformer"')
        sys.exit(1)

    print('PASS: InfluencerCriterion is registered in the MODELS registry.')
    print(f'      Registered class: {cls}')


def cmd_custom_imports(args):
    """Print the custom_imports snippet for mmdet3d config files."""
    snippet = (
        '# Add this at the top of any mmdet3d config to register InfluencerCriterion:\n'
        'custom_imports = dict(imports=["influencerformer"], allow_failed_imports=False)\n'
        '\n'
        '# Also ensure PYTHONPATH includes the InfluencerFormer repo root:\n'
        '#   export PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH\n'
        '# or install the package:\n'
        '#   pip install -e /path/to/InfluencerFormer\n'
    )
    print(snippet)


def cmd_patch(args):
    """Write a one-line import into OneFormer3D's __init__.py."""
    init_path = _find_oneformer3d_init()
    if init_path is None:
        print('ERROR: oneformer3d package not found.')
        print('       Install OneFormer3D first, then re-run.')
        sys.exit(1)

    with open(init_path, 'r') as f:
        content = f.read()

    if SENTINEL in content:
        print(f'INFO: Already patched — sentinel found in {init_path}')
        print('      No changes made.')
        return

    patched = PATCH_LINE + content
    with open(init_path, 'w') as f:
        f.write(patched)

    print(f'PATCHED: Wrote import sentinel to {init_path}')
    print('         Run --verify to confirm the registration works.')
    print()
    print('NOTE: This modifies OneFormer3D source. Prefer using custom_imports')
    print('      in your config file instead (see --custom-imports).')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify', action='store_true',
                       help='Check that InfluencerCriterion is registered')
    group.add_argument('--custom-imports', action='store_true',
                       help='Print the custom_imports config snippet')
    group.add_argument('--patch', action='store_true',
                       help='Patch OneFormer3D __init__.py (last resort)')
    args = parser.parse_args()

    if args.verify:
        cmd_verify(args)
    elif args.custom_imports:
        cmd_custom_imports(args)
    elif args.patch:
        cmd_patch(args)


if __name__ == '__main__':
    main()
