"""Find the maximum safe batch size for the CLEVR SetAutoencoder on one A100.

Run on a compute node (not the login node):

    python scripts/find_max_batch_size.py [--d_model 128] [--headroom 0.20]

Tries doubling batch sizes from 64 upward. For each size, runs a full
forward + backward pass, records peak GPU memory, and stops at OOM.
Prints the largest size that fits with the requested headroom fraction.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── path setup ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'notebooks'))
sys.path.insert(0, str(_REPO))

import clevr_utils as cu

# ── args ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d_model',   type=int,   default=128)
    p.add_argument('--headroom',  type=float, default=0.20,
                   help='Fraction of GPU memory to keep free (default 0.20)')
    return p.parse_args()


def try_batch(model: nn.Module, batch_size: int, loss_fn: nn.Module) -> dict:
    """Attempt one forward+backward with batch_size. Returns memory stats or raises."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Synthetic batch: same shapes as CLEVRSetDataset
    inp  = torch.randn(batch_size, cu.MAX_OBJECTS, cu.OBJ_DIM, device=cu.DEVICE)
    tgt  = torch.randn(batch_size, cu.MAX_OBJECTS, cu.OBJ_DIM, device=cu.DEVICE)
    mask = torch.ones(batch_size,  cu.MAX_OBJECTS,              device=cu.DEVICE)

    model.train()
    preds = model(inp, mask=mask)
    D = torch.cdist(preds, tgt)
    loss = loss_fn(D, mask=mask)
    loss.backward()
    model.zero_grad()

    peak  = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {'peak_gb': peak, 'total_gb': total, 'fraction': peak / total}


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print('ERROR: No CUDA device visible. Run on a compute node, not the login node.')
        print('  salloc -q interactive -A m4958 -C gpu -N 1 --gpus 4 -t 02:00:00')
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_gb    = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'Device: {device_name}  ({total_gb:.1f} GB total)')
    print(f'Model:  SetTransformerEncoder + MLPSetDecoder  (d_model={args.d_model})\n')

    from influencerformer.losses import PowerSoftMinLoss
    model   = cu.make_model(d_model=args.d_model, latent_dim=args.d_model, hidden_dim=args.d_model)
    loss_fn = PowerSoftMinLoss(temperature=0.12, power=3.0)

    batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    last_ok     = None
    last_stats  = None

    print(f'{"Batch":>8}  {"Peak (GB)":>10}  {"Fraction":>9}  {"Status":>8}')
    print('-' * 46)

    for bs in batch_sizes:
        try:
            stats = try_batch(model, bs, loss_fn)
            used_frac = stats['fraction']
            print(f'{bs:>8}  {stats["peak_gb"]:>10.2f}  {used_frac:>8.1%}  {"OK":>8}')
            if used_frac <= (1.0 - args.headroom):
                last_ok    = bs
                last_stats = stats
        except torch.cuda.OutOfMemoryError:
            print(f'{bs:>8}  {"—":>10}  {"—":>9}  {"OOM":>8}')
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f'{bs:>8}  error: {e}')
            break

    print()
    if last_ok is None:
        print('WARNING: Even batch_size=64 exceeds the headroom budget.')
        print('Reduce --headroom or use a smaller model.')
    else:
        print(f'Recommended batch size: {last_ok}')
        print(f'  Peak memory at {last_ok}: {last_stats["peak_gb"]:.2f} GB '
              f'({last_stats["fraction"]:.1%} of {last_stats["total_gb"]:.1f} GB)')
        print(f'  Remaining headroom: '
              f'{(1 - last_stats["fraction"]) * last_stats["total_gb"]:.1f} GB')
        print(f'\nUse this in train_study.py:')
        print(f'  --batch_size {last_ok}')


if __name__ == '__main__':
    main()
