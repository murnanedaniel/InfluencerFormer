# PointSWD Benchmark Setup

## Quick Start

```bash
# 1. Clone PointSWD
cd /path/to/workspace
git clone https://github.com/VinAIResearch/PointSWD.git
cd PointSWD

# 2. Install dependencies
pip install torch numpy scipy scikit-learn h5py tqdm gdown

# 3. Download ShapeNet data
cd dataset && bash download_shapenet_core55_catagories.sh && cd ..

# 4. Copy our custom losses
cp /path/to/InfluencerFormer/examples/pointswd_custom_losses.py loss/custom_losses.py

# 5. Patch loss/__init__.py to handle missing CUDA extensions
cat > loss/__init__.py << 'INIT'
try:
    from .chamfer import Chamfer
    from .emd import EMD
except (ImportError, ModuleNotFoundError):
    print("Warning: CUDA extensions not compiled.")
    Chamfer = None
    EMD = None
from .sw_variants import ASW, SWD, GenSW, MaxSW
from .custom_losses import PowerSoftMin, SoftMinChamfer, PWsoftmin, PureTorchChamfer
INIT

# 6. Patch train.py: add our losses to the elif chain (after line ~153)
# Add these loss options:
#   power_softmin, softmin_chamfer, pw_softmin, pytorch_chamfer

# 7. Remove interactive prompt in train.py (lines 64-74)

# 8. Run comparison
cp /path/to/InfluencerFormer/examples/pointswd_comparison.py .
python run_comparison.py  # CPU: ~7 hours. GPU: ~1 hour.

# Or quick sanity check
cp /path/to/InfluencerFormer/examples/pointswd_sanity.py .
python quick_sanity.py  # CPU: ~5 minutes
```

## What's Being Tested

ShapeNet point cloud autoencoding: PointNetAE (256-dim latent) encodes and
decodes 512-point 3D shapes. We compare reconstruction quality (Chamfer
distance) across different training losses.

## Losses Compared

| Loss | Type | Complexity |
|---|---|---|
| PureTorchChamfer | Baseline | O(NM) |
| PowerSoftMin p=3 | **Ours** | O(NM) |
| SoftMinChamfer | Ours | O(NM) |
| PWsoftmin | Ours | O(NM) |
| SWD | PointSWD paper | O(N log N) per projection |

## Key Notes

- Temperature τ=0.01 is calibrated for N=512 3D point clouds (inter-point
  distance ~0.05). Adjust if using different point counts.
- CUDA extensions (for the paper's original Chamfer/EMD) require compilation.
  Our losses use pure PyTorch and work without CUDA extensions.
- Batch size 256 (paper default) works for SWD but is too memory-heavy for
  our distance-matrix losses at N=512. Use batch_size=16-32 for our losses.
