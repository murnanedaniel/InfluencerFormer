from setuptools import setup, find_packages

setup(
    name="influencerformer",
    version="0.2.0",
    description="End-to-end instance segmentation via learned condensation with Influencer Loss",
    author="Daniel Murnane",
    author_email="dtmurnane@lbl.gov",
    url="https://github.com/murnanedaniel/InfluencerFormer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "pointcloud": [
            "torch-geometric",
            "open3d",
        ],
        # Full OneFormer3D integration stack.
        # NOTE: mmdet3d itself (the filaPro/oneformer3d fork) is not on PyPI
        # — install from source: pip install -e /path/to/oneformer3d
        # NOTE: spconv requires a CUDA-version-specific wheel. Select the one
        # matching your CUDA installation:
        #   CUDA 11.3 → spconv-cu113
        #   CUDA 11.6 → spconv-cu116
        #   CUDA 11.8 → spconv-cu118  (listed below as default example)
        #   CUDA 12.x → spconv-cu120
        # On macOS / CPU-only: use spconv-cpu instead.
        "mmdet3d": [
            "mmengine>=0.10.0",
            "mmcv>=2.0.0",
            "mmdet>=3.0.0",
            "spconv-cu118>=2.3.0 ; sys_platform!='darwin'",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "ruff",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
