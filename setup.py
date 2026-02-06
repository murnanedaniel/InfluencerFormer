from setuptools import setup, find_packages

setup(
    name="influencerformer",
    version="0.1.0",
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
        "dev": [
            "pytest",
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
