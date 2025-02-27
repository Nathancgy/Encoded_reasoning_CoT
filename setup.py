from setuptools import setup, find_packages

setup(
    name="encoded_reasoning_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "transformer_lens>=1.2.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "datasets>=2.13.0",
        "einops>=0.6.0",
        "wandb>=0.15.0",  # optional for tracking experiments
    ],
) 