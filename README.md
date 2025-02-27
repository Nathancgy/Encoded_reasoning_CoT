# Encoded Reasoning Detector

This project explores the use of Sparse Autoencoders (SAEs) to detect encoded reasoning in language models' Chain of Thought (CoT).

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -e .
```

## Running the Prototype

The prototype consists of several components:

1. `extract_activations.py` - Extracts activations from a small language model during CoT generation
2. `train_sae.py` - Trains a sparse autoencoder on the extracted activations
3. `analyze_features.py` - Analyzes SAE features to detect potential encoded reasoning
4. `run_experiment.py` - End-to-end script that runs the complete pipeline

To run the complete pipeline:
```bash
python run_experiment.py
```

## Components

- `src/` - Contains the core functionality
  - `models.py` - Model loading and utility functions
  - `sae.py` - Sparse autoencoder implementation
  - `data.py` - Dataset creation and processing
  - `visualization.py` - Visualization utilities

## Example

The prototype uses a small language model (Qwen 1.5B) to solve simple arithmetic problems. It then analyzes the model's activations to detect if the model encodes information that isn't explicitly stated in its CoT. 