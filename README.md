# Encoded Reasoning Detector

This project explores the use of Sparse Autoencoders (SAEs) to detect encoded reasoning in language models' Chain of Thought (CoT). It provides a framework for analyzing whether models hide information in their activations that isn't explicitly stated in their reasoning traces.

## Project Overview

Modern AI models often use Chain of Thought (CoT) to explain their reasoning, but if they hide critical steps internally (encoded reasoning), safety teams can't trust these explanations. This project uses Sparse Autoencoders to analyze model activations during CoT generation to detect potential encoded reasoning.

Key features:
- Extract activations from language models during CoT generation
- Train SAEs on these activations to identify meaningful features
- Analyze correlations between SAE features and hidden variables
- Visualize patterns in feature activations that might indicate encoded reasoning

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -e .
```

## Quick Start

Run the demo script to see the framework in action:
```bash
python demo.py
```

This will:
1. Generate a small dataset of arithmetic problems
2. Extract activations from a small language model (Qwen 1.5B)
3. Train a small SAE on these activations
4. Analyze and visualize the results

## Components

The project consists of several components:

### Core Modules

- `src/models.py` - Model loading and activation extraction
- `src/sae.py` - Sparse autoencoder implementation
- `src/data.py` - Dataset creation and processing
- `src/visualization.py` - Visualization utilities
- `src/complex_dataset.py` - Generation of more complex problems

### Scripts

- `demo.py` - Simple demonstration of the framework
- `quick_test.py` - Minimal test of the framework
- `run_experiment.py` - Full experiment with simple arithmetic problems
- `run_complex_experiment.py` - Experiment with more complex problems
- `analyze_sae.py` - Analyze a pre-trained SAE model

## Running Experiments

### Basic Experiment

Run a basic experiment with simple arithmetic problems:
```bash
python run_experiment.py --num_problems 50 --feature_dim 128 --epochs 20
```

### Complex Experiment

Run an experiment with more complex problems that might induce encoded reasoning:
```bash
python run_complex_experiment.py --num_problems 30 --feature_dim 256 --epochs 20
```

### Analyzing Results

Analyze a pre-trained SAE model:
```bash
python analyze_sae.py --sae_model results/sae_model.pt --output_dir analysis
```

You can also analyze a specific prompt:
```bash
python analyze_sae.py --sae_model results/sae_model.pt --prompt "Calculate 25 * 16."
```

## Example Use Cases

1. **Detecting Encoded Reasoning**: Identify if models are storing information in their activations that isn't explicitly stated in their CoT.

2. **Understanding CoT Mechanisms**: Gain insights into how models actually perform chain-of-thought reasoning.

3. **Evaluating Model Faithfulness**: Assess whether a model's explanations faithfully represent its internal reasoning process.

4. **Safety Monitoring**: Develop tools to detect potential deception or hidden reasoning in AI systems.

## Interpreting Results

When analyzing results, look for:

1. **Features that correlate with hidden variables**: If SAE features strongly correlate with variables not explicitly mentioned in the CoT, this suggests encoded reasoning.

2. **Token-specific features**: Features that activate on specific token types (numbers, operators) might encode semantic information.

3. **Activation patterns**: Patterns in feature activations across tokens might reveal how the model processes information.

## Extending the Project

This framework can be extended in several ways:

1. **Larger Models**: Apply the same techniques to larger models like Claude or GPT-4.

2. **Different Tasks**: Extend beyond arithmetic to more complex reasoning tasks.

3. **Intervention Studies**: Modify activations to see how they affect the model's outputs.

4. **Training Analysis**: Track the emergence of encoded reasoning throughout model training.

## Citation

If you use this code in your research, please cite:

```
@software{encoded_reasoning_detector,
  author = {Your Name},
  title = {Encoded Reasoning Detector},
  year = {2023},
  url = {https://github.com/yourusername/encoded_reasoning_detector}
}
``` 