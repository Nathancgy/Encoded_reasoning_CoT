#!/usr/bin/env python
"""
Simplified version of the experiment for quick testing.
Uses a very small dataset and minimal SAE training.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import ActivationExtractor, create_arithmetic_dataset
from src.sae import SparseAutoencoder
from src.data import create_hidden_variable_dataset
from src.visualization import plot_feature_correlations, plot_feature_prediction

def main():
    # Configuration
    model_name = "Qwen/Qwen1.5-1.8B"  # Small model for quick testing
    num_problems = 5  # Very small dataset for testing
    feature_dim = 32  # Small feature dimension
    epochs = 3  # Minimal training
    output_dir = "test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Running quick test of encoded reasoning detection ===")
    print(f"Model: {model_name}")
    print(f"Number of problems: {num_problems}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Create a tiny dataset
    print("Creating test dataset...")
    problems = create_arithmetic_dataset(num_problems=num_problems)
    
    # Step 2: Load model and extract activations
    print("Loading model and extracting activations...")
    extractor = ActivationExtractor(model_name=model_name)
    
    # Use middle layer
    layer = extractor.model.cfg.n_layers // 2
    print(f"Using middle layer: {layer}")
    
    # Generate solutions and extract activations
    problems, solutions, activations = extractor.create_dataset_from_problems(
        problems=problems,
        layer=layer,
        max_new_tokens=100  # Shorter generations for testing
    )
    
    print(f"Extracted activations with shape: {activations.shape}")
    
    # Save example solutions
    with open(os.path.join(output_dir, "test_solutions.txt"), "w") as f:
        for problem, solution in zip(problems, solutions):
            f.write(f"Problem: {problem}\n\n")
            f.write(f"Solution:\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Step 3: Create dataset with hidden variables
    print("Creating dataset with hidden variables...")
    activations_data, hidden_vars = create_hidden_variable_dataset(
        problems=problems,
        solutions=solutions,
        activations=activations
    )
    
    # Step 4: Train a tiny SAE
    print("Training SAE...")
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        l1_coefficient=1e-3
    )
    
    # Train SAE for just a few epochs
    metrics = sae.train_on_activations(activations, num_epochs=epochs)
    
    # Step 5: Encode activations with SAE
    print("Encoding activations with SAE...")
    feature_activations = sae.encode(activations)
    
    # Step 6: Create a few basic visualizations
    print("Creating visualizations...")
    
    # Plot correlations with final answer
    if 'final_answer' in hidden_vars:
        top_features = plot_feature_correlations(
            feature_activations,
            hidden_vars,
            'final_answer',
            save_path=os.path.join(output_dir, "test_correlations.png")
        )
        
        # Plot top feature vs final answer
        if len(top_features) > 0:
            top_feature_idx = top_features[0][0]
            plot_feature_prediction(
                feature_activations,
                hidden_vars,
                'final_answer',
                top_feature_idx,
                save_path=os.path.join(output_dir, "test_prediction.png")
            )
    
    print("\nQuick test completed!")
    print(f"Results saved to {output_dir}")
    print("\nThis was a minimal test with a tiny dataset.")
    print("For real experiments, use run_experiment.py with more data and training.")

if __name__ == "__main__":
    main() 