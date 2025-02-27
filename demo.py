#!/usr/bin/env python
"""
Demo script showing how to use the encoded reasoning detection framework.
This script runs a simplified version of the experiment with minimal configuration.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import ActivationExtractor, create_arithmetic_dataset
from src.sae import SparseAutoencoder
from src.data import create_hidden_variable_dataset
from src.visualization import plot_feature_correlations, plot_feature_prediction, plot_feature_activations

def main():
    print("=== Encoded Reasoning Detection Demo ===")
    
    # Configuration
    model_name = "Qwen/Qwen1.5-1.8B"  # Small model for quick testing
    num_problems = 10  # Small dataset for demo
    feature_dim = 64  # Small feature dimension
    epochs = 5  # Minimal training
    output_dir = "demo_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create a small dataset of arithmetic problems
    print("\nStep 1: Creating a small dataset of arithmetic problems...")
    problems = create_arithmetic_dataset(num_problems=num_problems)
    for i, problem in enumerate(problems[:3]):  # Show first few problems
        print(f"  Problem {i+1}: {problem}")
    
    # Step 2: Load model and extract activations
    print("\nStep 2: Loading model and extracting activations...")
    extractor = ActivationExtractor(model_name=model_name)
    
    # Use middle layer
    layer = extractor.model.cfg.n_layers // 2
    print(f"  Using middle layer: {layer}")
    
    # Generate solutions and extract activations
    print("  Generating solutions and extracting activations...")
    problems, solutions, activations = extractor.create_dataset_from_problems(
        problems=problems,
        layer=layer,
        max_new_tokens=100
    )
    
    print(f"  Extracted activations with shape: {activations.shape}")
    
    # Save example solutions
    print("  Saving example solutions...")
    with open(os.path.join(output_dir, "demo_solutions.txt"), "w") as f:
        for i, (problem, solution) in enumerate(zip(problems, solutions)):
            f.write(f"Problem: {problem}\n\n")
            f.write(f"Solution:\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Step 3: Create dataset with hidden variables
    print("\nStep 3: Creating dataset with hidden variables...")
    activations_data, hidden_vars = create_hidden_variable_dataset(
        problems=problems,
        solutions=solutions,
        activations=activations
    )
    
    print("  Hidden variables extracted:")
    for var_name, var_data in hidden_vars.items():
        if var_data.ndim == 1:
            print(f"    {var_name}: shape={var_data.shape}")
    
    # Step 4: Train SAE on activations
    print("\nStep 4: Training SAE on activations...")
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        l1_coefficient=1e-3
    )
    
    # Train SAE
    metrics = sae.train_on_activations(activations, num_epochs=epochs)
    
    # Save SAE
    sae.save(os.path.join(output_dir, "demo_sae_model.pt"))
    
    # Step 5: Encode activations with SAE
    print("\nStep 5: Encoding activations with SAE...")
    feature_activations = sae.encode(activations)
    
    print(f"  Feature activations shape: {feature_activations.shape}")
    
    # Step 6: Analyze features
    print("\nStep 6: Analyzing features...")
    
    # Create a directory for feature visualizations
    feature_dir = os.path.join(output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Find features that correlate with hidden variables
    for var_name in ['final_answer', 'is_correct', 'first_operand', 'second_operand']:
        if var_name in hidden_vars:
            print(f"  Analyzing correlation with {var_name}...")
            
            # Plot correlations
            top_features = plot_feature_correlations(
                feature_activations,
                hidden_vars,
                var_name,
                save_path=os.path.join(feature_dir, f"correlations_{var_name}.png")
            )
            
            # For top feature, plot prediction
            if len(top_features) > 0:
                top_feature_idx = top_features[0][0]
                top_corr = top_features[0][1]
                print(f"    Top feature {top_feature_idx} has correlation {top_corr:.3f}")
                
                is_classification = var_name == 'is_correct'
                plot_feature_prediction(
                    feature_activations,
                    hidden_vars,
                    var_name,
                    top_feature_idx,
                    save_path=os.path.join(feature_dir, f"feature_{top_feature_idx}_vs_{var_name}.png"),
                    is_classification=is_classification
                )
    
    # Step 7: Create a simple visualization of feature activations
    print("\nStep 7: Creating visualization of feature activations...")
    
    # Get top features by activation variance
    feature_vars = np.var(feature_activations, axis=0)
    top_feature_indices = np.argsort(feature_vars)[::-1][:10]  # Top 10 features
    
    plt.figure(figsize=(12, 8))
    plt.imshow(feature_activations[:, top_feature_indices].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Activation')
    plt.xlabel('Token Index')
    plt.ylabel('Feature Index')
    plt.title('Top Feature Activations Across Tokens')
    plt.yticks(range(len(top_feature_indices)), top_feature_indices)
    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, "feature_activations_heatmap.png"))
    
    print("\nDemo completed successfully!")
    print(f"Results saved to {output_dir}")
    print("\nNext steps:")
    print("1. Examine the feature visualizations in the output directory")
    print("2. Try running the full experiment with more data using run_experiment.py")
    print("3. Analyze the pre-trained SAE model using analyze_sae.py")

if __name__ == "__main__":
    main() 