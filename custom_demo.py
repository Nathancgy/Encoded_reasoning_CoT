#!/usr/bin/env python
"""
Simplified demo script for the encoded reasoning detection framework.
This version uses synthetic data instead of a real model to demonstrate the framework.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.sae import SparseAutoencoder
from src.visualization import plot_feature_correlations, plot_feature_prediction, plot_feature_activations

def create_synthetic_activations(num_samples=1000, input_dim=256, hidden_variables=None):
    """
    Create synthetic activations with encoded patterns for demonstration.
    
    Args:
        num_samples: Number of activation vectors to generate
        input_dim: Dimension of each activation vector
        hidden_variables: Dict of hidden variables to encode in the activations
        
    Returns:
        Tuple of (activations, hidden_vars)
    """
    print(f"Generating {num_samples} synthetic activations with dim {input_dim}...")
    
    # Create random base activations
    activations = np.random.randn(num_samples, input_dim) * 0.1
    
    # Create hidden variables if not provided
    if hidden_variables is None:
        hidden_variables = {}
        # Variable 1: A random number between 0 and 10
        hidden_variables['value'] = np.random.uniform(0, 10, size=num_samples)
        # Variable 2: A binary class
        hidden_variables['binary_class'] = np.random.choice([0, 1], size=num_samples)
        # Variable 3: A categorical variable with 4 classes (one-hot encoded)
        categories = np.random.choice(4, size=num_samples)
        hidden_variables['category'] = np.eye(4)[categories]
    
    # Encode hidden variables into the activations
    # For 'value': Create a correlation with a specific direction in activation space
    value_direction = np.random.randn(input_dim)
    value_direction = value_direction / np.linalg.norm(value_direction)
    for i in range(num_samples):
        # Add the value (scaled) in the value_direction
        activations[i] += value_direction * hidden_variables['value'][i] * 0.5
    
    # For 'binary_class': Create a different pattern for each class
    class0_pattern = np.random.randn(input_dim)
    class1_pattern = np.random.randn(input_dim)
    for i in range(num_samples):
        if hidden_variables['binary_class'][i] == 0:
            activations[i] += class0_pattern * 0.5
        else:
            activations[i] += class1_pattern * 0.5
    
    # For 'category': Create a different pattern for each category
    category_patterns = [np.random.randn(input_dim) for _ in range(4)]
    for i in range(num_samples):
        category_idx = np.argmax(hidden_variables['category'][i])
        activations[i] += category_patterns[category_idx] * 0.5
    
    # Add some noise
    activations += np.random.randn(num_samples, input_dim) * 0.1
    
    return activations, hidden_variables

def main():
    print("=== Simplified Encoded Reasoning Detection Demo ===")
    print("(Using synthetic data instead of a real model)\n")
    
    # Configuration
    num_samples = 1000
    input_dim = 256
    feature_dim = 64
    epochs = 10
    output_dir = "demo_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create synthetic activations with encoded patterns
    print("\nStep 1: Creating synthetic activations with encoded patterns...")
    activations, hidden_vars = create_synthetic_activations(
        num_samples=num_samples,
        input_dim=input_dim
    )
    
    # Extract the 1D variables for easier analysis
    simplified_hidden_vars = {
        'value': hidden_vars['value'],
        'binary_class': hidden_vars['binary_class'],
    }
    
    # Step 2: Train SAE on activations
    print("\nStep 2: Training SAE on synthetic activations...")
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        l1_coefficient=1e-3
    )
    
    # Train SAE
    metrics = sae.train_on_activations(activations, num_epochs=epochs)
    
    # Save SAE
    sae.save(os.path.join(output_dir, "demo_sae_model.pt"))
    
    # Step 3: Plot training metrics
    print("\nStep 3: Plotting training metrics...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot([m['loss'] for m in metrics])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot([m['recon_loss'] for m in metrics])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 3)
    plt.plot([m['sparsity'] for m in metrics])
    plt.title('Sparsity')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    
    # Step 4: Encode activations with SAE
    print("\nStep 4: Encoding activations with SAE...")
    feature_activations = sae.encode(activations)
    
    print(f"Feature activations shape: {feature_activations.shape}")
    
    # Step 5: Analyze features
    print("\nStep 5: Analyzing feature correlations with hidden variables...")
    
    # Create a directory for feature visualizations
    feature_dir = os.path.join(output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Find features that correlate with hidden variables
    for var_name in simplified_hidden_vars:
        print(f"  Analyzing correlation with {var_name}...")
        
        # Plot correlations
        top_features = plot_feature_correlations(
            feature_activations,
            simplified_hidden_vars,
            var_name,
            save_path=os.path.join(feature_dir, f"correlations_{var_name}.png")
        )
        
        # For top feature, plot prediction
        if len(top_features) > 0:
            top_feature_idx = top_features[0][0]
            top_corr = top_features[0][1]
            print(f"    Top feature {top_feature_idx} has correlation {top_corr:.3f}")
            
            is_classification = var_name == 'binary_class'
            plot_feature_prediction(
                feature_activations,
                simplified_hidden_vars,
                var_name,
                top_feature_idx,
                save_path=os.path.join(feature_dir, f"feature_{top_feature_idx}_vs_{var_name}.png"),
                is_classification=is_classification
            )
    
    # Step 6: Create a visualization of feature activations
    print("\nStep 6: Creating visualization of feature activations across samples...")
    
    # Get top features by activation variance
    feature_vars = np.var(feature_activations, axis=0)
    top_feature_indices = np.argsort(feature_vars)[::-1][:10]  # Top 10 features
    
    plt.figure(figsize=(12, 8))
    plt.imshow(feature_activations[:100, top_feature_indices].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Activation')
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Index')
    plt.title('Top Feature Activations Across Samples')
    plt.yticks(range(len(top_feature_indices)), top_feature_indices)
    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, "feature_activations_heatmap.png"))
    
    print("\nDemo completed successfully!")
    print(f"Results saved to {output_dir}")
    print("\nIn a real experiment, these features would represent patterns in the model's activations")
    print("that correlate with the hidden variables of the reasoning process.")

if __name__ == "__main__":
    main() 