#!/usr/bin/env python
"""
Analyze a pre-trained SAE model to detect encoded reasoning.
This script can be used to analyze a model that was trained with run_experiment.py.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import ActivationExtractor
from src.sae import SparseAutoencoder
from src.data import create_token_level_dataset
from src.visualization import plot_feature_activations, plot_feature_correlations, plot_feature_prediction

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a pre-trained SAE model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-1.8B", 
                        help="Language model to use (HuggingFace model ID)")
    parser.add_argument("--sae_model", type=str, required=True, 
                        help="Path to the pre-trained SAE model")
    parser.add_argument("--layer", type=int, default=None, 
                        help="Layer to extract activations from (default: middle layer)")
    parser.add_argument("--output_dir", type=str, default="analysis_results", 
                        help="Directory to save analysis results")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to analyze (if not provided, use default test prompts)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Analyzing pre-trained SAE model ===")
    print(f"Language model: {args.model}")
    print(f"SAE model: {args.sae_model}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Load the language model
    print("Step 1: Loading language model...")
    extractor = ActivationExtractor(model_name=args.model)
    
    # Use middle layer if not specified
    if args.layer is None:
        args.layer = extractor.model.cfg.n_layers // 2
        print(f"Using middle layer: {args.layer}")
    
    # Step 2: Load the SAE model
    print("Step 2: Loading SAE model...")
    sae = SparseAutoencoder.load(args.sae_model)
    print(f"Loaded SAE with {sae.feature_dim} features")
    
    # Step 3: Generate test data or use custom prompt
    if args.prompt:
        print("Step 3: Using custom prompt...")
        prompts = [args.prompt]
    else:
        print("Step 3: Generating test prompts...")
        # Create a few test prompts
        prompts = [
            "Calculate 25 + 37.",
            "Calculate 42 * 18.",
            "Calculate ((15 + 7) * 3 - 4) / 2.",
            "Find the next number in the sequence: 3, 7, 11, 15, 19, ...",
            "Alice has 24 apples. She gives 7 apples to Bob. How many apples does Alice have left?"
        ]
    
    # Step 4: Generate solutions and extract activations
    print("Step 4: Generating solutions and extracting activations...")
    solutions = []
    all_activations = []
    
    for prompt in tqdm(prompts):
        # Generate solution with chain of thought
        full_prompt = prompt + "\n" + "Let's solve this step by step:\n"
        solution, _ = extractor.generate_with_activations(
            full_prompt, 
            max_new_tokens=200,
            temperature=0.1
        )
        solutions.append(solution)
        
        # Extract activations from the solution text
        activations = extractor.get_activations_by_token(solution, args.layer)
        all_activations.append(activations.cpu().numpy())
    
    # Save the solutions
    with open(os.path.join(args.output_dir, "solutions.txt"), "w") as f:
        for i, (prompt, solution) in enumerate(zip(prompts, solutions)):
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Solution:\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Step 5: Create token-level dataset
    print("Step 5: Creating token-level dataset...")
    token_activations, token_hidden_vars, token_texts = create_token_level_dataset(
        solutions=solutions,
        activations_list=all_activations,
        tokenizer=extractor.model.tokenizer
    )
    
    # Step 6: Encode activations with SAE
    print("Step 6: Encoding activations with SAE...")
    token_feature_activations = sae.encode(token_activations)
    
    # Step 7: Analyze feature activations
    print("Step 7: Analyzing feature activations...")
    
    # Create a directory for feature visualizations
    feature_dir = os.path.join(args.output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Find features that correlate with token types
    for var_name in ['is_number', 'is_operator', 'is_equals']:
        if var_name in token_hidden_vars:
            # Plot correlations
            top_features = plot_feature_correlations(
                token_feature_activations,
                token_hidden_vars,
                var_name,
                save_path=os.path.join(feature_dir, f"correlations_{var_name}.png")
            )
            
            # For top features, plot activations and predictions
            for i, (feature_idx, corr) in enumerate(top_features[:5]):  # Top 5 features
                # Plot feature activations
                plot_feature_activations(
                    token_feature_activations,
                    token_texts,
                    feature_idx,
                    title=f"Feature {feature_idx} (corr with {var_name}: {corr:.3f})",
                    save_path=os.path.join(feature_dir, f"feature_{feature_idx}_tokens_{var_name}.png")
                )
                
                # Plot feature vs hidden variable
                plot_feature_prediction(
                    token_feature_activations,
                    token_hidden_vars,
                    var_name,
                    feature_idx,
                    save_path=os.path.join(feature_dir, f"feature_{feature_idx}_vs_{var_name}.png"),
                    is_classification=True
                )
    
    # Step 8: Analyze top tokens for each feature
    print("Step 8: Analyzing top tokens for each feature...")
    
    # Save top tokens for each feature
    top_tokens_file = os.path.join(args.output_dir, "top_tokens_per_feature.txt")
    with open(top_tokens_file, "w") as f:
        f.write("Top tokens for each SAE feature\n")
        f.write("=" * 50 + "\n\n")
        
        for feature_idx in range(min(50, sae.feature_dim)):  # Analyze first 50 features
            # Get activations for this feature
            feature_acts = token_feature_activations[:, feature_idx]
            
            # Get top tokens
            top_indices = np.argsort(feature_acts)[::-1][:20]
            top_tokens = [token_texts[i] for i in top_indices]
            top_activations = feature_acts[top_indices]
            
            f.write(f"Feature {feature_idx}:\n")
            for token, activation in zip(top_tokens, top_activations):
                if activation > 0:  # Only show tokens with non-zero activation
                    f.write(f"  {token}: {activation:.4f}\n")
            f.write("\n")
    
    # Step 9: Create a heatmap of feature activations for each solution
    print("Step 9: Creating feature activation heatmaps...")
    
    # Process each solution separately
    for i, solution in enumerate(solutions):
        # Tokenize the solution
        tokens = extractor.model.tokenizer.encode(solution)
        token_strings = [extractor.model.tokenizer.decode([t]) for t in tokens]
        
        # Get activations for this solution
        solution_acts = all_activations[i]
        
        # Encode with SAE
        feature_acts = sae.encode(solution_acts)
        
        # Create a heatmap of top features
        plt.figure(figsize=(15, 10))
        
        # Select top features by activation variance
        feature_vars = np.var(feature_acts, axis=0)
        top_feature_indices = np.argsort(feature_vars)[::-1][:20]  # Top 20 features
        
        # Select a subset of tokens to make the plot readable
        token_stride = max(1, len(token_strings) // 40)  # Show at most ~40 tokens
        selected_tokens = token_strings[::token_stride]
        selected_acts = feature_acts[::token_stride, :][:, top_feature_indices]
        
        # Create heatmap
        plt.imshow(selected_acts.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Feature Activation')
        plt.xlabel('Token Position')
        plt.ylabel('Feature Index')
        plt.title(f'Feature Activations for Solution {i+1}')
        
        # Set x-axis labels to token strings
        plt.xticks(range(len(selected_tokens)), selected_tokens, rotation=90)
        plt.yticks(range(len(top_feature_indices)), top_feature_indices)
        
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"heatmap_solution_{i+1}.png"))
        plt.close()
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to {args.output_dir}")
    print("\nKey findings to look for:")
    print("1. Features that strongly activate on specific token types")
    print("2. Patterns of feature activation across the solution")
    print("3. Features that might encode information not explicitly stated in the text")

if __name__ == "__main__":
    main() 