#!/usr/bin/env python
"""
Run the encoded reasoning detection experiment with a more complex dataset
that might be more likely to induce encoded reasoning.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from src.models import ActivationExtractor
from src.complex_dataset import create_encoded_reasoning_dataset
from src.sae import SparseAutoencoder
from src.data import create_token_level_dataset
from src.visualization import create_feature_report

def parse_args():
    parser = argparse.ArgumentParser(description="Run encoded reasoning detection experiment with complex dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-1.8B", 
                        help="Model to use (HuggingFace model ID)")
    parser.add_argument("--num_problems", type=int, default=30, 
                        help="Number of problems to generate")
    parser.add_argument("--layer", type=int, default=None, 
                        help="Layer to extract activations from (default: middle layer)")
    parser.add_argument("--feature_dim", type=int, default=256, 
                        help="Number of features in the SAE")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of epochs to train the SAE")
    parser.add_argument("--output_dir", type=str, default="complex_results", 
                        help="Directory to save results")
    parser.add_argument("--l1_coefficient", type=float, default=1e-3, 
                        help="L1 coefficient for SAE sparsity")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Running encoded reasoning detection experiment with complex dataset ===")
    print(f"Model: {args.model}")
    print(f"Number of problems: {args.num_problems}")
    print(f"SAE feature dimension: {args.feature_dim}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Create dataset of complex problems
    print("Step 1: Creating dataset of complex problems...")
    problems = create_encoded_reasoning_dataset(num_problems=args.num_problems)
    
    # Step 2: Load model and extract activations
    print("Step 2: Loading model and extracting activations...")
    extractor = ActivationExtractor(model_name=args.model)
    
    # Use middle layer if not specified
    if args.layer is None:
        args.layer = extractor.model.cfg.n_layers // 2
        print(f"Using middle layer: {args.layer}")
    
    # Generate solutions and extract activations
    print("Generating solutions and extracting activations...")
    solutions = []
    all_activations = []
    
    for problem in tqdm(problems):
        # Generate solution with chain of thought
        prompt = problem + "\n" + "Let's solve this step by step:\n"
        solution, _ = extractor.generate_with_activations(
            prompt, 
            max_new_tokens=300,
            temperature=0.1  # Lower temperature for more deterministic reasoning
        )
        solutions.append(solution)
        
        # Extract activations from the solution text
        activations = extractor.get_activations_by_token(solution, args.layer)
        all_activations.append(activations.cpu().numpy())
    
    # Save some example solutions
    with open(os.path.join(args.output_dir, "example_solutions.txt"), "w") as f:
        for i, (problem, solution) in enumerate(zip(problems, solutions)):
            if i >= 10:  # Just save a few examples
                break
            f.write(f"Problem: {problem}\n\n")
            f.write(f"Solution:\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Step 3: Create token-level dataset
    print("Step 3: Creating token-level dataset...")
    token_activations, token_hidden_vars, token_texts = create_token_level_dataset(
        solutions=solutions,
        activations_list=all_activations,
        tokenizer=extractor.model.tokenizer
    )
    
    print(f"Token activations shape: {token_activations.shape}")
    
    # Step 4: Train SAE on activations
    print("Step 4: Training SAE on token activations...")
    input_dim = token_activations.shape[1]
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=args.feature_dim,
        l1_coefficient=args.l1_coefficient
    )
    
    # Train SAE
    start_time = time.time()
    metrics = sae.train_on_activations(token_activations, num_epochs=args.epochs)
    training_time = time.time() - start_time
    print(f"SAE training completed in {training_time:.2f} seconds")
    
    # Save SAE
    sae.save(os.path.join(args.output_dir, "sae_model.pt"))
    
    # Plot training metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot([m['loss'] for m in metrics])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(2, 2, 2)
    plt.plot([m['recon_loss'] for m in metrics])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(2, 2, 3)
    plt.plot([m['l1_loss'] for m in metrics])
    plt.title('L1 Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(2, 2, 4)
    plt.plot([m['sparsity'] for m in metrics])
    plt.title('Sparsity')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sae_training.png"))
    
    # Step 5: Encode activations with SAE
    print("Step 5: Encoding activations with SAE...")
    token_feature_activations = sae.encode(token_activations)
    
    print(f"Token feature activations shape: {token_feature_activations.shape}")
    
    # Step 6: Analyze features and create visualizations
    print("Step 6: Analyzing features and creating visualizations...")
    
    # Create token-level report
    token_report_dir = os.path.join(args.output_dir, "token_level")
    os.makedirs(token_report_dir, exist_ok=True)
    
    # Generate report
    print("Generating token-level report...")
    create_feature_report(
        feature_activations=token_feature_activations,
        hidden_vars=token_hidden_vars,
        token_texts=token_texts,
        output_dir=token_report_dir
    )
    
    # Step 7: Analyze top features
    print("Step 7: Analyzing top features...")
    
    # Save top tokens for each feature
    top_tokens_file = os.path.join(args.output_dir, "top_tokens_per_feature.txt")
    with open(top_tokens_file, "w") as f:
        f.write("Top tokens for each SAE feature\n")
        f.write("=" * 50 + "\n\n")
        
        for feature_idx in range(min(50, args.feature_dim)):  # Analyze first 50 features
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
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to {args.output_dir}")
    print("\nKey findings to look for:")
    print("1. Features that activate on specific token types (numbers, operators)")
    print("2. Features that might encode intermediate calculations")
    print("3. Features that activate on specific reasoning patterns")
    print("\nCheck the visualization reports and top tokens file for detailed analysis.")

if __name__ == "__main__":
    main() 