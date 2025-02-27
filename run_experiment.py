#!/usr/bin/env python
"""
Main experiment script for detecting encoded reasoning in CoT using SAEs.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from src.models import ActivationExtractor, create_arithmetic_dataset
from src.sae import SparseAutoencoder
from src.data import create_hidden_variable_dataset, create_token_level_dataset
from src.visualization import create_feature_report

def parse_args():
    parser = argparse.ArgumentParser(description="Run encoded reasoning detection experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-1.8B", 
                        help="Model to use (HuggingFace model ID)")
    parser.add_argument("--num_problems", type=int, default=20, 
                        help="Number of problems to generate")
    parser.add_argument("--layer", type=int, default=None, 
                        help="Layer to extract activations from (default: middle layer)")
    parser.add_argument("--feature_dim", type=int, default=128, 
                        help="Number of features in the SAE")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train the SAE")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--l1_coefficient", type=float, default=1e-3, 
                        help="L1 coefficient for SAE sparsity")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Running encoded reasoning detection experiment ===")
    print(f"Model: {args.model}")
    print(f"Number of problems: {args.num_problems}")
    print(f"SAE feature dimension: {args.feature_dim}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Create dataset of arithmetic problems
    print("Step 1: Creating dataset of arithmetic problems...")
    problems = create_arithmetic_dataset(num_problems=args.num_problems)
    
    # Step 2: Load model and extract activations
    print("Step 2: Loading model and extracting activations...")
    extractor = ActivationExtractor(model_name=args.model)
    
    # Use middle layer if not specified
    if args.layer is None:
        args.layer = extractor.model.cfg.n_layers // 2
        print(f"Using middle layer: {args.layer}")
    
    # Generate solutions and extract activations
    problems, solutions, activations = extractor.create_dataset_from_problems(
        problems=problems,
        layer=args.layer,
        max_new_tokens=200
    )
    
    print(f"Extracted activations with shape: {activations.shape}")
    
    # Save some example solutions
    with open(os.path.join(args.output_dir, "example_solutions.txt"), "w") as f:
        for i, (problem, solution) in enumerate(zip(problems, solutions)):
            if i >= 5:  # Just save a few examples
                break
            f.write(f"Problem: {problem}\n\n")
            f.write(f"Solution:\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Step 3: Create dataset with hidden variables
    print("Step 3: Creating dataset with hidden variables...")
    activations_data, hidden_vars = create_hidden_variable_dataset(
        problems=problems,
        solutions=solutions,
        activations=activations
    )
    
    # Also create token-level dataset
    print("Creating token-level dataset...")
    # Get activations for each solution separately
    solution_activations = []
    for solution in tqdm(solutions):
        acts = extractor.get_activations_by_token(solution, args.layer)
        solution_activations.append(acts.cpu().numpy())
    
    token_activations, token_hidden_vars, token_texts = create_token_level_dataset(
        solutions=solutions,
        activations_list=solution_activations,
        tokenizer=extractor.model.tokenizer
    )
    
    # Step 4: Train SAE on activations
    print("Step 4: Training SAE on activations...")
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=args.feature_dim,
        l1_coefficient=args.l1_coefficient
    )
    
    # Train SAE
    start_time = time.time()
    metrics = sae.train_on_activations(activations, num_epochs=args.epochs)
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
    feature_activations = sae.encode(activations)
    token_feature_activations = sae.encode(token_activations)
    
    print(f"Feature activations shape: {feature_activations.shape}")
    print(f"Token feature activations shape: {token_feature_activations.shape}")
    
    # Step 6: Analyze features and create visualizations
    print("Step 6: Analyzing features and creating visualizations...")
    
    # Create solution-level report
    solution_report_dir = os.path.join(args.output_dir, "solution_level")
    os.makedirs(solution_report_dir, exist_ok=True)
    
    # Create token-level report
    token_report_dir = os.path.join(args.output_dir, "token_level")
    os.makedirs(token_report_dir, exist_ok=True)
    
    # Generate reports
    print("Generating solution-level report...")
    create_feature_report(
        feature_activations=feature_activations,
        hidden_vars=hidden_vars,
        token_texts=["Solution " + str(i) for i in range(len(solutions))],
        output_dir=solution_report_dir
    )
    
    print("Generating token-level report...")
    create_feature_report(
        feature_activations=token_feature_activations,
        hidden_vars=token_hidden_vars,
        token_texts=token_texts,
        output_dir=token_report_dir
    )
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to {args.output_dir}")
    print("\nKey findings to look for:")
    print("1. Features that strongly correlate with hidden variables")
    print("2. Features that activate on specific token types (numbers, operators)")
    print("3. Features that encode information not explicitly stated in the CoT")
    print("\nCheck the visualization reports in the output directory for detailed analysis.")

if __name__ == "__main__":
    main() 