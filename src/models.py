import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm

class ActivationExtractor:
    """
    Class to extract activations from a language model during inference.
    """
    def __init__(
        self, 
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hook_points: Optional[List[str]] = None
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model_name: Name of the model to load (HuggingFace model ID)
            device: Device to run the model on
            hook_points: List of hook points to extract activations from. 
                         If None, use default hooks.
        """
        self.device = device
        # Use TransformerLens to load the model
        print(f"Loading model {model_name}...")
        try:
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=device
            )
            print(f"Model loaded with {self.model.cfg.n_layers} layers.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with additional parameters...")
            try:
                # Try with additional parameters that might help with certain models
                self.model = HookedTransformer.from_pretrained(
                    model_name,
                    device=device,
                    dtype=torch.float32,  # Use float32 to avoid precision issues
                    fold_ln=False,  # Don't fold LayerNorm parameters
                    center_writing_weights=False,  # Don't center weights
                    center_unembed=False  # Don't center unembed
                )
                print(f"Model loaded with {self.model.cfg.n_layers} layers.")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Please check your internet connection or try a different model.")
                raise
        
        # Set default hook points if not specified
        if hook_points is None:
            # Default to getting activations from residual stream and MLP outputs
            self.hook_points = [
                f"blocks.{i}.mlp.output" for i in range(self.model.cfg.n_layers)
            ]
        else:
            self.hook_points = hook_points
            
        self.hooks = None
        self.current_activations = {}
        
    def _hook_fn(self, hook_point):
        """Create a hook function for a specific hook point"""
        def hook(act, hook):
            self.current_activations[hook_point] = act.detach().cpu()
        return hook
    
    def setup_hooks(self):
        """Set up hooks for activation extraction"""
        print(f"Setting up hooks for {len(self.hook_points)} hook points...")
        hook_dict = {
            hook_point: self._hook_fn(hook_point) for hook_point in self.hook_points
        }
        self.hooks = self.model.hook_points(
            hook_dict, 
            detach=True, 
            is_permanent=False
        )
    
    def remove_hooks(self):
        """Remove all hooks"""
        if self.hooks is not None:
            self.hooks.remove()
            self.hooks = None
            
    def generate_with_activations(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.7,
        extract_layer: Optional[int] = None
    ) -> Tuple[str, Dict[str, torch.Tensor]]:
        """
        Generate text from a prompt and extract activations.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            extract_layer: If specified, only extract activations from this layer's MLP
        
        Returns:
            Tuple of (generated text, activations dict)
        """
        # Clear previous activations
        self.current_activations = {}
        
        # Set up hooks for specific layer if requested
        if extract_layer is not None:
            old_hook_points = self.hook_points
            self.hook_points = [f"blocks.{extract_layer}.mlp.output"]
        
        # Set up hooks
        self.setup_hooks()
        
        # Generate text
        with torch.no_grad():
            tokens = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
            
            # Get generated text
            generated_text = self.model.tokenizer.decode(output_tokens[0])
            
        # Remove hooks
        self.remove_hooks()
        
        # Restore original hook points if needed
        if extract_layer is not None:
            self.hook_points = old_hook_points
            
        return generated_text, self.current_activations
    
    def get_activations_by_token(
        self, 
        text: str, 
        layer: int
    ) -> torch.Tensor:
        """
        Get activations for each token in the text.
        
        Args:
            text: Text to get activations for
            layer: Layer to extract activations from
            
        Returns:
            Tensor of shape [num_tokens, hidden_dim]
        """
        # Clear previous activations
        self.current_activations = {}
        
        # Set hook for this layer's output
        old_hook_points = self.hook_points
        self.hook_points = [f"blocks.{layer}.mlp.output"]
        self.setup_hooks()
        
        # Get activations
        with torch.no_grad():
            tokens = self.model.tokenizer.encode(text, return_tensors="pt").to(self.device)
            _ = self.model(tokens)
            
        # Get the activations
        act_key = f"blocks.{layer}.mlp.output"
        activations = self.current_activations[act_key][0]  # Remove batch dimension
        
        # Remove hooks
        self.remove_hooks()
        
        # Restore original hook points
        self.hook_points = old_hook_points
        
        return activations
    
    def create_dataset_from_problems(
        self,
        problems: List[str],
        solution_gen_prefix: str = "Let's solve this step by step:\n",
        max_new_tokens: int = 200,
        layer: int = None,
        progress_bar: bool = True
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Create a dataset of activations from a list of problems.
        
        Args:
            problems: List of problem statements
            solution_gen_prefix: Prefix to add to problems to generate solutions
            max_new_tokens: Maximum number of tokens to generate for solutions
            layer: Layer to extract activations from (if None, use middle layer)
            progress_bar: Whether to show a progress bar
        
        Returns:
            Tuple of (problems, solutions, activations)
        """
        if layer is None:
            # Default to middle layer
            layer = self.model.cfg.n_layers // 2
            
        solutions = []
        all_activations = []
        
        problems_iter = tqdm(problems) if progress_bar else problems
        
        for problem in problems_iter:
            # Generate solution with chain of thought
            prompt = problem + "\n" + solution_gen_prefix
            solution, _ = self.generate_with_activations(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=0.1  # Lower temperature for more deterministic reasoning
            )
            solutions.append(solution)
            
            # Extract activations from the solution text
            activations = self.get_activations_by_token(solution, layer)
            all_activations.append(activations.cpu().numpy())
        
        # Stack activations
        activations_dataset = np.vstack([act for acts in all_activations for act in acts])
        
        return problems, solutions, activations_dataset

# Helper function to create simple arithmetic problems
def create_arithmetic_dataset(num_problems: int = 100, operations: List[str] = ["+", "-", "*"]) -> List[str]:
    """Create a dataset of arithmetic problems"""
    problems = []
    for _ in range(num_problems):
        op = np.random.choice(operations)
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        problem = f"Calculate {a} {op} {b}."
        problems.append(problem)
    return problems 