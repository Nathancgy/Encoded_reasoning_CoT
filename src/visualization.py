import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from typing import Dict, List, Tuple, Optional, Union
import os

def plot_feature_activations(
    feature_activations: np.ndarray,
    token_texts: List[str],
    feature_idx: int,
    title: str = None,
    save_path: Optional[str] = None,
    top_k: int = 20
):
    """
    Plot the activations of a specific feature across tokens.
    
    Args:
        feature_activations: Feature activations matrix (n_tokens, n_features)
        token_texts: List of token texts
        feature_idx: Index of the feature to plot
        title: Plot title
        save_path: Path to save the plot
        top_k: Number of top tokens to display
    """
    # Get activations for the specified feature
    activations = feature_activations[:, feature_idx]
    
    # Sort tokens by activation value
    sorted_indices = np.argsort(activations)[::-1]
    top_indices = sorted_indices[:top_k]
    
    # Get top tokens and their activations
    top_tokens = [token_texts[i] for i in top_indices]
    top_activations = activations[top_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_tokens)), top_activations)
    plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right')
    plt.xlabel('Token')
    plt.ylabel('Activation')
    plt.title(title or f'Top {top_k} tokens for feature {feature_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_correlations(
    feature_activations: np.ndarray,
    hidden_vars: Dict[str, np.ndarray],
    var_name: str,
    title: str = None,
    save_path: Optional[str] = None,
    top_k: int = 10
):
    """
    Plot correlations between features and a hidden variable.
    
    Args:
        feature_activations: Feature activations matrix (n_tokens, n_features)
        hidden_vars: Dictionary of hidden variables
        var_name: Name of the hidden variable to correlate with
        title: Plot title
        save_path: Path to save the plot
        top_k: Number of top features to display
    """
    # Get the hidden variable
    hidden_var = hidden_vars[var_name]
    
    # Calculate correlation for each feature
    correlations = []
    for i in range(feature_activations.shape[1]):
        feature = feature_activations[:, i]
        # Use absolute correlation to find strongest relationships
        corr = np.abs(np.corrcoef(feature, hidden_var)[0, 1])
        if np.isnan(corr):
            corr = 0
        correlations.append((i, corr))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = correlations[:top_k]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    feature_indices = [f[0] for f in top_features]
    corr_values = [f[1] for f in top_features]
    
    plt.bar(range(len(feature_indices)), corr_values)
    plt.xticks(range(len(feature_indices)), feature_indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Absolute Correlation')
    plt.title(title or f'Top {top_k} features correlated with {var_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return top_features

def plot_feature_prediction(
    feature_activations: np.ndarray,
    hidden_vars: Dict[str, np.ndarray],
    var_name: str,
    feature_idx: int,
    title: str = None,
    save_path: Optional[str] = None,
    is_classification: bool = False
):
    """
    Plot the relationship between a feature and a hidden variable.
    
    Args:
        feature_activations: Feature activations matrix (n_tokens, n_features)
        hidden_vars: Dictionary of hidden variables
        var_name: Name of the hidden variable to predict
        feature_idx: Index of the feature to use for prediction
        title: Plot title
        save_path: Path to save the plot
        is_classification: Whether this is a classification task
    """
    # Get the feature and hidden variable
    feature = feature_activations[:, feature_idx]
    hidden_var = hidden_vars[var_name]
    
    # Remove NaN values
    valid_indices = ~np.isnan(hidden_var)
    feature = feature[valid_indices]
    hidden_var = hidden_var[valid_indices]
    
    if len(feature) == 0 or len(hidden_var) == 0:
        print(f"No valid data points for feature {feature_idx} and variable {var_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    if is_classification:
        # For classification tasks
        plt.scatter(feature, hidden_var, alpha=0.5)
        
        # Fit logistic regression
        X = feature.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(X, hidden_var)
        
        # Generate predictions for plotting
        x_range = np.linspace(min(feature), max(feature), 100)
        y_pred = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
        
        plt.plot(x_range, y_pred, 'r-', linewidth=2)
        
        # Calculate accuracy
        y_pred_class = model.predict(X)
        accuracy = accuracy_score(hidden_var, y_pred_class)
        plt.title(title or f'Feature {feature_idx} vs {var_name} (Accuracy: {accuracy:.3f})')
    else:
        # For regression tasks
        plt.scatter(feature, hidden_var, alpha=0.5)
        
        # Fit linear regression
        X = feature.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, hidden_var)
        
        # Generate predictions for plotting
        x_range = np.linspace(min(feature), max(feature), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        plt.plot(x_range, y_pred, 'r-', linewidth=2)
        
        # Calculate R²
        r2 = r2_score(hidden_var, model.predict(X))
        plt.title(title or f'Feature {feature_idx} vs {var_name} (R²: {r2:.3f})')
    
    plt.xlabel(f'Feature {feature_idx} Activation')
    plt.ylabel(var_name)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_embedding(
    feature_activations: np.ndarray,
    hidden_vars: Dict[str, np.ndarray],
    var_name: Optional[str] = None,
    method: str = 'pca',
    title: str = None,
    save_path: Optional[str] = None
):
    """
    Plot 2D embedding of feature activations, colored by a hidden variable.
    
    Args:
        feature_activations: Feature activations matrix (n_tokens, n_features)
        hidden_vars: Dictionary of hidden variables
        var_name: Name of the hidden variable to color by (optional)
        method: Dimensionality reduction method ('pca' or 'tsne')
        title: Plot title
        save_path: Path to save the plot
    """
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(feature_activations)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(feature_activations)
        method_name = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=(10, 8))
    
    if var_name is not None and var_name in hidden_vars:
        # Color by hidden variable
        hidden_var = hidden_vars[var_name]
        
        # Check if the variable is categorical
        unique_values = np.unique(hidden_var[~np.isnan(hidden_var)])
        if len(unique_values) <= 10:  # Assume categorical if few unique values
            # Create a scatter plot for each category
            for value in unique_values:
                mask = hidden_var == value
                plt.scatter(embedding[mask, 0], embedding[mask, 1], label=f'{var_name}={value}', alpha=0.7)
            plt.legend()
        else:
            # Continuous variable
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=hidden_var, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=var_name)
    else:
        # No coloring
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
    
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.title(title or f'{method_name} embedding of feature activations')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_feature_report(
    feature_activations: np.ndarray,
    hidden_vars: Dict[str, np.ndarray],
    token_texts: List[str],
    output_dir: str,
    top_k_features: int = 10,
    top_k_tokens: int = 20
):
    """
    Create a comprehensive report of feature analysis.
    
    Args:
        feature_activations: Feature activations matrix (n_tokens, n_features)
        hidden_vars: Dictionary of hidden variables
        token_texts: List of token texts
        output_dir: Directory to save the report
        top_k_features: Number of top features to analyze for each variable
        top_k_tokens: Number of top tokens to show for each feature
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze correlations with each hidden variable
    for var_name in hidden_vars:
        # Skip variables that are not 1D
        if hidden_vars[var_name].ndim > 1:
            continue
            
        # Find top correlated features
        top_features = plot_feature_correlations(
            feature_activations, 
            hidden_vars, 
            var_name,
            save_path=os.path.join(output_dir, f'correlations_{var_name}.png'),
            top_k=top_k_features
        )
        
        # For each top feature, create detailed plots
        for i, (feature_idx, corr) in enumerate(top_features):
            # Plot feature activations across tokens
            plot_feature_activations(
                feature_activations,
                token_texts,
                feature_idx,
                title=f'Feature {feature_idx} (corr with {var_name}: {corr:.3f})',
                save_path=os.path.join(output_dir, f'feature_{feature_idx}_tokens_{var_name}.png'),
                top_k=top_k_tokens
            )
            
            # Plot feature vs hidden variable
            is_classification = np.all(np.logical_or(
                np.isclose(hidden_vars[var_name], 0),
                np.isclose(hidden_vars[var_name], 1)
            ))
            
            plot_feature_prediction(
                feature_activations,
                hidden_vars,
                var_name,
                feature_idx,
                save_path=os.path.join(output_dir, f'feature_{feature_idx}_vs_{var_name}.png'),
                is_classification=is_classification
            )
    
    # Create embeddings
    plot_feature_embedding(
        feature_activations,
        hidden_vars,
        method='pca',
        save_path=os.path.join(output_dir, 'pca_embedding.png')
    )
    
    # Create embeddings for each variable
    for var_name in hidden_vars:
        # Skip variables that are not 1D
        if hidden_vars[var_name].ndim > 1:
            continue
            
        plot_feature_embedding(
            feature_activations,
            hidden_vars,
            var_name=var_name,
            method='pca',
            save_path=os.path.join(output_dir, f'pca_embedding_{var_name}.png')
        ) 