import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for analyzing neural network activations.
    Detects learned features in the activation space.
    """
    def __init__(
        self,
        input_dim,
        feature_dim,
        l1_coefficient=1e-3,
        bias_decay=0.9,
        lr=1e-3,
        train_batch_size=4096,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.l1_coefficient = l1_coefficient
        self.bias_decay = bias_decay
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.device = device
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(input_dim, feature_dim, bias=True)
        self.decoder = nn.Linear(feature_dim, input_dim, bias=True)
        
        # Initialize with random weights
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
        
        # Initialize biases to small negative values to encourage sparsity
        self.encoder.bias.data.fill_(-1.0)
        
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        # Encoder (with ReLU to ensure non-negative activations)
        encoded = F.relu(self.encoder(x))
        # Decoder
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def train_step(self, batch):
        """Single training step on a batch of data"""
        self.optimizer.zero_grad()
        
        # Forward pass
        encoded, decoded = self(batch)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(decoded, batch)
        
        # L1 sparsity loss
        l1_loss = self.l1_coefficient * encoded.abs().mean()
        
        # Total loss
        loss = recon_loss + l1_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Decay bias to encourage sparsity over time
        with torch.no_grad():
            self.encoder.bias.data *= self.bias_decay
        
        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "dead_features": (encoded.abs().sum(0) == 0).sum().item(),
            "sparsity": (encoded == 0).float().mean().item()
        }
    
    def train_on_activations(self, activations, num_epochs=10):
        """Train the autoencoder on a dataset of activations"""
        # Convert to tensor if not already
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, dtype=torch.float32)
        
        activations = activations.to(self.device)
        dataset_size = activations.shape[0]
        
        metrics_history = []
        
        for epoch in range(num_epochs):
            # Shuffle data each epoch
            indices = torch.randperm(dataset_size)
            activations = activations[indices]
            
            # Train in batches
            epoch_metrics = []
            for i in range(0, dataset_size, self.train_batch_size):
                batch = activations[i:i + self.train_batch_size]
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
            
            # Average metrics for the epoch
            avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
            metrics_history.append(avg_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: " + 
                  f"Loss: {avg_metrics['loss']:.4f}, " + 
                  f"Recon Loss: {avg_metrics['recon_loss']:.4f}, " + 
                  f"L1 Loss: {avg_metrics['l1_loss']:.4f}, " + 
                  f"Sparsity: {avg_metrics['sparsity']:.4f}, " + 
                  f"Dead Features: {avg_metrics['dead_features']}/{self.feature_dim}")
        
        return metrics_history
    
    def encode(self, activations):
        """Encode activations to get feature activations"""
        with torch.no_grad():
            if not isinstance(activations, torch.Tensor):
                activations = torch.tensor(activations, dtype=torch.float32)
            activations = activations.to(self.device)
            return F.relu(self.encoder(activations)).cpu().numpy()
    
    def get_decoder_weights(self):
        """Get the decoder weights (features)"""
        return self.decoder.weight.data.cpu().numpy().T
    
    def save(self, path):
        """Save the autoencoder model"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'feature_dim': self.feature_dim,
                'l1_coefficient': self.l1_coefficient,
                'bias_decay': self.bias_decay,
            }
        }, path)
    
    @classmethod
    def load(cls, path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved autoencoder model"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            feature_dim=config['feature_dim'],
            l1_coefficient=config['l1_coefficient'],
            bias_decay=config['bias_decay'],
            device=device
        )
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        return model 