"""
FedREDefense: Defending against Model Poisoning Attacks for Federated Learning
using Model Update Reconstruction Error (ICML 2024)

Based on: https://github.com/xyq7/FedREDefense

Core idea: genuine training updates lie in a low-dimensional subspace and can be
reconstructed well via PCA, while malicious updates have high reconstruction error.
Clients with reconstruction error above a threshold are filtered out.

OPTIMIZATION NOTE: Uses TruncatedSVD (randomized) instead of full PCA for 
high-dimensional gradients (millions of parameters). This is O(n*d*k) instead 
of O(min(n²d, nd²)), making it much faster for large models.
"""

import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .base import _BaseAggregator


class FedREDefense(_BaseAggregator):
    """
    FedREDefense aggregation via reconstruction error filtering.
    
    Steps:
        1. Stack client gradients into a matrix G ∈ R^{n×d}
        2. Fit TruncatedSVD (fast randomized PCA) with k components
        3. Reconstruct each gradient from low-rank projection
        4. Compute per-client normalized reconstruction error (vectorized)
        5. Flag clients above threshold as malicious
        6. Average remaining benign gradients
    
    Args:
        n_clients: Total number of clients
        n_components: Number of SVD components for reconstruction (default: 3)
        threshold: Normalized reconstruction error threshold (default: 0.6)
    """
    
    def __init__(self, n_clients, n_components=3, threshold=0.6):
        super(FedREDefense, self).__init__()
        self.n_clients = n_clients
        self.n_components = n_components
        self.threshold = threshold
        self.rounds = 0
        self.detection_stats = {}
    
    def __call__(self, inputs):
        """
        Aggregate using reconstruction error filtering.
        
        Args:
            inputs: List of flattened gradient tensors from clients
            
        Returns:
            Aggregated gradient tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        n = len(inputs)
        device = inputs[0].device
        inputs = [g.to(device) for g in inputs]
        
        # Stack into matrix
        grads = torch.stack(inputs, dim=0)  # (n, d)
        
        # Check for NaN/Inf and filter them out
        grads_np = grads.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)
        
        if n_invalid > 0:
            print(f"FedREDefense: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out")
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # All gradients are invalid - fallback to simple mean
                print("FedREDefense: All gradients invalid, using simple mean")
                self.rounds += 1
                return grads.mean(dim=0)
            
            # Use only valid gradients
            grads = grads[valid_indices]
            grads_np = grads_np[valid_indices]
            n = len(valid_indices)
        
        # Determine number of components (can't exceed n-1)
        k = min(self.n_components, n - 1)
        
        if k < 1 or n <= 2:
            # Not enough clients for meaningful decomposition, fallback to mean
            self.rounds += 1
            return grads.mean(dim=0)
        
        # grads_np already computed above (and filtered if needed)
        
        # Use TruncatedSVD (randomized, much faster than full PCA for high-d)
        # Complexity: O(n*d*k) instead of O(min(n²d, nd²))
        svd = TruncatedSVD(n_components=k, random_state=42)
        
        try:
            # Project to low-dim and reconstruct
            projected = svd.fit_transform(grads_np)        # (n, k)
            reconstructed = svd.inverse_transform(projected)  # (n, d)
            
            # Vectorized error computation (much faster than loop)
            reconstruction_errors = np.sum((grads_np - reconstructed) ** 2, axis=1)  # (n,)
            gradient_norms = np.sum(grads_np ** 2, axis=1) + 1e-10  # (n,)
            errors = reconstruction_errors / gradient_norms  # (n,)
            
        except Exception as e:
            # Fallback to simple mean if SVD fails (rare)
            print(f"FedREDefense SVD failed: {e}, using mean")
            self.rounds += 1
            return grads.mean(dim=0)
        
        # Filter clients with high reconstruction error
        benign_mask = errors < self.threshold
        benign_indices = np.where(benign_mask)[0].tolist()
        
        # Fallback: if all filtered out, use the ones with lowest error
        if len(benign_indices) == 0:
            # Keep the better half
            sorted_idx = np.argsort(errors)
            benign_indices = sorted_idx[:max(1, n // 2)].tolist()
        
        # Aggregate benign gradients
        benign_grads = grads[benign_indices]
        aggregated = benign_grads.mean(dim=0)
        
        # Store detection stats
        self.detection_stats = {
            'total_clients': n,
            'benign_detected': len(benign_indices),
            'filtered_out': n - len(benign_indices),
            'benign_ratio': len(benign_indices) / n,
            'mean_recon_error': float(np.mean(errors)),
            'max_recon_error': float(np.max(errors)),
        }
        self.rounds += 1
        
        return aggregated
    
    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'FedREDef-benign-ratio': self.detection_stats.get('benign_ratio', 1.0),
            'FedREDef-mean-recon-err': self.detection_stats.get('mean_recon_error', 0.0),
        }
