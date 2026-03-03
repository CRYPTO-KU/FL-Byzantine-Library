"""
FLAME: Taming Backdoors in Federated Learning (USENIX Security 2022)

Based on: SkyMask repo (https://github.com/KoalaYan/SkyMask)

Core idea: Uses HDBSCAN clustering on pairwise cosine distances to identify the 
benign cluster. Applies gradient clipping using the median norm, then adds 
calibrated Gaussian noise for differential privacy.
"""

import torch
import torch.nn.functional as F
import numpy as np
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    try:
        import hdbscan
        HDBSCAN_obj = hdbscan.HDBSCAN
        HAS_HDBSCAN = True
    except ImportError:
        from sklearn.cluster import DBSCAN
        HAS_HDBSCAN = False
from .base import _BaseAggregator


class Flame(_BaseAggregator):
    """
    FLAME aggregation: HDBSCAN clustering + norm clipping + adaptive noise.
    
    Steps per round:
        1. Compute pairwise cosine distances between all client gradients
        2. HDBSCAN clustering (min_cluster_size = n//2 + 1) to find the benign cluster
        3. Compute clipping bound = median Euclidean norm of all updates
        4. Clip gradients in the benign cluster to the clipping bound
        5. Aggregate clipped gradients via simple average
        6. Add adaptive Gaussian noise for DP (optional)
    
    Args:
        epsilon: Differential privacy epsilon parameter (default: 3000)
        delta: Differential privacy delta parameter (default: 0.01)
        add_noise: Whether to add DP noise (default: True)
    """
    
    def __init__(self, epsilon=3000, delta=0.01, add_noise=True):
        super(Flame, self).__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.add_noise = add_noise
        self.rounds = 0
        self.detection_stats = {}
    
    def __call__(self, inputs):
        """
        Aggregate using FLAME: clustering + clipping + noise.
        
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
        
        # Stack all gradients
        grads = torch.stack(inputs, dim=0)  # (n, d)
        
        # Check for NaN/Inf and filter them out
        grads_np = grads.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)
        
        if n_invalid > 0:
            print(f"FLAME: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out")
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # All gradients are invalid - fallback to simple mean
                print("FLAME: All gradients invalid, using simple mean")
                self.rounds += 1
                return grads.mean(dim=0)
            
            # Use only valid gradients
            grads = grads[valid_indices]
            inputs_valid = [inputs[i] for i in valid_indices]
            n = len(inputs_valid)
        else:
            inputs_valid = inputs
        
        # Step 1: Compute pairwise cosine distances (vectorized)
        from sklearn.metrics.pairwise import cosine_distances
        grads_np = grads.detach().cpu().float().numpy()
        np_cos_dist = np.clip(cosine_distances(grads_np), 0, None)  # clamp negatives from float precision
        
        if HAS_HDBSCAN:
            # Use the optimized HDBSCAN (either from sklearn or standalone lib)
            # Use the local reference to the class we found in imports
            # (Note: we didn't alias it to HDBSCAN in imports to avoid name collision with the import attempt)
            ClustClass = HDBSCAN if 'HDBSCAN' in globals() else HDBSCAN_obj
            clusterer = ClustClass(
                metric='precomputed',
                min_samples=1,
                min_cluster_size=(n // 2) + 1,
                cluster_selection_epsilon=0.0,
                allow_single_cluster=True
            ).fit(np_cos_dist)
        else:
            # Fallback to DBSCAN if HDBSCAN is unavailable
            # Note: DBSCAN requires an epsilon. FLAME's epsilon is for DP, but here 
            # we need a clustering epsilon. 0.5 is a common default for cosine distance.
            clusterer = DBSCAN(
                metric='precomputed',
                min_samples=1,
                eps=0.5
            ).fit(np_cos_dist)
        
        
        cluster_labels = clusterer.labels_.tolist()
        
        # Find the main cluster (label 0, or largest non-noise cluster)
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        if len(unique_labels) == 0:
            # All labeled as noise — fallback to all clients
            benign_indices = list(range(n))
        else:
            # Find the largest cluster
            label_counts = {}
            for lb in unique_labels:
                label_counts[lb] = cluster_labels.count(lb)
            main_cluster = max(label_counts, key=label_counts.get)
            benign_indices = [i for i, lb in enumerate(cluster_labels) if lb == main_cluster]
        
        # Step 3: Compute clipping bound = median Euclidean norm
        euclid_norms = torch.norm(grads, p=2, dim=1)  # (n,)
        clipping_bound = torch.median(euclid_norms).item()
        
        # Step 4: Clip gradients in the benign cluster
        clipped_grads = []
        for i in benign_indices:
            grad_norm = euclid_norms[i].item()
            gamma = min(1.0, clipping_bound / (grad_norm + 1e-10))
            clipped_grads.append(inputs_valid[i] * gamma)
        
        # Step 5: Aggregate
        if len(clipped_grads) > 0:
            global_update = torch.stack(clipped_grads).mean(dim=0)
        else:
            global_update = grads.mean(dim=0)
        
        # Step 6: Add adaptive noise for DP (optional)
        if self.add_noise and self.epsilon > 0:
            std = clipping_bound * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = torch.normal(
                mean=0, std=std, 
                size=global_update.shape, 
                device=device,
                dtype=global_update.dtype
            )
            global_update = global_update + noise
        
        # Stats
        self.detection_stats = {
            'total_clients': n,
            'benign_detected': len(benign_indices),
            'filtered_out': n - len(benign_indices),
            'benign_ratio': len(benign_indices) / n,
            'clipping_bound': clipping_bound,
            'n_clusters': len(unique_labels) if 'unique_labels' in dir() else 0,
        }
        self.rounds += 1
        
        return global_update
    
    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'FLAME-benign-ratio': self.detection_stats.get('benign_ratio', 1.0),
            'FLAME-clip-bound': self.detection_stats.get('clipping_bound', 0.0),
        }
