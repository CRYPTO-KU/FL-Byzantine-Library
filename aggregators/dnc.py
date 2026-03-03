"""
Divide-and-Conquer (DnC) Defense

Based on: "Manipulating the Byzantine: Optimizing Model Poisoning Attacks 
and Defenses for Federated Learning" (NDSS 2021)

Core idea: Performs dimensionality reduction via random subsampling, then uses 
SVD-based spectral methods to detect and remove outlier gradients. Multiple 
random subsampling iterations are used, and the final set of benign clients 
is the intersection of all "good" sets.

Algorithm:
    1. For niters iterations:
       a. Randomly sample b dimensions from the d-dimensional gradients
       b. Center the subsampled gradients by subtracting the mean
       c. Compute top right singular vector v of the centered matrix
       d. Score each client as s_i = (<g_i - μ, v>)^2
       e. Keep (n - c*m) clients with the lowest scores
    2. Final benign set = intersection of all "good" sets
    3. Aggregate = mean of gradients in the final benign set
"""

import torch
import numpy as np
from .base import _BaseAggregator


class DnC(_BaseAggregator):
    """
    Divide-and-Conquer robust aggregation.
    
    Args:
        num_clients: Total number of clients
        num_byzantine: Number of suspected Byzantine clients
        sub_dim: Dimensionality of random subsamples (b in the paper)
        num_iters: Number of random subsampling iterations (niters)
        filter_frac: Filtering fraction (c). Removes c*num_byzantine clients per iter
    """
    
    def __init__(self, num_clients, num_byzantine, sub_dim=10000, 
                 num_iters=5, filter_frac=1.0):
        super(DnC, self).__init__()
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.filter_frac = filter_frac
        self.rounds = 0
        self.detection_stats = {}
    
    def __call__(self, inputs):
        """
        Aggregate using DnC: random subsampling + SVD spectral filtering.
        
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
        d = inputs[0].shape[0]
        
        # Stack into matrix
        grads = torch.stack(inputs, dim=0)  # (n, d)
        
        # NaN/Inf check
        grads_np = grads.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)
        
        if n_invalid > 0:
            print(f"DnC: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out")
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                print("DnC: All gradients invalid, using simple mean")
                self.rounds += 1
                return grads.mean(dim=0)
            grads_np = grads_np[valid_indices]
            n = len(valid_indices)
        else:
            valid_indices = np.arange(n)
        
        # Number of clients to remove per iteration
        num_to_remove = int(self.filter_frac * self.num_byzantine)
        num_to_keep = max(1, n - num_to_remove)
        
        if num_to_remove <= 0 or n <= num_to_keep:
            # Nothing to filter
            self.rounds += 1
            return grads[valid_indices].mean(dim=0) if n_invalid > 0 else grads.mean(dim=0)
        
        # Subsample dimension (can't exceed d)
        b = min(self.sub_dim, d)
        
        # Collect good sets across iterations
        good_sets = []
        
        for _ in range(self.num_iters):
            # Step 1: Random dimension subsampling
            r = np.sort(np.random.choice(d, size=b, replace=False))
            sub_grads = grads_np[:, r]  # (n, b)
            
            # Step 2: Center the subsampled gradients
            mu = sub_grads.mean(axis=0)  # (b,)
            centered = sub_grads - mu  # (n, b)
            
            # Step 3: Top right singular vector via SVD
            # For n << b (typical in FL), it's more efficient to compute SVD
            # on the n×b matrix directly
            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                v = Vt[0]  # Top right singular vector, shape (b,)
            except np.linalg.LinAlgError:
                # SVD failed, skip this iteration
                good_sets.append(set(range(n)))
                continue
            
            # Step 4: Compute outlier scores: s_i = (<g_i - μ, v>)^2
            projections = centered @ v  # (n,)
            scores = projections ** 2  # (n,)
            
            # Step 5: Keep num_to_keep clients with lowest scores
            sorted_indices = np.argsort(scores)
            good_indices = set(sorted_indices[:num_to_keep].tolist())
            good_sets.append(good_indices)
        
        # Step 6: Final benign set = intersection of all good sets
        if len(good_sets) > 0:
            final_set = good_sets[0]
            for s in good_sets[1:]:
                final_set = final_set.intersection(s)
            benign_indices = sorted(final_set)
        else:
            benign_indices = list(range(n))
        
        # Fallback if intersection is empty
        if len(benign_indices) == 0:
            # Use the union instead — the intersection was too aggressive
            union_set = set()
            for s in good_sets:
                union_set = union_set.union(s)
            benign_indices = sorted(union_set)
        
        if len(benign_indices) == 0:
            benign_indices = list(range(n))
        
        # Map back to original indices if we filtered NaN clients
        original_benign = valid_indices[benign_indices]
        
        # Aggregate benign gradients
        aggregated = grads[original_benign].mean(dim=0)
        
        # Stats
        self.detection_stats = {
            'total_clients': len(inputs),
            'valid_clients': n,
            'benign_detected': len(benign_indices),
            'filtered_out': n - len(benign_indices),
            'benign_ratio': len(benign_indices) / n if n > 0 else 0.0,
            'sub_dim_used': b,
            'num_iters': self.num_iters,
        }
        self.rounds += 1
        
        return aggregated
    
    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'DnC-benign-ratio': self.detection_stats.get('benign_ratio', 1.0),
            'DnC-filtered': self.detection_stats.get('filtered_out', 0),
        }
