import torch
import numpy as np
from .base import _BaseAggregator
from .krum import multi_krum, pairwise_euclidean_distances
"""
Implementing the Median-Krum aggregator from the paper 

“Median-krum: A joint distance-statistical
based byzantine-robust algorithm in federated learning
” 
"""

class MedKrum(_BaseAggregator):
    """
    Median Krum aggregator that combines Krum client selection with median aggregation.
    
    This aggregator:
    1. Uses Multi-Krum to select trusted clients
    2. Stacks the selected client updates
    3. Computes the coordinate-wise median for final aggregation
    """

    def __init__(self, n, f, m=None):
        super(MedKrum, self).__init__()
        self.n = n
        self.f = f
        self.m = m if m is not None else n - f - 2  # Default Multi-Krum selection size
        self.krum_bypassed = 0
        self.krum_impact_ratio = 0
        self.rounds = 0

    def __call__(self, inputs):
        # Step 1: Use Multi-Krum to select trusted clients
        distances = pairwise_euclidean_distances(inputs)
        selected_indices = multi_krum(distances, self.n, self.f, self.m)
        
        # Track Byzantine clients that passed Krum selection
        byzantine_clients = list(range(self.n - self.f, self.n))  # Last f clients are Byzantine
        krum_bypassed_count = 0
        for cl in byzantine_clients:
            if cl in selected_indices:
                krum_bypassed_count += 1
        
        if self.f > 0:
            self.krum_bypassed = krum_bypassed_count / self.f
            self.krum_impact_ratio = krum_bypassed_count / len(selected_indices) if len(selected_indices) > 0 else 0
        else:
            self.krum_bypassed = 0
            self.krum_impact_ratio = 0
        
        # Step 2: Stack selected client updates
        selected_inputs = [inputs[i] for i in selected_indices]
        
        if len(selected_inputs) == 0:
            # Fallback: return zero tensor if no clients selected
            return torch.zeros_like(inputs[0])
        
        if len(selected_inputs) == 1:
            # If only one client selected, return that client's update
            self.rounds += 1
            return selected_inputs[0]
        
        # Step 3: Compute coordinate-wise median
        stacked_updates = torch.stack(selected_inputs, dim=1)

        # Compute median along the client indexes (dim=1)
        median_result = torch.median(stacked_updates, dim=1).values
        
        self.rounds += 1
        return median_result
    
    def get_attack_stats(self) -> dict:
        return {
            'MedKrum-Bypassed': self.krum_bypassed,
            'MedKrum-Impact': self.krum_impact_ratio
        }
