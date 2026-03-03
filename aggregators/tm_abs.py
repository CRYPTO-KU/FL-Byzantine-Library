import torch
from .base import _BaseAggregator
import numpy as np

class TM_Abs(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM_Abs, self).__init__()
        self.tm_bypassed = 0
        self.tm_impact_ratio = 0
        self.tm_locs = None
        self.rounds = 0

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        
        byz = len(inputs) - b
        stacked = torch.stack(inputs, dim=0)
        
        # Get absolute values for sorting
        abs_stacked = torch.abs(stacked)
        
        # Sort by absolute values and get indices of largest absolute values to trim
        largest_abs, largest_abs_indices = torch.topk(abs_stacked, b, 0)
        
        # Check if Byzantine clients are detected in the largest absolute values
        detect1 = byz <= largest_abs_indices
        
        # For consistency with original TM, we also check smallest values
        # (though with absolute sorting, this might be redundant)
        smallest, smallest_indices = torch.topk(-abs_stacked, b, 0)
        detect2 = byz <= smallest_indices
        
        if self.tm_locs is None:
            self.tm_locs = (b - detect2.int().sum(0)) / b
        else:
            self.tm_locs += (b - detect2.int().sum(0)) / b

        self.tm_bypassed = (1-((detect1.sum() + detect2.sum()) / largest_abs_indices.numel()).item())
        
        # Calculate TM impact ratio: proportion of malicious entries in retained entries
        total_entries = len(inputs) * stacked.shape[1]
        total_trimmed_entries = largest_abs_indices.numel() + smallest_indices.numel()
        total_retained_entries = total_entries - total_trimmed_entries
        
        malicious_entries_detected = (detect1.sum() + detect2.sum()).item()
        total_malicious_entries = b * stacked.shape[1]
        malicious_entries_retained = total_malicious_entries - malicious_entries_detected
        
        if total_retained_entries > 0:
            self.tm_impact_ratio = malicious_entries_retained / total_retained_entries
        else:
            self.tm_impact_ratio = 0
        
        # Vectorized approach: sort by absolute values and trim
        abs_stacked = torch.abs(stacked)
        
        # Sort by absolute values for each coordinate (dimension)
        sorted_abs_vals, sorted_indices = torch.sort(abs_stacked, dim=0)
        
        # Keep middle values (trim b largest and b smallest absolute values)
        # For each coordinate, keep indices from b to -b
        middle_indices = sorted_indices[b:-b] if b > 0 else sorted_indices
        
        # Gather the original signed values using the middle indices
        # Create coordinate indices for advanced indexing
        coord_indices = torch.arange(stacked.shape[1]).unsqueeze(0).expand(middle_indices.shape[0], -1)
        
        # Use advanced indexing to get the trimmed values with original signs
        trimmed_values = stacked[middle_indices, coord_indices]
        
        # Compute mean of trimmed values
        result = trimmed_values.mean(dim=0)
        
        self.rounds += 1
        return result
    
    def get_attack_stats(self) -> dict:
        attack_stats = {
            'TM-Abs-missed': self.tm_bypassed,
            'TM-Abs-Attacker-Impact': self.tm_impact_ratio
        }
        return attack_stats
