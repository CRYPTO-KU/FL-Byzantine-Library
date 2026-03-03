from .base import _BaseAggregator
import torch

class TM_capped(_BaseAggregator):
    """
    Trimmed mean with capping: replaces top/bottom m outliers with the max/min of the non-outlier (middle) values.
    
    Args:
        n (int): Total number of clients
        m (int): Number of malicious/Byzantine clients
    """
    
    def __init__(self, n, m):
        super(TM_capped, self).__init__()
        self.n = n
        self.m = m

    def __call__(self, inputs):
        """
        Trimmed mean with capping: replaces top/bottom m outliers with the max/min of the non-outlier (middle) values.
        """
        tm_b = self.m
        if len(inputs) <= 2 * tm_b:
            return inputs

        stacked = torch.stack(inputs, dim=0)
        sorted_vals, sorted_idx = torch.sort(stacked, dim=0)
        mid_vals = sorted_vals[tm_b:-tm_b]
        cap_max = mid_vals.max(dim=0).values
        cap_min = mid_vals.min(dim=0).values

        # Simple element-wise approach
        result = []
        for tensor in inputs:
            # Create a capped version of each tensor
            capped_tensor = torch.clone(tensor)
            
            # Apply min cap (don't go below the min of middle values)
            capped_tensor = torch.maximum(capped_tensor, cap_min)
            
            # Apply max cap (don't go above the max of middle values)
            capped_tensor = torch.minimum(capped_tensor, cap_max)
            
            result.append(capped_tensor)

        return sum(result) / len(result)
