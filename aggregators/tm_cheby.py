from .base import _BaseAggregator
import torch
import numpy as np

class ChebyshevAggregator(_BaseAggregator):
    """
    Chebyshev-based Trimmed Mean aggregator that uses Chebyshev inequality 
    to identify and bound outliers based on standard deviation distance.
    
    Instead of assuming a fixed number of malicious clients, this aggregator
    uses statistical bounds to identify outliers and clamps them to acceptable
    ranges based on the distribution of updates.
    
    Args:
        n (int): Total number of clients
        k_sigma (float): Number of standard deviations for Chebyshev bound (default: 1.0)
    """

    def __init__(self, n, k_sigma=1.0):
        super(ChebyshevAggregator, self).__init__()
        self.n = n
        self.k_sigma = k_sigma

    def __call__(self, inputs):
        """
        Apply Chebyshev-based trimmed mean aggregation.
        
        For each parameter index, compute mean and standard deviation,
        then use Chebyshev inequality to identify outliers and clamp them
        to acceptable bounds rather than eliminating them.
        
        Args:
            inputs: List of client update tensors
            
        Returns:
            Aggregated tensor with outliers bounded by Chebyshev inequality
        """
        if len(inputs) <= 1:
            return inputs[0] if inputs else torch.zeros_like(inputs[0])
        
        # Stack all client updates
        stacked = torch.stack(inputs, dim=0)  # Shape: [n_clients, ...]
        
        # Compute statistics for each parameter index
        mean_vals = torch.mean(stacked, dim=0)
        std_vals = torch.std(stacked, dim=0, unbiased=True)
        
        # Avoid division by zero for parameters with zero variance
        std_vals = torch.where(std_vals > 1e-8, std_vals, torch.ones_like(std_vals))
        
        # Apply Chebyshev-based outlier detection and clamping
        clamped_updates = self.chebyshev_clamp(stacked, mean_vals, std_vals)
        
        # Compute final aggregated result
        result = torch.mean(clamped_updates, dim=0)
        
        return result

    def chebyshev_clamp(self, stacked_updates, mean_vals, std_vals):
        """
        Apply Chebyshev inequality to clamp outliers for each parameter index.
        
        Chebyshev's inequality states that for any random variable X with mean μ and variance σ²,
        P(|X - μ| ≥ k*σ) ≤ 1/k²
        
        We use this to identify values that are more than k_sigma standard deviations
        away from the mean and clamp them to the acceptable bounds.
        
        Args:
            stacked_updates: Tensor of shape [n_clients, ...] with all client updates
            mean_vals: Mean values for each parameter index
            std_vals: Standard deviation values for each parameter index
            
        Returns:
            Clamped updates tensor
        """
        # Calculate bounds based on Chebyshev inequality
        upper_bound = mean_vals + self.k_sigma * std_vals
        lower_bound = mean_vals - self.k_sigma * std_vals
        
        # For each client update, clamp values that exceed Chebyshev bounds
        clamped_updates = torch.clamp(stacked_updates, min=lower_bound, max=upper_bound)
        
        return clamped_updates

    def get_outlier_statistics(self, inputs):
        """
        Get statistics about outliers detected by Chebyshev inequality.
        
        Returns:
            dict: Statistics including number of outliers, outlier probability, etc.
        """
        if len(inputs) <= 1:
            return {"outliers_detected": 0, "outlier_probability": 0.0}
        
        stacked = torch.stack(inputs, dim=0)
        mean_vals = torch.mean(stacked, dim=0)
        std_vals = torch.std(stacked, dim=0, unbiased=True)
        std_vals = torch.where(std_vals > 1e-8, std_vals, torch.ones_like(std_vals))
        
        # Calculate how many values exceed Chebyshev bounds
        deviations = torch.abs(stacked - mean_vals) / std_vals
        outliers_mask = deviations > self.k_sigma
        
        total_parameters = torch.numel(stacked)
        outliers_detected = torch.sum(outliers_mask).item()
        outlier_probability = outliers_detected / total_parameters
        
        # Theoretical maximum probability according to Chebyshev inequality
        theoretical_max_prob = 1.0 / (self.k_sigma ** 2)
        
        return {
            "outliers_detected": outliers_detected,
            "total_parameters": total_parameters,
            "outlier_probability": outlier_probability,
            "theoretical_max_probability": theoretical_max_prob,
            "k_sigma_used": self.k_sigma,
            "chebyshev_bound_satisfied": outlier_probability <= theoretical_max_prob
        }