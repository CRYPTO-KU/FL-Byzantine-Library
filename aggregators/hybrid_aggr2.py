import torch
import numpy as np
from .base import _BaseAggregator
from .krum import Krum, multi_krum, pairwise_euclidean_distances
from .trimmed_mean import TM
from .clipping import Clipping
from .rfa import RFA, smoothed_weiszfeld 

class HybridAggregator(_BaseAggregator):
    """
    Hybrid aggregator that combines multiple Byzantine-robust aggregation methods.
    
    This aggregator performs a multi-stage filtering and aggregation process:
    1. Each specified aggregator performs elimination/sanitization
    2. The intersection of selected clients from all aggregators is computed
    3. Final aggregation is performed using the last specified aggregator
    
    Args:
        n (int): Total number of clients
        m (int): Number of malicious/Byzantine clients
        aggregator_list (list of str): List of aggregator names to use sequentially
    """

    def __init__(self, n, m, tau, aggregator_list):
        super(HybridAggregator, self).__init__()
        self.n = n
        self.m = m 
        self.tau = tau
        self.aggregator_list = aggregator_list
        self.last_aggregated = None
        
        # Track escape and impact ratios for individual aggregators
        self.krum_bypassed = 0
        self.krum_impact_ratio = 0
        self.tm_bypassed = 0
        self.tm_impact_ratio = 0
        self.rounds = 0

    def centered_clipping(self, inputs):
        ref = self.last_aggregated
        clipped_inputs = [self.clip(val - ref) + ref for val in inputs]
        return clipped_inputs
    
    def trimmed_mean(self, inputs):
        """
        Trimmed mean algorithm that discards the outlier m values coordinate-wise.
        
        Args:
            inputs: List of input tensors from clients
            
        Returns:
            List of tensors with outliers discarded (not aggregated)
        """
        if len(inputs) <= 2 * self.m:
            # If we don't have enough inputs to trim, return all
            return inputs
        
        # Stack inputs for easier computation
        stacked_inputs = torch.stack(inputs, dim=0)  # Shape: [num_clients, tensor_dims...]
        
        # Use exact same logic as TM.py
        b = self.m
        n = len(inputs)
        
        # Byzantine clients start at index (n - b)
        byz = n - b
        
        # Apply trimmed mean and track which indices get trimmed (exact same as TM.py)
        largest, largest_indices = torch.topk(stacked_inputs, b, 0)
        detect1 = byz <= largest_indices  # Check if Byzantine indices appear in largest
        neg_smallest, smallest_indices = torch.topk(-stacked_inputs, b, 0)
        detect2 = byz <= smallest_indices  # Check if Byzantine indices appear in smallest
        
        # Calculate bypass ratio (exact same as TM.py)
        self.tm_bypassed = (1 - ((detect1.sum() + detect2.sum()) / largest_indices.numel()).item())
        
        # Calculate TM impact ratio (exact same as TM.py)
        total_entries = len(inputs) * stacked_inputs.shape[1]
        total_trimmed_entries = largest_indices.numel() + smallest_indices.numel()
        total_retained_entries = total_entries - total_trimmed_entries
        
        malicious_entries_detected = (detect1.sum() + detect2.sum()).item()
        total_malicious_entries = b * stacked_inputs.shape[1]
        malicious_entries_retained = total_malicious_entries - malicious_entries_detected
        
        if total_retained_entries > 0:
            self.tm_impact_ratio = malicious_entries_retained / total_retained_entries
        else:
            self.tm_impact_ratio = 0
        
        # Sort along the client dimension (dim=0) for each coordinate
        sorted_values, sorted_indices = torch.sort(stacked_inputs, dim=0)
        
        # Remove m smallest and m largest values for each coordinate
        # Keep the middle values: from index m to -m
        trimmed_values = sorted_values[self.m:-self.m]  # Shape: [num_clients-2*m, tensor_dims...]
        
        # Convert back to list of individual tensors
        trimmed_inputs = [trimmed_values[i] for i in range(trimmed_values.shape[0])]
        return trimmed_inputs
    
    def multi_krum(self, inputs):
        """
        Multi-Krum aggregation using the original Krum implementation from krum.py.
        
        Args:
            inputs: List of input tensors from clients
            
        Returns:
            List of tensors with selected updates (not aggregated)
        """
        if len(inputs) <= self.m + 2:
            # If we don't have enough inputs to apply Multi-Krum, return all
            return inputs
        
        n = len(inputs)
        f = self.m  # Number of Byzantine clients
        
        # Check if we have enough clients for Krum
        if 2 * f + 2 > n:
            return inputs
        
        # Number of clients to select for multi-krum
        num_krum_aggregated = n - f - 2 if n > 10 else n - f
        
        # Use the original pairwise_euclidean_distances function from krum.py
        distances = pairwise_euclidean_distances(inputs)
        
        # Use the original multi_krum function from krum.py
        selected_indices = multi_krum(distances, n, f, num_krum_aggregated)
        
        # Track Byzantine clients that passed Krum (using EXACT same logic as krum.py)
        byzantine_clients = list(range(n - f, n))  # Last f clients are Byzantine
        krum_bypassed_count = 0
        for cl in byzantine_clients:
            if cl in selected_indices:
                krum_bypassed_count += 1
        
        if f > 0:
            self.krum_bypassed = krum_bypassed_count / f
            self.krum_impact_ratio = krum_bypassed_count / len(selected_indices) if len(selected_indices) > 0 else 0
        else:
            self.krum_bypassed = 0
            self.krum_impact_ratio = 0
        
        # Select the corresponding inputs
        selected_inputs = [inputs[i] for i in selected_indices]
        return selected_inputs

    def clip(self, v):
            v_norm = torch.norm(v)
            scale = min(1, self.tau / v_norm)
            return v * scale
    
    def tm_normed(self, inputs):
        """
        Get client selection from Trimmed Mean by identifying non-trimmed clients.
        This is an approximation since TM doesn't explicitly select clients.
        """
        # Get the aggregator instance to use its configured parameters
        tm_b = self.m
        
        # Compute distances from mean to identify potential outliers
        stacked = torch.stack(inputs, dim=0)
        mean_estimate = stacked.mean(dim=0)
        
        # Calculate distances from mean
        distances = torch.norm(stacked - mean_estimate.unsqueeze(0), dim=1)
        #print('Distances from mean:', distances)
        
        # Select clients that would not be trimmed (middle clients)
        sorted_indices = torch.argsort(distances)
        #print('Sorted indices:', sorted_indices)
        if tm_b is not None and tm_b > 0 and len(inputs) > 2 * tm_b:
            selected_indices = sorted_indices[tm_b:-tm_b]
        else:
            selected_indices = sorted_indices
        selected_inputs = [inputs[i] for i in selected_indices]

        return selected_inputs
    
    def tm_capped(self, inputs):
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

        return result

    def aggregation_sequence(self,inputs):
        aggr_methods = {'krum':self.multi_krum,
                        'cc': self.centered_clipping,
                        'tm': self.trimmed_mean,
                        'tm_normed': self.tm_normed,
                        'tmCapped': self.tm_capped}
        if self.last_aggregated is None:
            self.last_aggregated = torch.zeros_like(inputs[0])
        for i, aggr in enumerate(self.aggregator_list):
            self.n = len(inputs) 
            inputs = aggr_methods[aggr](inputs) # Update n to the current number of inputs
        return inputs
    
    def aggregate(self, inputs):
        aggr = sum(inputs) / len(inputs)
        # Final aggregation using the last aggregator in the list
        return aggr

    def __call__(self, inputs):
        """
        Perform hybrid aggregation by applying multiple aggregators sequentially.
        
        Args:
            inputs: List of input tensors from clients
            
        Returns:
            Aggregated tensor after applying all specified aggregators
        """
        final_inputs = self.aggregation_sequence(inputs)
        res = self.aggregate(final_inputs)
        self.last_aggregated = res
        self.rounds += 1
        return torch.clone(res).detach()
    
    def get_attack_stats(self) -> dict:
        """
        Get attack statistics for aggregators used in the hybrid.
        Only returns stats for aggregators that were actually used.
        """
        stats = {}
        
        # Check if krum was used in the aggregator list
        if 'krum' in self.aggregator_list:
            stats['Hybrid-Krum-Bypassed'] = self.krum_bypassed
            stats['Hybrid-Krum-Impact'] = self.krum_impact_ratio
        
        # Check if tm was used in the aggregator list
        if 'tm' in self.aggregator_list:
            stats['Hybrid-TM-Bypassed'] = self.tm_bypassed
            stats['Hybrid-TM-Impact'] = self.tm_impact_ratio
        
        return stats