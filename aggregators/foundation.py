import torch
import numpy as np
from .base import _BaseAggregator
from .trimmed_mean import TM
from .cm import CM  # Assuming CM is coordinate-wise median


class FoundationFL(_BaseAggregator):
    """
    FoundationFL: A novel defense mechanism against poisoning attacks.
    
    From "Do We Really Need to Design New Byzantine-robust Aggregation Rules?"
    
    FoundationFL enhances the robustness of well-established aggregation rules by:
    1. Finding dimension-wise maximum and minimum across all client updates
    2. Calculating scores for each client based on distance from these extremes
    3. Selecting the client update that deviates most from the extremes
    4. Using multiple copies of this selected update as synthetic updates
    5. Applying existing Byzantine-robust foundational aggregation rules (Trimmed-mean or Median)
       to combine clients' model updates with the synthetic ones
    
    Algorithm:
    - g_max^t[k] = max{g_1^t[k], ..., g_n^t[k]} for each dimension k
    - g_min^t[k] = min{g_1^t[k], ..., g_n^t[k]} for each dimension k  
    - s_i^t = min{||g_i^t - g_max^t||, ||g_i^t - g_min^t||} for each client i
    - i_* = argmax_i s_i^t (select client with highest score)
    - Generate m synthetic updates as copies of g_i*^t
    
    Parameters:
        base_aggregator: The foundational aggregation rule to use ('trimmed_mean' or 'median')
        b: Number of Byzantine clients (for trimmed mean)
        num_synthetic: Number of synthetic updates to generate (m in the paper)
    """
    
    def __init__(self, base_aggregator='trimmed_mean', b=1, num_synthetic=2):
        """
        Initialize FoundationFL
        
        Args:
            base_aggregator: The foundational aggregation rule to use ('trimmed_mean' or 'median')
            b: Number of Byzantine clients (for trimmed mean)
            num_synthetic: Number of synthetic updates to generate (m in the paper)
        """
        super(FoundationFL, self).__init__()
        
        self.base_aggregator = base_aggregator
        self.b = b
        self.num_synthetic = num_synthetic
        
        # Initialize the base aggregator
        if base_aggregator == 'trimmed_mean':
            # Adjust b for the total number of updates (clients + synthetic)
            self.aggregator = TM(b=b)
        elif base_aggregator == 'median':
            self.aggregator = CM()
        else:
            raise ValueError(f"Unsupported base aggregator: {base_aggregator}")
        
        # Statistics tracking
        self.rounds = 0
        self.synthetic_stats = {}
        
    def generate_synthetic_updates(self, client_updates):
        """
        Generate synthetic updates using FoundationFL algorithm
        
        The algorithm:
        1. Find dimension-wise max and min across all client updates
        2. Calculate scores for each client based on distance from extremes
        3. Select client with highest score (most deviation from extremes)
        4. Use multiple copies of this selected update as synthetic updates
        
        Args:
            client_updates: List of client update tensors
            
        Returns:
            List of synthetic update tensors
        """
        synthetic_updates = []
        
        if len(client_updates) == 0:
            return synthetic_updates
            
        # Convert to tensor for easier computation
        stacked_updates = torch.stack(client_updates, dim=0)  # Shape: [n_clients, d]
        
        # Step 1: Find dimension-wise maximum and minimum across all clients
        g_max_t, _ = torch.max(stacked_updates, dim=0)  # Shape: [d]
        g_min_t, _ = torch.min(stacked_updates, dim=0)  # Shape: [d]
        
        # Step 2: Calculate score for each client
        # s_i^t = min{||g_i^t - g_max^t||, ||g_i^t - g_min^t||}
        scores = []
        for i, g_i in enumerate(client_updates):
            dist_to_max = torch.norm(g_i - g_max_t)
            dist_to_min = torch.norm(g_i - g_min_t)
            score = torch.min(dist_to_max, dist_to_min)
            scores.append(score.item())
        
        # Step 3: Select client with highest score (most deviation from extremes)
        # i_* = argmax_i s_i^t
        i_star = torch.argmax(torch.tensor(scores)).item()
        selected_update = client_updates[i_star]
        
        # Step 4: Generate m synthetic updates as copies of the selected update
        # The paper uses multiple copies of the selected update
        for _ in range(self.num_synthetic):
            synthetic_updates.append(selected_update.clone())
        
        i_star_is_mal = i_star >= len(client_updates) - self.b
        # Store statistics for analysis
        self.synthetic_stats.update({
            'Malicious_selected': i_star_is_mal * 1
        })
        
        return synthetic_updates
    
    def __call__(self, inputs):
        """
        Aggregate client updates using FoundationFL
        
        Args:
            inputs: List of client update tensors
            
        Returns:
            Aggregated update tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        # Step 1: Generate synthetic updates
        synthetic_updates = self.generate_synthetic_updates(inputs)
        
        # Step 2: Combine client updates with synthetic updates
        all_updates = inputs + synthetic_updates
        
        # Step 3: Apply the base aggregator to the combined updates
        aggregated_update = self.aggregator(all_updates)
        
        # Update statistics
        self.rounds += 1
        self.synthetic_stats.update({
            'Malicious_selected': self.synthetic_stats.get('Malicious_selected', 0),
        })
        
        return aggregated_update
    
    def get_attack_stats(self):
        """Get attack statistics from the base aggregator and FoundationFL"""
        base_stats = self.aggregator.get_attack_stats() if hasattr(self.aggregator, 'get_attack_stats') else {}
        
        foundation_stats = {
            'Malicious_selected': self.synthetic_stats.get('Malicious_selected', 0),
        }
        
        # Combine stats
        if base_stats:
            foundation_stats.update(base_stats)
            
        return foundation_stats


class FoundationFL_TrimmedMean(FoundationFL):
    """FoundationFL with Trimmed Mean as base aggregator"""
    def __init__(self, b=1, num_synthetic=2):
        super().__init__(base_aggregator='trimmed_mean', b=b, num_synthetic=num_synthetic)


class FoundationFL_Median(FoundationFL):
    """FoundationFL with Coordinate-wise Median as base aggregator"""
    def __init__(self, num_synthetic=2):
        super().__init__(base_aggregator='median', num_synthetic=num_synthetic)


# Convenience factory function
def create_foundation_fl(base_aggregator='trimmed_mean', **kwargs):
    """
    Factory function to create FoundationFL instances
    
    Args:
        base_aggregator: 'trimmed_mean' or 'median'
        **kwargs: Additional parameters for FoundationFL
        
    Returns:
        FoundationFL instance
    """
    if base_aggregator == 'trimmed_mean':
        return FoundationFL_TrimmedMean(**kwargs)
    elif base_aggregator == 'median':
        return FoundationFL_Median(**kwargs)
    else:
        return FoundationFL(base_aggregator=base_aggregator, **kwargs)
