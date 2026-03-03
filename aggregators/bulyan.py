import torch
from .base import _BaseAggregator
from .krum import multi_krum, pairwise_euclidean_distances

def trimmed_mean(inputs, b):
    stacked = torch.stack(inputs)
    sorted_vals, sorted_inds = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[b:len(inputs)-b]
    return trimmed.mean(dim=0)

class Bulyan(_BaseAggregator):
    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m
        self.num_aggr = n - 2 * m
        self.krum_bypassed = 0
        self.tm_bypassed = 0
        self.tm_bypassed_cum = 0  # TM bypass relative to ALL Byzantine clients
        self.krum_impact_ratio = 0
        self.tm_impact_ratio = 0
        self.rounds = 0

    def __call__(self, inputs):
        n, m = self.n, self.m
        
        # Use the same Multi-Krum implementation as in krum.py
        distances = pairwise_euclidean_distances(inputs)
        selected_indices = multi_krum(distances, n, m,self.num_aggr) 
        
        # Track Byzantine clients that passed Krum (using EXACT same logic as krum.py)
        byzantine_clients = list(range(n - m, n))  # Last m clients
        krum_bypassed_count = 0
        for cl in byzantine_clients:
            if cl in selected_indices:
                krum_bypassed_count += 1
        if m > 0:
            self.krum_bypassed = krum_bypassed_count / m
        else:
            self.krum_bypassed = 0
        
        # Calculate Krum impact ratio: proportion of malicious clients in selected set
        if len(selected_indices) > 0:
            self.krum_impact_ratio = krum_bypassed_count / len(selected_indices)
        else:
            self.krum_impact_ratio = 0
        
        selected = [inputs[i] for i in selected_indices]
        b = m
        
        # Apply trimmed mean and track which indices get trimmed
        stacked = torch.stack(selected)
        b_trim = min(b, (len(selected) - 1) // 2)  # Ensure we don't trim everything
        
        # Use the same logic as trimmed_mean.py
        largest, largest_indices = torch.topk(stacked, b_trim, 0)
        neg_smallest, smallest_indices = torch.topk(-stacked, b_trim, 0)
        
        # Track which Byzantine clients get detected/trimmed
        # Byzantine clients are detected if their indices appear in topk results
        byz_clients_in_selected = [i for i, idx in enumerate(selected_indices) if idx in byzantine_clients]
        
        # Check if Byzantine client indices appear in the largest or smallest topk
        detected_byz = 0
        total_byz_positions = 0
        
        for byz_pos in byz_clients_in_selected:
            # Count how many times this Byzantine client appears in largest/smallest indices
            byz_detected_largest = (largest_indices == byz_pos).sum().item()
            byz_detected_smallest = (smallest_indices == byz_pos).sum().item()
            detected_byz += byz_detected_largest + byz_detected_smallest
            total_byz_positions += largest_indices.numel() + smallest_indices.numel()
        
        # Calculate TM bypass rate for Byzantine clients that passed Krum
        if len(byz_clients_in_selected) > 0 and total_byz_positions > 0:
            # Byzantine clients that are NOT detected by trimmed mean
            byz_not_detected = len(byz_clients_in_selected) - (detected_byz / (largest_indices.numel() + smallest_indices.numel()) * len(byz_clients_in_selected))
            self.tm_bypassed = max(0, byz_not_detected) / len(byz_clients_in_selected)
        else:
            self.tm_bypassed = 0
        
        # Calculate TM bypass rate relative to ALL Byzantine indices (b * dim)
        total_byz_indices = b * stacked.shape[1]  # b Byzantine clients * dimensions
        if total_byz_indices > 0:
            # Count Byzantine indices that passed both Krum and TM
            byz_indices_passed_krum = len(byz_clients_in_selected) * stacked.shape[1]  # Byzantine clients that passed Krum * dims
            byz_indices_passed_tm = byz_indices_passed_krum - detected_byz  # Subtract those detected by TM
            self.tm_bypassed_cum = max(0, byz_indices_passed_tm) / total_byz_indices
        else:
            self.tm_bypassed_cum = 0
        # TM bypass rate should be proportional to Byzantine clients that passed Krum
        if krum_bypassed_count > 0:
            # Already calculated above
            pass
        else:
            self.tm_bypassed = 0
        
        # Calculate TM impact ratio: proportion of malicious indices in non-trimmed (retained) indices
        total_retained_positions = (len(selected) - 2 * b_trim) * stacked.shape[1]  # Number of retained positions
        if total_retained_positions > 0:
            # Count malicious positions that were NOT trimmed (i.e., retained)
            total_malicious_positions = len(byz_clients_in_selected) * stacked.shape[1]  # Total malicious positions
            retained_malicious_positions = total_malicious_positions - detected_byz
            self.tm_impact_ratio = retained_malicious_positions / total_retained_positions
        else:
            self.tm_impact_ratio = 0
        
        self.rounds += 1
        # Compute the actual trimmed mean result
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(selected) - 2 * b_trim
        return new_stacked
    
    def get_attack_stats(self) -> dict:
        return {
            'Bulyan-Krum-Bypassed': self.krum_bypassed,
            'Bulyan-TM-Bypassed': self.tm_bypassed,
            'Bulyan-TM-Bypassed-Cumulative': self.tm_bypassed_cum,
            'Bulyan-Krum-Impact': self.krum_impact_ratio,
            'Bulyan-TM-Impact': self.tm_impact_ratio,
        }