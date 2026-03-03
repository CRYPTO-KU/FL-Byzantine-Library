import torch
from .base import _BaseAggregator
import numpy as np

class TM(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM, self).__init__()
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
        largest, largest_indices = torch.topk(stacked, b, 0)
        detect1 = byz <= largest_indices
        neg_smallest, smallest_indices = torch.topk(-stacked, b, 0)
        detect2 = byz <= smallest_indices
        if self.tm_locs is None:
            self.tm_locs = (b - detect2.int().sum(0)) / b
        else:
            self.tm_locs += (b - detect2.int().sum(0)) / b

        self.tm_bypassed = (1-((detect1.sum() + detect2.sum()) / largest_indices.numel()).item())
        
        # Calculate TM impact ratio: proportion of malicious entries in retained entries
        # Total entries = number of clients * number of coordinates
        total_entries = len(inputs) * stacked.shape[1]
        
        # Total trimmed entries = entries from largest + entries from smallest
        total_trimmed_entries = largest_indices.numel() + smallest_indices.numel()
        
        # Total retained entries = total - trimmed
        total_retained_entries = total_entries - total_trimmed_entries
        
        # Malicious entries detected (trimmed) = detect1.sum() + detect2.sum()
        malicious_entries_detected = (detect1.sum() + detect2.sum()).item()
        
        # Total malicious entries = b clients * number of coordinates 
        total_malicious_entries = b * stacked.shape[1]
        
        # Malicious entries retained = total malicious - detected malicious
        malicious_entries_retained = total_malicious_entries - malicious_entries_detected
        
        if total_retained_entries > 0:
            self.tm_impact_ratio = malicious_entries_retained / total_retained_entries
        else:
            self.tm_impact_ratio = 0
        #print(np.mean(self.tm_bypassed))
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        self.rounds +=1
        return new_stacked
    
    def get_attack_stats(self) -> dict:
        attack_stats = {
            'TM-missed': self.tm_bypassed,
            'TM-Attacker-Impact': self.tm_impact_ratio
        }
        return attack_stats
