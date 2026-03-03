"""
Inspired from "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
Paper: https://arxiv.org/abs/1911.11815

Implements:
1. Krum attack variants
2. Trimmed Mean attack variants
3. Edge-case attacks
4. Min-Max variants
5. Adaptive and stealthy variants
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import _BaseByzantine
from utils import *


class KrumAttack(_BaseByzantine):
    """
    Targeted attack against Krum aggregation rule.
    
    This attack exploits the Krum selection mechanism by creating malicious
    updates that appear close to benign ones but bias the selection.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.z = z if z is not None else 1.0  # Attack strength parameter
        self.target_distance = eps if eps is not None else 0.1
        
    def omniscient_callback(self, benign_gradients):
        """Generate malicious updates targeting Krum selection."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        # Stack benign gradients
        benign_stack = torch.stack(benign_gradients, dim=0)
        n_benign = benign_stack.shape[0]
        
        if n_benign < 2:
            # Not enough benign gradients, use simple attack
            mu = torch.mean(benign_stack, dim=0)
            self.adv_momentum = -self.z * mu
            return
        
        # Find the gradient that would be selected by Krum
        # Compute pairwise distances
        distances = torch.zeros(n_benign)
        for i in range(n_benign):
            dist_sum = 0
            dist_list = []
            for j in range(n_benign):
                if i != j:
                    dist = torch.norm(benign_stack[i] - benign_stack[j]).item()
                    dist_list.append(dist)
            
            # Sort and take the closest n-m-2 distances (Multi-Krum parameter)
            dist_list.sort()
            n_closest = max(1, n_benign - self.m - 2)
            distances[i] = sum(dist_list[:n_closest])
        
        # Find the gradient with minimum score (Krum selection)
        krum_idx = torch.argmin(distances)
        krum_gradient = benign_stack[krum_idx]
        
        # Create malicious update close to Krum selection but biased
        attack_direction = -krum_gradient / (torch.norm(krum_gradient) + 1e-8)
        malicious_update = krum_gradient + self.target_distance * attack_direction
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class TrimmedMeanAttack(_BaseByzantine):
    """
    Targeted attack against Trimmed Mean aggregation rule.
    
    This attack exploits the trimming mechanism by creating updates
    that survive the trimming but bias the final mean.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.z = z if z is not None else 1.0  # Attack strength
        self.trim_ratio = eps if eps is not None else 0.1  # Expected trimming ratio
        
    def omniscient_callback(self, benign_gradients):
        """Generate malicious updates targeting Trimmed Mean."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        n_benign = benign_stack.shape[0]
        
        # Compute statistics of benign gradients
        mu = torch.mean(benign_stack, dim=0)
        
        # For trimmed mean, we need to position our malicious updates
        # so they survive trimming but bias the result
        
        # Compute coordinate-wise quantiles to understand trimming bounds
        benign_sorted, _ = torch.sort(benign_stack, dim=0)
        
        # Estimate trimming bounds (assuming symmetric trimming)
        trim_count = max(1, int(self.trim_ratio * (n_benign + self.m)))
        
        if n_benign > 2 * trim_count:
            # Use upper bound that won't be trimmed
            upper_bound = benign_sorted[n_benign - trim_count - 1]
            lower_bound = benign_sorted[trim_count]
            
            # Create malicious update at the boundary
            # Choose direction that maximizes bias while staying within bounds
            attack_direction = -torch.sign(mu)
            
            # Position at upper bound in attack direction
            malicious_update = upper_bound + self.z * attack_direction * torch.abs(upper_bound - lower_bound)
        else:
            # Fallback: simple bias attack
            malicious_update = mu - self.z * torch.norm(mu) * torch.sign(mu)
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class EdgeCaseAttack(_BaseByzantine):
    """
    Edge-case attack that targets specific vulnerabilities in aggregation rules.
    
    This attack creates updates that are at the edge of what aggregators consider
    as valid, exploiting the boundaries of robust aggregation.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.epsilon = eps if eps is not None else 0.1  # Edge parameter
        
    def omniscient_callback(self, benign_gradients):
        """Generate edge-case malicious updates."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        
        # Compute statistics
        mu = torch.mean(benign_stack, dim=0)
        
        # Find the benign update that is furthest from the mean
        distances = torch.norm(benign_stack - mu.unsqueeze(0), dim=1)
        furthest_idx = torch.argmax(distances)
        furthest_update = benign_stack[furthest_idx]
        
        # Create edge-case update slightly beyond the furthest benign update
        direction = furthest_update - mu
        direction_norm = torch.norm(direction)
        
        # Normalize direction to avoid extreme values
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            direction = torch.randn_like(direction)
            direction = direction / torch.norm(direction)
            
        edge_update = mu + (1 + self.epsilon) * direction_norm * direction
        
        self.adv_momentum = edge_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class LocalMinMaxAttack(_BaseByzantine):
    """
    Local model poisoning variant of Min-Max attack.
    
    Optimizes malicious updates to maximize the distance from benign updates
    while staying within detection bounds.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.scale_factor = z if z is not None else 1.0
        
    def omniscient_callback(self, benign_gradients):
        """Generate malicious updates using local min-max strategy."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        
        # Compute mean of benign updates
        mu = torch.mean(benign_stack, dim=0)
        
        # Compute pairwise distances between benign updates
        n_benign = benign_stack.shape[0]
        max_distance = 0
        
        for i in range(n_benign):
            for j in range(i + 1, n_benign):
                dist = torch.norm(benign_stack[i] - benign_stack[j])
                max_distance = max(max_distance, dist.item())
        
        # If all gradients are identical, use a small default distance
        if max_distance < 1e-8:
            max_distance = 0.1
        
        # Create malicious update at maximum allowable distance
        # Direction: opposite to the mean (normalized)
        direction = -mu
        direction_norm = torch.norm(direction)
        
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            # If mean is zero, use random direction
            direction = torch.randn_like(mu)
            direction = direction / torch.norm(direction)
            
        malicious_update = mu + self.scale_factor * max_distance * direction
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class AdaptiveKrumAttack(_BaseByzantine):
    """
    Adaptive Krum attack that adjusts its strategy based on aggregation history.
    
    Monitors the effectiveness of attacks against Krum and adapts accordingly.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.z_max = z if z is not None else 1.0
        self.epsilon = eps if eps is not None else 0.1
        self.adaptation_rate = 0.1
        self.current_scale = self.z_max
        self.success_history = []
        
    def omniscient_callback(self, benign_gradients):
        """Generate adaptive malicious updates targeting Krum."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        n_benign = benign_stack.shape[0]
        
        if n_benign < 2:
            mu = torch.mean(benign_stack, dim=0)
            self.adv_momentum = -self.current_scale * mu
            return
        
        # Simulate Krum selection on benign gradients
        distances = torch.zeros(n_benign)
        for i in range(n_benign):
            dist_list = []
            for j in range(n_benign):
                if i != j:
                    dist = torch.norm(benign_stack[i] - benign_stack[j]).item()
                    dist_list.append(dist)
            
            dist_list.sort()
            n_closest = max(1, n_benign - self.m - 2)
            distances[i] = sum(dist_list[:n_closest])
        
        krum_idx = torch.argmin(distances)
        krum_gradient = benign_stack[krum_idx]
        
        # Adaptive strategy: adjust attack strength based on gradient diversity
        gradient_diversity = torch.std(benign_stack).item()
        
        if gradient_diversity > self.epsilon:
            self.current_scale = min(self.z_max, self.current_scale + self.adaptation_rate)
        else:
            self.current_scale = max(0.1, self.current_scale - self.adaptation_rate)
        
        # Create adaptive malicious update
        attack_direction = -krum_gradient / (torch.norm(krum_gradient) + 1e-8)
        malicious_update = krum_gradient + self.current_scale * attack_direction
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class LocalTrimmedMeanAttack(_BaseByzantine):
    """
    Local model poisoning variant of Trimmed Mean attack.
    
    Targets the trimmed mean aggregation by positioning malicious updates
    strategically within the trimming bounds.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.alpha = z if z is not None else 0.5  # Attack intensity parameter
        self.epsilon = eps if eps is not None else 0.1  # Threshold parameter
        
    def omniscient_callback(self, benign_gradients):
        """Generate malicious updates using local trimmed mean strategy."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        n_benign = benign_stack.shape[0]
        
        # Compute coordinate-wise statistics
        mu = torch.mean(benign_stack, dim=0)
        
        # For each coordinate, find the trimming bounds
        benign_sorted, _ = torch.sort(benign_stack, dim=0)
        
        # Estimate how many values will be trimmed (Byzantine ratio)
        trim_count = max(1, self.m)
        
        if n_benign > 2 * trim_count:
            # Position at boundaries that won't be trimmed
            lower_safe = benign_sorted[trim_count]
            upper_safe = benign_sorted[n_benign - trim_count - 1]
            
            # Choose attack direction (opposite to mean)
            attack_direction = -torch.sign(mu)
            
            # Create malicious update at safe boundary
            range_size = upper_safe - lower_safe
            malicious_update = mu + self.alpha * range_size * attack_direction
        else:
            # Fallback: simple statistical attack
            std = torch.std(benign_stack, dim=0)
            attack_direction = -torch.sign(mu)
            malicious_update = mu + self.alpha * std * attack_direction
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class StealthyAttack(_BaseByzantine):
    """
    Stealthy attack that tries to remain undetected while causing damage.
    
    This attack generates updates that are statistically similar to benign ones
    but contain subtle poisoning.
    """
    
    def __init__(self, n, m, z=None, eps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.stealth_factor = z if z is not None else 0.05
        self.noise_threshold = eps if eps is not None else 0.01
        
    def omniscient_callback(self, benign_gradients):
        """Generate stealthy malicious updates."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        benign_stack = torch.stack(benign_gradients, dim=0)
        
        # Compute benign statistics
        mu = torch.mean(benign_stack, dim=0)
        std = torch.std(benign_stack, dim=0, unbiased=False)
        
        # Generate update that looks like a benign gradient but with subtle bias
        # Add small perturbation in direction opposite to mean
        stealth_noise = torch.randn_like(mu) * self.noise_threshold
        
        # Combine stealth factor with statistical camouflage
        direction = -mu / (torch.norm(mu) + 1e-8)
        malicious_component = self.stealth_factor * torch.norm(std) * direction
        
        # Final update: appears normal but contains malicious component
        malicious_update = mu + stealth_noise + malicious_component
        
        self.adv_momentum = malicious_update.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None
