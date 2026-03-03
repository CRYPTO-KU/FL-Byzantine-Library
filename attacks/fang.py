from .base import _BaseByzantine
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector

class fang(_BaseByzantine):
    """
    Fang et al. "Local Model Poisoning Attacks" (USENIX Security 2020)
    
    Implements tailored attacks for:
    1. Krum/Multi-Krum: Maximizes lambda in w' = w_re - lambda * s s.t. Krum selects w'
    2. Trimmed Mean/Median: Pushes malicious updates to the boundary of the benign distribution
       in the inverse direction of the global change.
    
    Both full-knowledge and partial-knowledge variants are supported.
    - Full knowledge:   attacker knows all benign updates (default w/ --MITM)
    - Partial knowledge: attacker only knows compromised workers' own updates
    """
    def __init__(self, n, m, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m

    def omniscient_callback(self, benign_gradients):
        """
        Dispatches to the appropriate attack based on aggregator and knowledge model.

        Args:
            benign_gradients: In MITM mode these are true benign gradients (full knowledge).
                              Otherwise they are EMA predictions from compromised data (partial knowledge).
        """
        # Stack gradients: (n, d)
        benign_stack = torch.stack(benign_gradients, dim=0)
        
        # Determine aggregator
        aggr = self.args.aggr
        if isinstance(aggr, list):
             aggr = aggr[0]
        
        # Determine knowledge model
        partial = not self.args.MITM
             
        if aggr in ['krum', 'multi_krum', 'mkrum']:
            if partial:
                self.adv_momentum = self.attack_krum_partial(benign_stack)
            else:
                self.adv_momentum = self.attack_krum(benign_stack)
        elif aggr in ['tm', 'trimmed_mean', 'median', 'md']:
            if partial:
                self.adv_momentum = self.attack_trimmed_mean_partial(benign_stack)
            else:
                self.adv_momentum = self.attack_trimmed_mean(benign_stack)
        else:
            # Fallback: use Krum attack logic
            if partial:
                self.adv_momentum = self.attack_krum_partial(benign_stack)
            else:
                self.adv_momentum = self.attack_krum(benign_stack)

    # ──────────────────────────────────────────────────────────────────────
    #  Full Knowledge Attacks
    # ──────────────────────────────────────────────────────────────────────

    def attack_krum(self, benign_updates):
        """
        Full-knowledge Krum attack.
        
        Goal: max lambda s.t. w' = w_re - lambda * s is selected by Krum
        among all benign + malicious updates.
        
        s = sign(sum(benign_updates))
        Binary search for lambda.
        """
        n_benign, d = benign_updates.shape
        n_attackers = self.m
        
        # Global benign direction
        benign_sum = torch.sum(benign_updates, dim=0)
        s = torch.sign(benign_sum)
        
        # Scale estimate for binary search upper bound
        avg_norm = torch.mean(torch.norm(benign_updates, dim=1)).item() if n_benign > 0 else 1.0
        lambda_max = avg_norm * 10.0
        lambda_min = 0.0
        lambda_best = 0.0
        
        for _ in range(20):
            lambda_curr = (lambda_min + lambda_max) / 2
            mal_update = -lambda_curr * s
            
            mal_stack = torch.stack([mal_update] * n_attackers, dim=0)
            all_updates = torch.cat((benign_updates, mal_stack), dim=0)
            
            if self._check_krum_selection(all_updates, n_attackers):
                lambda_best = lambda_curr
                lambda_min = lambda_curr
            else:
                lambda_max = lambda_curr
                
        return -lambda_best * s

    def attack_trimmed_mean(self, benign_updates):
        """
        Full-knowledge Trimmed Mean / Median attack.
        
        Pushes malicious values to the boundary of benign distribution
        in the inverse direction of the global gradient.
        """
        w_max, _ = torch.max(benign_updates, dim=0)
        w_min, _ = torch.min(benign_updates, dim=0)
        
        benign_sum = torch.sum(benign_updates, dim=0)
        s = torch.sign(benign_sum)
        
        b = 2.0
        mal_update = torch.zeros_like(s)
        
        # s = -1 → push positive (above w_max)
        mask_neg = (s < 0)
        target_neg = torch.where(w_max > 0, w_max * b, w_max / b)
        mal_update[mask_neg] = target_neg[mask_neg]
        
        # s = 1 → push negative (below w_min)
        mask_pos = (s > 0)
        target_pos = torch.where(w_min > 0, w_min / b, w_min * b)
        mal_update[mask_pos] = target_pos[mask_pos]
        
        # s = 0 → copy mean
        mask_zero = (s == 0)
        if mask_zero.any():
            mal_update[mask_zero] = torch.mean(benign_updates[:, mask_zero], dim=0)
        
        return mal_update

    # ──────────────────────────────────────────────────────────────────────
    #  Partial Knowledge Attacks
    # ──────────────────────────────────────────────────────────────────────

    def attack_krum_partial(self, compromised_updates):
        """
        Partial-knowledge Krum attack (Fang et al. §3.2).
        
        The attacker does NOT know benign workers' local models. Instead:
        - Estimate direction s_tilde from mean of compromised local models
        - Craft w'_1 = w_Re - lambda * s_tilde
        - Binary search for lambda where Krum selects w'_1 among only
          the crafted model + the c compromised before-attack models
        - If no solution, add more copies of w'_1
        
        Args:
            compromised_updates: (c, d) tensor of compromised workers' own gradients
        """
        n_comp, d = compromised_updates.shape
        
        # Estimate changing direction from compromised mean
        comp_mean = torch.mean(compromised_updates, dim=0)
        s_tilde = torch.sign(comp_mean)
        
        # Scale estimate
        avg_norm = torch.mean(torch.norm(compromised_updates, dim=1)).item() if n_comp > 0 else 1.0
        lambda_max = avg_norm * 10.0
        lambda_min = 0.0
        lambda_best = 0.0
        
        for _ in range(20):
            lambda_curr = (lambda_min + lambda_max) / 2
            mal_update = -lambda_curr * s_tilde
            
            # Pool: crafted w'_1 + c compromised before-attack models
            crafted = mal_update.unsqueeze(0)  # (1, d)
            pool = torch.cat((crafted, compromised_updates), dim=0)  # (c+1, d)
            
            # Check if Krum selects index 0 (the crafted one)
            if self._check_krum_selection_partial(pool):
                lambda_best = lambda_curr
                lambda_min = lambda_curr
            else:
                lambda_max = lambda_curr
        
        return -lambda_best * s_tilde

    def attack_trimmed_mean_partial(self, compromised_updates):
        """
        Partial-knowledge Trimmed Mean / Median attack (Fang et al. §3.3).
        
        The attacker estimates w_max and w_min using the mean (mu) and
        standard deviation (sigma) of the compromised workers' parameters:
        - When s_j = -1: sample from [mu_j + 3*sigma_j, mu_j + 4*sigma_j]
        - When s_j =  1: sample from [mu_j - 4*sigma_j, mu_j - 3*sigma_j]
        
        Args:
            compromised_updates: (c, d) tensor of compromised workers' own gradients
        """
        n_comp, d = compromised_updates.shape
        
        # Estimate direction from compromised mean
        comp_mean = torch.mean(compromised_updates, dim=0)  # (d,)
        s_tilde = torch.sign(comp_mean)
        
        # Compute per-coordinate stats
        mu = comp_mean
        sigma = torch.std(compromised_updates, dim=0) if n_comp > 1 else torch.ones_like(mu)
        
        mal_update = torch.zeros_like(mu)
        
        # s = -1 → push in positive direction: sample from [mu + 3*sigma, mu + 4*sigma]
        mask_neg = (s_tilde < 0)
        lo_neg = mu + 3.0 * sigma
        hi_neg = mu + 4.0 * sigma
        # Use midpoint for deterministic single-vector output
        mal_update[mask_neg] = ((lo_neg + hi_neg) / 2.0)[mask_neg]
        
        # s = 1 → push in negative direction: sample from [mu - 4*sigma, mu - 3*sigma]
        mask_pos = (s_tilde > 0)
        lo_pos = mu - 4.0 * sigma
        hi_pos = mu - 3.0 * sigma
        mal_update[mask_pos] = ((lo_pos + hi_pos) / 2.0)[mask_pos]
        
        # s = 0 → use compromised mean
        mask_zero = (s_tilde == 0)
        mal_update[mask_zero] = mu[mask_zero]
        
        return mal_update

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_krum_selection(self, all_updates, n_attackers):
        """
        Check if Krum selects any malicious update.
        Malicious updates are the LAST n_attackers indices.
        """
        n_total = all_updates.shape[0]
        k = max(n_total - n_attackers - 2, 1)
        
        scores = torch.zeros(n_total, device=self.device)
        for i in range(n_total):
            dists = torch.norm(all_updates - all_updates[i], dim=1)
            sorted_dists, _ = torch.sort(dists)
            scores[i] = torch.sum(sorted_dists[1:k+1])
            
        best_idx = torch.argmin(scores).item()
        return best_idx >= (n_total - n_attackers)

    def _check_krum_selection_partial(self, pool):
        """
        Check if Krum selects the crafted update (index 0) from the pool.
        Pool = [crafted_w'_1, comp_w_1, ..., comp_w_c]
        """
        n_total = pool.shape[0]
        # For partial knowledge: k = n_total - 2 (one crafted, rest compromised)
        k = max(n_total - 2, 1)
        
        scores = torch.zeros(n_total, device=self.device)
        for i in range(n_total):
            dists = torch.norm(pool - pool[i], dim=1)
            sorted_dists, _ = torch.sort(dists)
            scores[i] = torch.sum(sorted_dists[1:k+1])
            
        best_idx = torch.argmin(scores).item()
        return best_idx == 0  # crafted is at index 0

    def local_step(self, batch):
        pass

    def train_(self, embd_momentum=None):
        pass
