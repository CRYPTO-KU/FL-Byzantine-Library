import math
from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np


class sparse_optimized(_BaseByzantine):
    def __init__(self, n, m, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        self.z_1 = norm.ppf(cdf_value) if self.args.z_max is None else self.args.z_max
        self.n_good = n - m
        self.mask = None
        self.th_method = self.args.sparse_th
        self.opted_z_vals = []

    def omniscient_callback(self, benign_gradients):
        # Loop over good workers and accumulate their gradients
        self.mask = self.mask.to(self.device)
        stacked_gradients = torch.stack(benign_gradients, 1)
        stack2 = torch.stack(benign_gradients, 0)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        sparse_attack_vals = std[self.mask.bool()]
        sparse_attack_fix = self.threshold(sparse_attack_vals,self.th_method)
        std[self.mask.bool()] = sparse_attack_fix
        per_large = self.args.sparse_scale * self.mask * std
        z_1 = self.z_small(stack2, mu,per_large)
        pert_small = std * z_1 * (1 - self.mask)
        pert = pert_small + per_large
        self.adv_momentum = mu - pert
        self.opted_z_vals.append(z_1)

    def z_small(self, all_updates, model_re,pert_large):
        all_updates = all_updates.to(self.device)
        deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().to(self.device)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - (lamda * deviation * (1-self.mask)) - pert_large)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        #print('z1',lamda_succ)
        return lamda_succ
    
    def threshold(self, sparse_attack,method='iqr'):
        """Fix the exploding values in extreme cases"""

        if method == 'iqr':
            q75, q25 = torch.quantile(sparse_attack, 0.75), torch.quantile(sparse_attack, 0.25)
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr  # Standard outlier threshold
            sparse_attack = torch.clamp(sparse_attack, max=upper_fence)

        elif method == 'z_score':
        # Method 2: Use Z-score based detection (adaptive)
        # Good for relatively large masks 
            target_mean = sparse_attack.mean()
            target_std = sparse_attack.std()
            z_threshold = 2.0  # Could be adaptive based on data size
            max_allowed = target_mean + z_threshold * target_std
            sparse_attack = torch.clamp(sparse_attack, max=max_allowed)

            # Method 3: Use gradient-based approach (most adaptive)

        elif method == 'gradient':
            # Find the "elbow" where the gradient changes significantly
            sorted_vals, _ = torch.sort(sparse_attack, descending=True)
            diffs = sorted_vals[:-1] - sorted_vals[1:]
            diff_ratios = diffs[:-1] / (diffs[1:] + 1e-8)
            elbow_idx = torch.argmax(diff_ratios).item() + 1
            adaptive_threshold = sorted_vals[elbow_idx].item()
            sparse_attack = torch.clamp(sparse_attack, max=adaptive_threshold)
        return sparse_attack

    def local_step(self, batch):
        return None

    def train_(self, embd_momentum=None):
        return None
