from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np
from utils import count_parameters
from math import sin,cos,radians,sqrt
from torch.linalg import norm as tnorm
import math


def gaussian_ppf(cdf_value: float) -> float:
    """Return the Gaussian quantile (inverse CDF) for a given CDF value.

    Wrapper around scipy.stats.norm.ppf so the calculation is centralized.
    """
    return float(norm.ppf(cdf_value))

class sparse(_BaseByzantine):
    def __init__(self,n,m,z=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.omniscient = True
        self.adv_momentum = None
        self.mask = None
        self.sparse_scale = self.args.sparse_scale
        self.th_method = self.args.sparse_th
        self.attack_sign_rand = self.args.sparse_sign
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)

        # Compute default z_alt using the Gaussian quantile; helper exposes alternatives
        self.z_alt = gaussian_ppf(cdf_value)
        self.cos_sim = []
        self.attack_sign = None
        if z is not None:
            self.z_max = z
        else:
            self.z_max = self.z_alt

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        if self.attack_sign is None or self.attack_sign_rand=='inv_std':
            self.attack_sign = torch.ones_like(std)
        else:
            self.attack_sign = torch.sign(mu)

        pert_mult = self.mask * self.sparse_scale + (1 - self.mask) * self.z_max
        if self.th_method != None:
            sparse_attack_vals = std[self.mask.bool()]
            sparse_attack_fix = self.threshold(sparse_attack_vals)
            std[self.mask.bool()] = sparse_attack_fix
        final_pert = std * pert_mult * self.attack_sign
        attack = mu.add(final_pert, alpha=-1)
        #self.print_(get_angle(ud,final_pert),'angle pert')
        self.adv_momentum = attack


    def threshold(self,sparse_attack):
        """Fix the exploding values in extreme cases"""
        if self.th_method == 'iqr':
            q75, q25 = torch.quantile(sparse_attack, 0.75), torch.quantile(sparse_attack, 0.25)
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr  # Standard outlier threshold
            sparse_attack = torch.clamp(sparse_attack, max=upper_fence)

        elif self.th_method == 'z_score':
        # Method 2: Use Z-score based detection (adaptive)
        # Good for relatively large masks 
            target_mean = sparse_attack.mean()
            target_std = sparse_attack.std()
            z_threshold = 2.0  # Could be adaptive based on data size
            max_allowed = target_mean + z_threshold * target_std
            sparse_attack = torch.clamp(sparse_attack, max=max_allowed)

            # Method 3: Use gradient-based approach (most adaptive)
        elif self.th_method == 'gradient':
            # Find the "elbow" where the gradient changes significantly
            sorted_vals, _ = torch.sort(sparse_attack, descending=True)
            diffs = sorted_vals[:-1] - sorted_vals[1:]
            diff_ratios = diffs[:-1] / (diffs[1:] + 1e-8)
            elbow_idx = torch.argmax(diff_ratios).item() + 1
            adaptive_threshold = sorted_vals[elbow_idx].item()
            sparse_attack = torch.clamp(sparse_attack, max=adaptive_threshold)
        else:
            raise ValueError("Unknown thresholding method on mask locations: {}".format(method))
        return sparse_attack