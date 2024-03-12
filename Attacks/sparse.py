from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np
from utils import count_parameters
from math import sin,cos,radians,sqrt
from torch.linalg import norm as tnorm
import math

class Sparse(_BaseByzantine):
    def __init__(self,n,m,mask,z=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.adv_momentum = torch.zeros(count_parameters(self.model), device=self.device)
        self.sparse_mask = mask
        self.sparse_scale = self.args.sparse_scale
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        self.z_alt = norm.ppf(cdf_value)
        if z is not None:
            self.z_max = z
        else:
            self.z_max = self.z_alt

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        pert_multer = self.sparse_mask * self.sparse_scale + (1 - self.sparse_mask) * self.z_max
        final_pert = std * pert_multer
        attack = mu.add(final_pert,alpha=-1)
        self.adv_momentum = attack