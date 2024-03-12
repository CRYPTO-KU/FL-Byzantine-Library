from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np

class alie(_BaseByzantine):
    def __init__(self,n,m,z=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m
        self.alie_z_max = self.args.alie_z_max if self.args.alie_z_max is not None else self.z_max

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        pert = std * self.alie_z_max
        self.adv_momentum = mu - pert


    def local_step(self,batch):
        return None

    def train_(self, embd_momentum=None):
        return None
