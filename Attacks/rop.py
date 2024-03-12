from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np
from utils import count_parameters
import math
from math import radians

class reloc(_BaseByzantine): ## ortho to ps & proj start rand
    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relocate = True
        self.first = True
        self.global_momentum = torch.zeros(count_parameters(self.model), device=self.device)
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.pi = self.args.pi


    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1) ## mean of the benign gradients
        ud = self.global_momentum.clone() ## reference point (global momentum)
        lamb = self.args.lamb # reference point for the angled attack
        pi = self.args.pi ## determines location of atack
        if self.first:
            ud = mu.clone() ## if first iteration, set the reference point to the mean of the benign momentums
            self.first = False

        attack_location = self.global_momentum * pi + mu.mul(1-pi) # attack location to perturb
        ud = ud.mul(lamb).add(mu,alpha=1-lamb) ## reference point
        pert = torch.ones_like(mu).to(self.device) # inital perturbation
        proj_pert = ud.mul((pert @ ud) / (ud @ ud)) # vector projection
        pert.sub_(proj_pert) # vector orthogonal to ud (rejecting the projection)
        n_ud = ud / ud.norm() # normalised reference point
        pert = pert / pert.norm() # normalised pert
        angle = self.args.angle # desired angle of the perturbation
        sin, cos = math.sin(radians(angle)), math.cos(radians(angle))
        pert = (pert * sin + n_ud * cos) # rotate the perturbation
        z = self.z_max / pert.norm() # scale the perturbation
        attack = attack_location.add(pert.mul(z)) # final attack added to desired location
        self.adv_momentum = attack # set as the momentum