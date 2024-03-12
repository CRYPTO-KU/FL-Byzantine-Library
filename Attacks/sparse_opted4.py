from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np


class sparse_opted4(_BaseByzantine):
    def __init__(self, n, m, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        self.z_1 = norm.ppf(cdf_value) if self.args.z_max is None else self.args.z_max
        self.n_good = n - m
        self.mask = mask
        self.delta_coeff = self.args.delta_coeff

    def omniscient_callback(self, benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        stack2 = torch.stack(benign_gradients, 0)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        z_small = self.z_small(stack2, mu)
        pert_small = std * z_small * (1 - self.mask)
        per_large = self.opted(stack2, mu, pert_small,z_small) * self.mask * std
        pert = pert_small + per_large
        self.adv_momentum = mu - pert

    def opted(self, all_updates, model_re, pert_small, z_small):

        deviation = torch.std(all_updates, 0)
        lamda = torch.Tensor([10]).float().to(self.device)
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
            pert_total = pert_small + (lamda * deviation * self.mask)
            mal_update = (model_re - pert_total)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        if lamda_succ > self.args.sparse_scale:
            lamda_succ = self.args.sparse_scale
        print('z2',lamda_succ)
        return lamda_succ

    def z_small(self, all_updates, model_re):

        deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10]).float().to(self.device)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores) * self.delta_coeff
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        if lamda_succ < self.z_1:
            lamda_succ = self.z_1
        #print('z1', lamda_succ)
        return lamda_succ

    def local_step(self, batch):
        return None

    def train_(self, embd_momentum=None):
        return None
