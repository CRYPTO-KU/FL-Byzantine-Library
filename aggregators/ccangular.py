import numpy as np
import torch
from .base import _BaseAggregator
import torch.nn.functional as F


class AngularClipping(_BaseAggregator):
    def __init__(self, tau, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(AngularClipping, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        sim_orig = F.cosine_similarity(v, self.momentum, dim=0)
        sim = (sim_orig.item() + 1) / 2
        #print(sim)
        sim = 1
        scale = min(1, self.tau * sim / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        for _ in range(self.n_iter):
            cc_cos = []
            mean_cos = []
            mean = sum(v for v in inputs) / len(inputs)
            for v in inputs:
                cc_cos.append(round(F.cosine_similarity(v,self.momentum,dim=0).item(),2))
                mean_cos.append(round(F.cosine_similarity(v, mean, dim=0).item(), 2))
            print(cc_cos)
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )

        return torch.clone(self.momentum).detach()



