import torch
from .base import _BaseAggregator


class cc_tm(_BaseAggregator):
    def __init__(self, tau,b, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        self.b = b
        super(cc_tm, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def tm(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        return new_stacked

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
        tm_imputs = []

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )
            tm_imputs = [self.clip(v - self.momentum).add(self.momentum) for v in inputs]
        new_inputs = self.tm(tm_imputs)
        return new_inputs



