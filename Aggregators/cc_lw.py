import torch
from .base import _BaseAggregator
from utils import get_model_layers


class cc_lw(_BaseAggregator):
    def __init__(self, args, n_iter=1):
        self.tau = args.tau
        self.n_iter = n_iter
        self.device = args.gpu_id
        super(cc_lw, self).__init__()
        self.momentum = None
        self.layers = get_model_layers(args)
        print(self.layers)

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        clips = torch.zeros_like(inputs[0], device=self.device)
        for _ in range(self.n_iter):
            for v in inputs:
                s = 0
                for layer in self.layers:
                    clips[s:layer].add_(self.clip(v[s:layer]-self.momentum[s:layer]))
                    s = layer
        self.momentum = self.momentum.add(clips,alpha=1/len(inputs))
        return torch.clone(self.momentum).detach()



