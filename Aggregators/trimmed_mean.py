import torch
from .base import _BaseAggregator
import numpy as np

class TM(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM, self).__init__()
        self.tm_bypassed = 0

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        byz = len(inputs) - b
        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        detect1 = byz <= _
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        detect2 = byz <= _

        self.tm_bypassed = (1-((detect1.sum() + detect2.sum()) / _.numel()).item())
        #print(np.mean(self.tm_bypassed))
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        return new_stacked
