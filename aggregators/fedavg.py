from .base import _BaseAggregator
import torch

class fedAVG(_BaseAggregator):
    def __init__(self,*args):
        super(fedAVG, self).__init__()
        self.momentum = None

    def __call__(self, inputs):
        self.momentum = sum(v for v in inputs) / len(inputs)
        return torch.clone(self.momentum).detach()
