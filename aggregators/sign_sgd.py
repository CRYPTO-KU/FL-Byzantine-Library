import torch
from .base import _BaseAggregator


class SignSGD(_BaseAggregator):
    def __init__(self,*args):
        super(SignSGD,self).__init__()

    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        signs = torch.sign(stacked)
        aggr = torch.sign(signs.sum(dim=0))
        return aggr