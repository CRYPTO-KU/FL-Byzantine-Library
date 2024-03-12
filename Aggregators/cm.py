import torch
from .base import _BaseAggregator


class CM(_BaseAggregator):
    def __init__(self,*args):
        super(CM,self).__init__()

    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        locs_1 = _ > 19
        values_lower, _ = (-stacked).median(dim=0)
        locs_2 = _ > 19
        #print(locs_1.sum() / _.numel(),  locs_2.sum() / _.numel())
        #return values_upper
        return (values_upper - values_lower) / 2