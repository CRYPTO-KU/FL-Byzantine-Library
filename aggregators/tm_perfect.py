import torch
from .base import _BaseAggregator
import numpy as np

class TM_perfect(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM_perfect, self).__init__()
        self.tm_bypassed = 0
        self.tm_locs = None
        self.rounds = 0
        self.last_aggregated = None

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        b_up = b//2
        b_down = b - b_up
        stacked = torch.stack(inputs[:-b], dim=0)
        largest, _ = torch.topk(stacked, b_up, 0)
        neg_smallest, _ = torch.topk(-stacked, b_down, 0)
        new_stacked = torch.cat([stacked, -largest,neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        #print(stacked.shape,new_stacked.shape, len(inputs), b, b_up, b_down)
        self.rounds +=1
        return new_stacked
    
if __name__ == "__main__": 
    # Test the trimmed mean aggregator
    def test_trimmed_mean():
        # Create sample inputs
        inputs = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([0.5, 1.5, 2.5]),
            torch.tensor([3.0, 4.0, 5.0]),
            torch.tensor([1.5, 2.5, 3.5])
        ]
        print(torch.sort(torch.stack(inputs,dim=0),dim=0)[0])
        
        # Initialize trimmed mean with b=1 (trim 1 from each end)
        tm = TM_perfect(b=1)
        
        # Test aggregation
        result = tm(inputs)
        print(f"Input tensors: {len(inputs)}")
        print(f"Trimmed mean result: {result}")
        print(f"TM bypassed ratio: {tm.tm_bypassed:.4f}")
        print(f"Rounds completed: {tm.rounds}")
        

    test_trimmed_mean()
