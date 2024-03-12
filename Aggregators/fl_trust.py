import torch
from .base import _BaseAggregator
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FL_trust(_BaseAggregator):
    def __init__(self,root_dataset,model,args,device):
        super(FL_trust, self).__init__()
        self.loader = DataLoader(root_dataset,batch_size=64,shuffle=True)
        self.model = model
        self.args = args
        self.device = device
        self.momentum = torch.zeros(self.count_parameters(), device=self.device)


    def cos_sims(self,inputs):
        sims = [F.relu(F.cosine_similarity(m,self.momentum,dim=0)) for m in inputs]
        return sims

    def norm_inputs(self,inputs):
        n = torch.norm(self.momentum)
        norm_scales = [n / torch.norm(m) for m in inputs]
        return norm_scales

    def local_step(self, batch):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.step_sgd()

    def __call__(self, inputs):
        for data in self.loader:
            self.local_step(data)
            break
        if sum(self.cos_sims(inputs)) > 0:
            scale = 1 / sum(self.cos_sims(inputs))
        else:
            scale = 1
        aggr = [s * n * i for s,n,i in zip(self.cos_sims(inputs),self.norm_inputs(inputs),inputs)]
        return sum(aggr) * scale

    def step_sgd(self):
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)

                if self.momentum is None:
                    buf = torch.clone(d_p).detach()
                else:
                    length, dims = d_p.numel(), d_p.size()
                    buf = self.momentum[last_ind:last_ind + length].view(dims).detach()
                    buf.mul_(args.Lmomentum)
                    buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                    self.momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                    last_ind += length

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)



