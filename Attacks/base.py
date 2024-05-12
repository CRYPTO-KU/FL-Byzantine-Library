import torch
from client import client
from utils import *
import numpy as np
import math
import torch.nn.functional as F
from math import radians
from torch.utils.data import DataLoader


class _BaseByzantine(client):
    """Base class of Byzantines (omniscient ones).
        Extension of this byzantines are capable of
        gathering data and communication between them.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.adv_momentum = None
        self.psuedo_momentum = None
        self.omniscient = True
        self.global_momentum = None
        self.debug_byz = False

    def update_dataloader(self,dataset):
        bs = int(self.args.bs * (1-self.args.traitor) / self.args.traitor)
        self.loader = DataLoader(dataset,batch_size=bs,shuffle=True)

    def omniscient_callback(self,benign_gradients):
        return NotImplementedError

    def get_global_m(self,m):
        self.global_momentum = m
    def adv_pred(self,batch,momentum):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.adv_step(momentum)

    def adv_step(self,momentum):
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)
                length, dims = d_p.numel(), d_p.size()
                buf = momentum[last_ind:last_ind + length].view(dims).detach()
                buf.mul_(args.Lmomentum)
                buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer

    def train_psuedo_moments(self):
        iterator = iter(self.loader)
        flat_model = get_model_flattened(self.model, self.device)
        if self.psuedo_momentum is None:
            self.psuedo_momentum = torch.tensor(torch.zeros_like(flat_model))
        self.psuedo_momentum .to(self.device)
        for i in range(self.local_steps):
            batch = iterator.__next__()
            self.adv_pred(batch,self.psuedo_momentum )
        self.psuedo_momentum.to('cpu')

    def get_grad(self):
        return torch.clone(self.momentum).detach()

    def get_benign_preds(self):
        return self.psuedo_momentum.clone().detach()

    def print_(self,*args): #debug print for one byzantine
        if self.debug_byz:
            print(*args)

    def get_angle(self,ref,pert):
        angle = math.degrees(math.acos(F.cosine_similarity(pert, ref, dim=0).item()))
        return angle
