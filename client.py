import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *
from nn_classes import get_net

class client():
    def __init__(self,id,dataset,device,args,**kwargs):
        self.id = id
        self.model = get_net(args).to(device)
        self.args = args
        self.device = device
        self.loader = DataLoader(dataset,batch_size=args.bs,shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = None
        self.momentum2 = None
        self.local_steps = args.localIter
        self.lr = args.lr
        self.mean_loss = None
        self.omniscient = False
        self.relocate = False
        self.step = 0
        self.opt_step = self.get_optim(args)

    def local_step(self, batch):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.mean_loss = loss.item()
        self.opt_step()

    def train_(self, embd_momentum=None):
        iterator = iter(self.loader)
        flat_model = get_model_flattened(self.model, self.device)
        if embd_momentum is not None:
            self.momentum = torch.tensor(embd_momentum, device=self.device)
        elif self.momentum is None:
            self.momentum = torch.tensor(torch.zeros_like(flat_model, device=self.device))
        for i in range(self.local_steps):
            batch = iterator.__next__()
            self.local_step(batch)

    def get_grad(self):
        if self.args.opt == 'sgd':
            return torch.clone(self.momentum).detach()
        else: #for adams
            eps = 1e-08
            beta1, beta2 = self.args.betas
            new_moment = self.momentum.clone().detach() / (1- beta1**self.step)
            moment2 = self.momentum2.clone().detach() / (1- beta2 ** self.step)
            return new_moment / (torch.sqrt(moment2) + eps)

    def update_model(self, net_ps):
        pull_model(self.model, net_ps)

    def lr_step(self):
        self.lr *= .1

    def get_optim(self,args):
        if args.opt == 'sgd':
            return self.step_sgd
        elif args.opt == 'adam':
            return self.step_adam
        elif args.opt == 'adamw': # if local iter is 1, regularization has no impact
            return self.step_adamw
        else:
            raise NotImplementedError('Invalid optimiser name')

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
                    if not args.embd_momentum:
                        self.momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                    last_ind += length
                if args.nesterov:
                    d_p = d_p.add(buf, alpha=args.Lmomentum)
                else:
                    d_p = buf
                p.data.add_(d_p, alpha=-self.lr)

    def step_adam(self):
        last_ind = 0
        args = self.args
        eps = 1e-08
        self.step += 1
        if self.momentum2 is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.tensor(torch.zeros(model_size,device=self.device))
            self.momentum2 = torch.tensor(torch.zeros(model_size, device=self.device))
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas[0]) + d_p.mul(1-args.betas[0])
                v_t = buf2.mul(args.betas[1]) + torch.pow(d_p,2).mul(1-args.betas[1])
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - (args.betas[0]**self.step))
                vt_h = v_t.div(1 - (args.betas[1]**self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)


    def step_adamw(self):
        args = self.args
        last_ind = 0
        eps = 1e-08
        self.step += 1
        if self.momentum is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.tensor(torch.zeros(model_size, device=args.device))
            self.momentum2 = torch.tensor(torch.zeros(model_size, device=args.device))
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                p.data.add_(p.data, alpha=args.wd * -self.lr)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas(0)) + d_p.mul(1-args.betas(0))
                v_t = buf2.mul(args.betas(1)) + torch.pow(d_p,2).mul(1-args.betas(1))
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - torch.pow(args.betas(0), self.step))
                vt_h = v_t.div(1 - torch.pow(args.betas(1), self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)

