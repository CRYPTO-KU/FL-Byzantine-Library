from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np

class lasa_attack(_BaseByzantine):
    def __init__(self,n,m,layer_inds,z=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.z_max = z if z is not None else norm.ppf(1 - m / (2 * n))
        self.k = 0.3
        self.m = m
        self.layer_inds = layer_inds
        self.sign_mask = None
        self.attack_large = self.args.sparse_scale
        self.bottomk = self.args.lasa_attack_k1
        self.bottomk2 = self.args.lasa_attack_k2 # threshold for small perturbations, should be larger than bottomk or 0

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 0)
        mean_sign = torch.sign(torch.mean(stacked_gradients, dim=0))
        std = torch.std(stacked_gradients)
        bottom_inds = [val.abs().topk(int(val.numel()*self.k),largest=False)[1] for val in stacked_gradients]
        bottom_locs = torch.zeros_like(stacked_gradients[0])
        for inds in bottom_inds:
            bottom_locs[inds] += 1
        attack = torch.zeros_like(stacked_gradients[0])
        #attack_locs = (bottom_locs < self.m//2).int().nonzero().squeeze()
        attack_locs_small = torch.topk(bottom_locs, int(bottom_locs.numel() * self.bottomk2), largest=False)[1]
        attack[attack_locs_small] = std * self.z_max
        attack_locs = torch.topk(bottom_locs, int(bottom_locs.numel() * self.bottomk), largest=False)[1]
        attack[attack_locs] = std * self.attack_large
        sign = mean_sign
        #sign = self.sign_mask
        attack_locs_layered = self.split_to_layers_single(attack * sign)
        layer_grads = self.split_to_layers(stacked_gradients)
        layer_norms = [torch.norm(grad, dim=1) for grad in layer_grads]
        median_norms = [torch.median(norm) for norm in layer_norms]
        pert = self.generate_perturbation(attack_locs_layered,median_norms)
        self.adv_momentum = pert
        

    def split_to_layers(self, stacked_grads):
        # Split gradients into layers
        layered = []
        for i in range(len(self.layer_inds)-1):
            start_idx = self.layer_inds[i]
            end_idx = self.layer_inds[i + 1]
            layered.append(stacked_grads[:, start_idx:end_idx])
        return layered
    
    def split_to_layers_single(self, grad):
        # Split a single gradient into layers
        layered = []
        for i in range(len(self.layer_inds)-1):
            start_idx = self.layer_inds[i]
            end_idx = self.layer_inds[i + 1]
            layered.append(grad[start_idx:end_idx])
        return layered

    def generate_perturbation(self, attack_locs_layered,layer_norms):
        # Generate perturbation based on benign gradients
        pert_per_layer = []
        for norm,locs in zip(layer_norms,attack_locs_layered):
            pert = locs 
            pert_norm = torch.norm(pert.float())
            if pert_norm == 0:
                pert_per_layer.append(torch.zeros_like(pert))
            else:
                pert = pert / torch.norm(pert.float()) * norm
                pert_per_layer.append(-pert)
            #print(norm,pert.norm(),'med - layer pert norm')
        pert_flat = torch.cat(pert_per_layer, dim=0)
        #print((pert_flat>0).sum() / pert_flat.numel())
        return pert_flat
