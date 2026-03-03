import torch
import numpy
import random
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def critical_params(net,device):
    import torch.nn as nn
    mask = torch.empty((0)).to(device)
    first_conv = True
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.ones_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer,nn.Conv2d):
            if first_conv:
                mask_ = torch.ones_like(layer.weight.data.detach().flatten())
                first_conv = False
            else:
                mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None: ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
    return mask

def non_prunables(net,device):
    import torch.nn as nn
    mask = torch.empty((0)).to(device)
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.ones_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer,nn.Conv2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None: ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
    return mask



def get_random_mask(net,args,device):
    params = count_parameters(net)
    base_mask = non_prunables(net,device)
    amount = int(args.pruning_factor * params)
    if args.prune_method == 11:
        amount += base_mask.sum().item()
    perms = torch.randperm(params,device=device,dtype=torch.int)
    indx = perms[:int(amount)]
    indx = indx.long()
    m = torch.zeros(params,device=device)
    m[indx] = 1
    if args.prune_method==12:
        m.add_(base_mask)
        m = (m>0).int()
    return m

def layer_wise_random_with_crits(net,args,device):
    import torch.nn as nn
    mask = torch.empty((0)).to(device)
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.ones_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            amount = int(args.pruning_factor * mask_.numel()) + 1
            keep =  torch.randperm(mask_.numel(),device=device,dtype=torch.long)[:amount]
            mask_[keep] = 1
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer,nn.Conv2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            amount = int(args.pruning_factor * mask_.numel()) + 1
            keep = torch.randperm(mask_.numel(), device=device, dtype=torch.long)[:amount]
            mask_[keep] = 1
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None: ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
    print(mask.numel(),mask.shape,mask.sum())
    return mask

def layer_wise_random_no_crits(net,args,device):
    import torch.nn as nn
    mask = torch.empty((0)).to(device)
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            amount = int(args.pruning_factor * mask_.numel()) + 1
            keep = torch.randperm(mask_.numel(), device=device, dtype=torch.long)[:amount]
            mask_[keep] = 1
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:
                mask_bias = torch.zeros_like(layer.bias.data.detach().flatten())
                amount = int(args.pruning_factor * mask_bias.numel()) + 1
                keep = torch.randperm(mask_bias.numel(), device=device, dtype=torch.long)[:amount]
                mask_bias[keep] = 1
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            amount = int(args.pruning_factor * mask_.numel()) + 1
            keep =  torch.randperm(mask_.numel(),device=device,dtype=torch.long)[:amount]
            mask_[keep] = 1
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.zeros_like(layer.bias.data.detach().flatten())
                amount = int(args.pruning_factor * mask_bias.numel()) + 1
                keep = torch.randperm(mask_bias.numel(), device=device, dtype=torch.long)[:amount]
                mask_bias[keep] = 1
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer,nn.Conv2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            amount = int(args.pruning_factor * mask_.numel()) + 1
            keep = torch.randperm(mask_.numel(), device=device, dtype=torch.long)[:amount]
            mask_[keep] = 1
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None: ## if there is no bn
                mask_bias = torch.zeros_like(layer.bias.data.detach().flatten())
                amount = int(args.pruning_factor * mask_bias.numel()) + 1
                keep = torch.randperm(mask_bias.numel(), device=device, dtype=torch.long)[:amount]
                mask_bias[keep] = 1
                mask = torch.cat((mask, mask_bias),dim=0)
    return mask