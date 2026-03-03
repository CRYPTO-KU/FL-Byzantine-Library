import numpy as np
import matplotlib.pyplot as plt
from .force import get_prune_mask
from .synflow import get_synflow_mask
from .synflow_new import get_synflow_mask as get_synflow_mask2
from .random_prune import get_random_mask, layer_wise_random_with_crits,non_prunables,layer_wise_random_no_crits
from .erk import googleAI_ERK,tim_ERK
from .force2 import get_prune_mask_var
from .gem_miner import get_gem_mask
from .epi import get_epi_mask
from .lamp import get_lamp_mask
from .prune_basic import get_mask_basic
import torch.nn as nn
import torch
from os.path import exists, join
from utils import count_parameters
from data_loader import get_pruning_dataset,get_pruning_datasets_dist, get_pruning_datasets_dist_omniscient
from copy import deepcopy


def initialize_model(model, init_method='kaiming_normal'):
    """
    Initialize a neural network model with the specified initialization method.
    
    Args:
        model (torch.nn.Module): The neural network model to initialize
        init_method (str): The initialization method to use. Options:
            - 'xavier_uniform' or 'glorot_uniform': Xavier/Glorot uniform initialization
            - 'xavier_normal' or 'glorot_normal': Xavier/Glorot normal initialization  
            - 'kaiming_uniform' or 'he_uniform': Kaiming/He uniform initialization
            - 'kaiming_normal' or 'he_normal': Kaiming/He normal initialization
            - 'normal': Normal distribution with mean=0, std=0.01
            - 'uniform': Uniform distribution in [-0.1, 0.1]
            - 'zeros': Initialize with zeros
            - 'ones': Initialize with ones
            - 'orthogonal': Orthogonal initialization
    
    Returns:
        torch.nn.Module: The initialized model
    """
    
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            if init_method in ['xavier_uniform', 'glorot_uniform']:
                nn.init.xavier_uniform_(m.weight)
            elif init_method in ['xavier_normal', 'glorot_normal']:
                nn.init.xavier_normal_(m.weight)
            elif init_method in ['kaiming_uniform', 'he_uniform','ku']:
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_method in ['kaiming_normal', 'he_normal','kn']:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init_method == 'uniform':
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            elif init_method == 'zeros':
                nn.init.zeros_(m.weight)
            elif init_method == 'ones':
                nn.init.ones_(m.weight)
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")
            
            # Initialize bias if it exists
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Initialize batch norm layers
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    # Apply initialization to all modules
    model.apply(init_weights)
    return model


list_loaders = ['force_var', 'force_var_weight', 'force_var_grad', 'force_var_mixed','basic']


def get_sparse_mask(args, loader, net, device):
    """
    Get pruning mask using method name instead of magic numbers.
    
    Args:
        args: Arguments containing prune_method (str) and other configs
        loader: Data loader for pruning
        net: Neural network model
        device: Device to run on
    
    Returns:
        torch.Tensor: Binary mask for pruning
    """
    net_copy = deepcopy(net).to(device)
    if args.init != '-' and args.init is not None:
        initialize_model(net_copy, init_method=args.init)

    # Define pruning method mappings
    pruning_methods = {
        # Progressive Skeletonization methods (1-3)
        'iter_snip': lambda: get_prune_mask(args, net_copy, loader, device),
        'grasp_it': lambda: get_prune_mask(args, net_copy, loader, device),
        'force': lambda: get_prune_mask(args, net_copy, loader, device),

        # SynFlow-based methods (4-6)
        'synflow': lambda: get_synflow_mask2(args, net_copy, loader, device),
        'grasp': lambda: get_synflow_mask(args, loader, device),
        'snip': lambda: get_synflow_mask(args, loader, device),
        
        'erk': lambda: googleAI_ERK(net, args.pruning_factor),
        'uniform': lambda: get_synflow_mask(args, loader, device),
        'uniform_plus': lambda: get_synflow_mask(args, loader, device),
        
        # Random methods (11-13, 16)
        'random': lambda: get_random_mask(net_copy, args, device),
        'random_plus': lambda: get_random_mask(net_copy, args, device),
        'random_layerwise': lambda: layer_wise_random_with_crits(net, args, device),
        'random_no_crits': lambda: layer_wise_random_no_crits(net, args, device),
        'basic': lambda: get_mask_basic(args, net_copy, loader, device),

        # Modified force methods, Variance enhanced force
        'force_var': lambda: get_prune_mask_var(args, net_copy, loader, device),
        'force_var_weight': lambda: get_prune_mask_var(args, net_copy, loader, device),
        'force_var_grad': lambda: get_prune_mask_var(args, net_copy, loader, device),
        'force_var_mixed': lambda: get_prune_mask_var(args, net_copy, loader, device),
        # SOTA methods
        'gem': lambda: get_gem_mask(args, net_copy, loader, device),
        'lamp': lambda: get_lamp_mask(args, net_copy, loader, device),
        'epi': lambda: get_epi_mask(args, net_copy, loader, device)
    }
    
    # Legacy support: map old integer methods to new string methods
    legacy_mapping = {
        1: 'iter_snip', 2: 'grasp_it', 3: 'force',
        4: 'synflow', 5: 'grasp', 6: 'snip',
        7: 'lamp', 8: 'erk', 9: 'uniform', 10: 'uniform_plus',
        11: 'random', 12: 'random_plus', 13: 'random_layerwise',
        14: 'force_var', 15: 'non_prunables', 16: 'random_no_crits',
        17: 'force_cons', 18: 'force_dist', 19: 'erk',
        20: 'force_layerwise', 21: 'force_layerwise_fc', 22: 'force_layer',
        24: 'force_omniscient', 28: 'force_adaptive'
    }
    
    # Get method name (support both string and legacy int)
    method = args.prune_method
    if isinstance(method, int):
        if method in legacy_mapping:
            method = legacy_mapping[method]
        else:
            raise ValueError(f"Unknown pruning method number: {method}")
    
    # Execute pruning method
    if method not in pruning_methods:
        raise ValueError(f"Unknown pruning method: {method}. Available methods: {list(pruning_methods.keys())}")
    
    mask = pruning_methods[method]()
    
    # Calculate and report statistics
    left = mask.sum().item()
    total = mask.numel()
    sparsity_percent = (left / total) * 100
    
    print(f'Pruning method: {method}')
    print(f'Remaining parameters: {sparsity_percent:.2f}%')

    del net_copy

    return mask



def bn_mask(net,device):
    import torch.nn as nn
    mask = torch.empty((0),device=device)
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


def get_attack_locations(args,net_ps,dataset,byzantine_inds,benign_inds,device):
    bn_locs = bn_mask(net_ps,device)
    num_benign = (args.num_client * (1 - args.traitor) if args.traitor < 1 else args.num_client - args.traitor)
    mask_file = 'Masks'
    if args.load_mask:
        prune_mask = torch.zeros(count_parameters(net_ps),device=device)
        loc = join(mask_file,args.load_mask)
        assert exists(loc), "Mask file does not exist:" + loc
        inds = torch.load(loc).to(device)
        prune_mask[inds] = 1
        print('Prune mask loaded from file')
    else:
        if args.prune_method in list_loaders:
            if args.omniscient_pruning:
                prune_loader = get_pruning_datasets_dist_omniscient(dataset,benign_inds,args.prune_bs)
            else:
                prune_loader = get_pruning_datasets_dist(args,dataset,byzantine_inds,num_benign,args.prune_bs)
        else:
            prune_loader = get_pruning_dataset(args,dataset)
        prune_mask = get_sparse_mask(args,prune_loader,net_ps,device)
        pruned_histogram(net_ps, prune_mask)
    print('left weights from prunable parameters:', round((prune_mask.sum() / prune_mask.numel()).item(),3))
    return prune_mask, bn_locs

def save_mask(mask, net, args):
    dims, flat_dims, names, colors = get_layer_dims(net)
    d = {'dims': dims, 'flat_dims': flat_dims, 'names': names, 'colors': colors}
    d['mask'] = mask
    # Use pickle for saving dictionary objects
    import pickle
    filename = f'Prune_mask-method_{args.prune_method}-sparsity_{args.pruning_factor}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(d, f)

def pruned_histogram(net,mask,exclude_bns = True):
    dims, flat_dims,names,colors = get_layer_dims(net)
    cum_dim = [0]
    cum_dim.extend(flat_dims)
    cum_dim = np.cumsum(cum_dim)
    mask = mask.detach().cpu().numpy()
    totals = []
    total_max = 0
    for i in range(len(cum_dim)-1):
        b, e = cum_dim[i], cum_dim[i + 1]
        p = mask[b:e]
        size = len(p)
        pruned = np.sum(p)
        total = round((pruned / size) * 100, 1)
        totals.append(total)
        if 99 > total > total_max:
            total_max = total
    if exclude_bns:
        colors = np.asarray(colors)
        weights = colors != 'tab:red'
        colors = colors[weights]
        totals = np.asarray(totals)[weights]
        names = np.asarray(names)[weights]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(names,totals,color=colors)
    ax.set_ylabel('Remaining parameters %')
    #ax.set_title('')
    ax.set_xlabel('Layers')
    ax.set_ylim(0, total_max + 5)
    plt.tight_layout()
    plt.show()
    return

def get_layer_dims(net):
    dims = []
    names = []
    colors = []
    b,c,f = 1,1,1
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            dims.append(layer.weight.shape)
            colors.append('tab:red')
            names.append('bnw{}'.format(b))
            if layer.bias is not None:
                dims.append(layer.bias.shape)
                names.append('bnb{}'.format(b))
                colors.append('tab:red')
            b+=1
        elif isinstance(layer, nn.Linear):
            dims.append(layer.weight.shape)
            colors.append('tab:blue')
            names.append('Fc{}'.format(f))
            if layer.bias is not None:
                dims.append(layer.weight.shape)
                names.append('fcb{}'.format(f))
                colors.append('tab:red')
            f+=1
        elif isinstance(layer, nn.Conv2d):
            dims.append(layer.weight.shape)
            colors.append('tab:blue')
            names.append('Cw{}'.format(c))
            if layer.bias is not None:
                dims.append(layer.bias.shape)
                names.append('cb{}'.format(c))
                colors.append('tab:red')
            c+=1
    flat_dims = [np.prod(d) for d in dims]
    return dims, flat_dims,names,colors

