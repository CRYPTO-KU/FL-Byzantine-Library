import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import gc
import sys
import os
from typing import List, Tuple, Union, Optional, Dict, Any
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:
    psutil = None


def get_prune_mask(args: Any, net_ps: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Main function to get pruning mask with improved memory management and error handling.
    """
    net = copy.deepcopy(net_ps).to(device)

    method_dic = {
        "Iter SNIP": 1,
        "GRASP-It": 2,
        "FORCE": 3,
    }
    prune_method = method_dic.get(args.prune_method, 3)  # Default to FORCE if not found

    keep_masks = iterative_pruning(net, loader, device, args.pruning_factor,
                                   prune_method=prune_method,
                                   num_steps=args.num_steps,
                                   mode=args.mode, num_batches=args.num_batches,
                                       last_layer_constraint=getattr(args, 'last_layer_constraint', -1),
                                       first_layer_constraint=getattr(args, 'first_layer_constraint', -1),
                                   layer_constraint=getattr(args, 'layer_constraint', -1))
    # Cleanup
    cleanup_gpu_memory()
    return keep_masks


def cleanup_gpu_memory() -> None:
    """Comprehensive GPU memory cleanup for multiprocessing environments"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    gc.collect()
    gc.collect()  # Multiple calls sometimes needed


def apply_prune_mask(net: nn.Module, keep_masks: List[torch.Tensor], apply_hooks: bool = True) -> List[Any]:
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    If apply_hooks == False, then set weight to 0 but do not block the gradient.
    This is used for FORCE algorithm that sparsifies the net instead of pruning.
    """

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all non-prunable modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())
    #This is how filter works: filter(func,iterable); filter returns the elements such that func(element in iterable) == True

    # List of hooks to be applied on the gradients. It's useful to save them in order to remove
    # them later
    hook_handlers = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape) # gives an error if the condition is not true

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook #returns a hook function which work as masking

        # Step 1: Set the masked weights to zero (Biases are ignored)
        layer.weight.data[keep_mask == 0.] = 0. # Masked the weights

        # Step 2: Make sure their gradients remain zero (not with FORCE)
        # Register_hook: Registers a backward hook; hook is called whenever gradient is computed
        if apply_hooks:
            hook_handlers.append(layer.weight.register_hook(hook_factory(keep_mask))) # Automatically masks the gradients

    return hook_handlers

def get_average_gradients(net: nn.Module, train_dataloader: DataLoader, device: torch.device, num_batches: int = -1) -> List[torch.Tensor]:
    """
    Function to compute gradients and average them over several batches.
    num_batches: Number of batches to be used to approximate the gradients.
                 When set to -1, uses the whole training set.
    Returns a list of tensors, with gradients for each prunable layer.
    """

    # Prepare list to store gradients
    gradients = []
    for layer in net.modules():
        # Select only prunable layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gradients.append(0)

    # Take a whole epoch
    count_batch = 0
    for batch_idx in range(len(train_dataloader)):
        inputs, targets = next(iter(train_dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Compute gradients (but don't apply them)
        net.zero_grad()
        outputs = net(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()

        # Store gradients
        counter = 0 # Sum gradients over layers
        for layer in net.modules():
            # Select only prunable layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    gradients[counter] += layer.weight.grad
                counter += 1
        count_batch += 1
        if batch_idx == num_batches - 1:
            break
    avg_gradients = [x / count_batch for x in gradients]

    return avg_gradients

######################## Pruning with saliency metric ##################################
def get_average_saliencies(net: nn.Module, train_dataloader: DataLoader, device: torch.device, prune_method: int = 3, num_batches: int = -1,
                           original_weights: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """
    Get saliencies with averaged gradients.
    num_batches: Number of batches to be used to approximate the gradients.
                 When set to -1, uses the whole training set.
    prune_method: Which method to use to prune the layers, refer to https://arxiv.org/abs/2006.09081.
                   1: Use Iter SNIP.
                   2: Use GRASP-It.
                   3: Use FORCE (default).
    Returns a list of tensors with saliencies for each weight.
    """

    def pruning_criteria(method):
        if method == 2:
            # GRASP-It method
            result = layer_weight_grad ** 2  # Custom gradient norm approximation
        else:
            # FORCE / Iter SNIP method
            result = torch.abs(layer_weight * layer_weight_grad) # ------> Saliency metric is hadamard product of weight and gradient
        return result

    gradients = get_average_gradients(net, train_dataloader, device, num_batches) #Compute average gradient

    saliency = []
    idx = 0 #index of the layer
    for layer in net.modules(): # Prune only convolutional and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if prune_method == 3 and original_weights is not None:
                layer_weight = original_weights[idx] # Use the original weights for the FORCE algorithm
            else:
                layer_weight = layer.weight # In other algorithms use the current weight
            layer_weight_grad = gradients[idx] # Recall the average gradient values
            idx += 1
            saliency.append(pruning_criteria(prune_method))

    return saliency


###################################################
############# Iterative pruning ###################
###################################################

def get_mask(saliency: List[torch.Tensor], pruning_factor: float) -> List[torch.Tensor]:
    """
    Given a list of saliencies and a pruning factor (sparsity),
    returns a list with binary tensors which correspond to pruning masks.
    """
    if not saliency or len(saliency) == 0:
        return []
    
    # Validate pruning_factor
    if pruning_factor <= 0:
        print(f"Warning: pruning_factor={pruning_factor} <= 0, returning all-zero masks")
        return [torch.zeros_like(s) for s in saliency]
    
    if pruning_factor >= 1.0:
        print(f"Warning: pruning_factor={pruning_factor} >= 1.0, returning all-ones masks")
        return [torch.ones_like(s) for s in saliency]
    
    # Check for NaN or Inf values in saliencies and fix them
    clean_saliency = []
    for i, s in enumerate(saliency):
        if torch.isnan(s).any() or torch.isinf(s).any():
            print(f"Warning: Layer {i} has NaN/Inf saliencies, replacing with random values")
            s = numeric_fix(s)
        clean_saliency.append(s)
    
    # Add small random noise to break ties (with fixed seed for reproducibility)
    torch.manual_seed(0)
    noisy_saliency = [s + torch.randn_like(s) * 1e-10 for s in clean_saliency]

    all_scores = torch.cat([torch.flatten(x) for x in noisy_saliency])
    total_params = len(all_scores)
    num_params_to_keep = round(total_params * pruning_factor)
    
    # Ensure we keep at least 1 parameter for extreme sparsity
    num_params_to_keep = max(1, num_params_to_keep)
    num_params_to_keep = min(num_params_to_keep, total_params)
    
    if num_params_to_keep <= 0:
        return [torch.zeros_like(s) for s in saliency]
    
    if num_params_to_keep >= len(all_scores):
        return [torch.ones_like(s) for s in saliency]
    
    threshold, indices = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    
    # Create masks
    prune_masks = []
    current_offset = 0
    for i, m in enumerate(noisy_saliency):
        flat_saliency = m.flatten()
        layer_size = len(flat_saliency)
        
        # Get the indices for this layer from the global topk selection
        layer_indices = indices[(current_offset <= indices) & (indices < current_offset + layer_size)] - current_offset
        
        # Create mask: 1 for selected indices, 0 for others
        mask = torch.zeros_like(flat_saliency)
        if len(layer_indices) > 0:
            mask[layer_indices] = 1.0
        
        # Reshape back to original shape
        mask = mask.view_as(m)
        prune_masks.append(mask)
        current_offset += layer_size
    
    return prune_masks


def numeric_fix(tensor: torch.Tensor) -> torch.Tensor:
    """Replace inf and NaN values in a tensor with appropriate finite numbers."""
    tensor = tensor.clone()
    tensor[torch.isinf(tensor)] = 1e9 
    tensor[torch.isnan(tensor)] = 0.0  # Replace NaN with zero
    return tensor


def f_flatten(saliency_list: List[torch.Tensor], unassigned_indices: List[int]) -> torch.Tensor:
    """
    Flatten Saliencies from Unassigned Layers
    """
    flattened_scores = []
    for i in unassigned_indices:
        flattened_scores.append(torch.flatten(saliency_list[i]))
    return torch.cat(flattened_scores) if flattened_scores else torch.tensor([])


def f_th(flattened_scores: torch.Tensor, n_target: int) -> float:
    """
    Global Threshold Computation
    """
    if len(flattened_scores) == 0 or n_target <= 0:
        return float('inf')
    
    n_target = int(n_target)
    n_target = min(n_target, len(flattened_scores))
    if n_target >= len(flattened_scores):
        return float('-inf')
    
    threshold, _ = torch.topk(flattened_scores, n_target, sorted=True)
    return threshold[-1].item()


def f_maskglobal(S: List[torch.Tensor], A: List[int], u_th: float) -> List[torch.Tensor]:
    """
    Global Mask Generation
    """
    masks = []
    unassigned_set = set(A)
    
    for i in range(len(S)):
        if i in unassigned_set:
            mask = (S[i] >= u_th).float()
        else:
            mask = torch.zeros_like(S[i])
        masks.append(mask)
    return masks


def f_masklocal(saliency: torch.Tensor, n_target: int) -> torch.Tensor:
    """
    Local Mask Generation
    """
    if n_target <= 0:
        return torch.zeros_like(saliency)
    
    flat_saliency = saliency.flatten()
    if n_target >= len(flat_saliency):
        return torch.ones_like(saliency)
    
    threshold, _ = torch.topk(flat_saliency, n_target, sorted=True)
    local_threshold = threshold[-1]
    mask = (flat_saliency >= local_threshold).float().view_as(saliency)
    return mask


def get_mask_layer_constraint(saliency: List[torch.Tensor], pruning_factor: float, layer_constraint: float) -> List[torch.Tensor]:
    """
    Layer-Constrained Mask Generation
    """
    if not saliency or len(saliency) == 0:
        return []
    
    if pruning_factor <= 0:
        return [torch.zeros_like(s) for s in saliency]
    if pruning_factor >= 1.0:
        return [torch.ones_like(s) for s in saliency]
    if layer_constraint <= 0:
        return [torch.zeros_like(s) for s in saliency]
    
    # Calculate total parameters and target
    total_params = sum(s.numel() for s in saliency)
    n_target = math.ceil(total_params * pruning_factor)
    
    # Calculate n_target for each layer based on layer_constraint
    n_target_per_layer = [math.ceil(s.numel() * layer_constraint) for s in saliency]
    
    # Initialize outputs
    L = len(saliency)
    final_masks: List[Optional[torch.Tensor]] = [None] * L
    unassigned = list(range(L))
    
    while unassigned:
        # Flatten saliencies from unassigned layers
        S = f_flatten(saliency, unassigned)
        
        # Compute global threshold
        u_th = f_th(S, n_target)
        
        # Apply global threshold to all layers
        M = f_maskglobal(saliency, unassigned, u_th)
        
        const_flag = 0
        layers_to_remove = []
        
        # Check layer-wise constraint
        for i in unassigned:
            mask_sum = M[i].sum().item()
            if mask_sum > n_target_per_layer[i]:
                # Generate local mask for this layer
                final_masks[i] = f_masklocal(saliency[i], n_target_per_layer[i])
                layers_to_remove.append(i)
                n_target -= n_target_per_layer[i]
                const_flag = 1
        
        # Remove assigned layers from unassigned list
        for i in layers_to_remove:
            unassigned.remove(i)
        
        # If no constraint violations, use the globally constructed masks
        if const_flag == 0:
            for i in unassigned:
                final_masks[i] = M[i]
            break
    
    return final_masks


def get_mask_layer_constraint_first_last2(saliency: List[torch.Tensor], pruning_factor: float, 
                                         first_layer_constraint: float, last_layer_constraint: float) -> List[torch.Tensor]:
    """
    Layer-Constrained Mask Generation for First and Last Layers Only
    
    Args:
        saliency: List of saliency tensors for each layer
        pruning_factor: Global fraction of weights to keep (0 to 1)
        first_layer_constraint: Maximum fraction of weights the first layer can keep (0 to 1)
        last_layer_constraint: Maximum fraction of weights the last layer can keep (0 to 1)
    
    Returns:
        List of binary mask tensors
    """
    if not saliency or len(saliency) == 0:
        return []
    
    if pruning_factor <= 0:
        return [torch.zeros_like(s) for s in saliency]
    if pruning_factor >= 1.0:
        return [torch.ones_like(s) for s in saliency]
    
    # Calculate total parameters and global target
    total_params = sum(s.numel() for s in saliency)
    n_target = math.ceil(total_params * pruning_factor)
    
    # Calculate n_target for first and last layers based on their constraints
    L = len(saliency)
    n_target_per_layer = [None] * L
    
    # Set constraints only for first and last layers
    if first_layer_constraint > 0:
        n_target_per_layer[0] = int(saliency[0].numel() * first_layer_constraint)
    if last_layer_constraint > 0:
        n_target_per_layer[L-1] = int(saliency[L-1].numel() * last_layer_constraint)
    
    # Initialize outputs
    final_masks: List[Optional[torch.Tensor]] = [None] * L
    unassigned = list(range(L))
    constrained_layers = []
    
    # Mark first and last layers as constrained if they have valid constraints
    if first_layer_constraint > 0:
        constrained_layers.append(0)
    if last_layer_constraint > 0 and L > 1:
        constrained_layers.append(L-1)
    
    # Pre-assign constrained layers if their constraints are active
    for i in constrained_layers:
        if n_target_per_layer[i] is not None and n_target_per_layer[i] > 0:
            # Apply local mask directly based on constraint
            final_masks[i] = f_masklocal(saliency[i], n_target_per_layer[i])
            unassigned.remove(i)
            n_target -= n_target_per_layer[i]
    
    # Now handle remaining layers with global pruning
    if unassigned and n_target > 0:
        # Flatten saliencies from unassigned layers
        S = f_flatten(saliency, unassigned)
        
        # Compute global threshold for remaining parameters
        u_th = f_th(S, n_target)
        
        # Apply global threshold to remaining layers
        for i in unassigned:
            final_masks[i] = (saliency[i] >= u_th).float()
    elif unassigned:
        # No more budget left for unassigned layers
        for i in unassigned:
            final_masks[i] = torch.zeros_like(saliency[i])
    
    return final_masks

def get_mask_layer_constraint_first_last(saliency: List[torch.Tensor], pruning_factor: float, 
                                         first_layer_constraint: float, last_layer_constraint: float) -> List[torch.Tensor]:
    """
    Layer-Constrained Mask Generation for First and Last Layers Only
    
    Args:
        saliency: List of saliency tensors for each layer
        pruning_factor: Global fraction of weights to keep (0 to 1)
        first_layer_constraint: Maximum fraction of weights the first layer can keep (0 to 1)
        last_layer_constraint: Maximum fraction of weights the last layer can keep (0 to 1)
    
    Returns:
        List of binary mask tensors
    """
    if not saliency or len(saliency) == 0:
        return []
    
    if pruning_factor <= 0:
        return [torch.zeros_like(s) for s in saliency]
    if pruning_factor >= 1.0:
        return [torch.ones_like(s) for s in saliency]
    
    # Calculate total parameters and global target
    total_params = sum(s.numel() for s in saliency)
    n_target = math.ceil(total_params * pruning_factor)
    
    # Calculate n_target for first and last layers based on their constraints
    L = len(saliency)
    n_target_per_layer = [None] * L
    
    # Set constraints only for first and last layers
    if first_layer_constraint > 0:
        n_target_per_layer[0] = math.ceil(saliency[0].numel() * first_layer_constraint)
    if last_layer_constraint > 0:
        n_target_per_layer[L-1] = math.ceil(saliency[L-1].numel() * last_layer_constraint)
    
    # Initialize outputs
    final_masks: List[Optional[torch.Tensor]] = [None] * L
    unassigned = list(range(L))
    constrained_layers = []
    
    # Mark first and last layers as constrained if they have valid constraints
    if first_layer_constraint > 0:
        constrained_layers.append(0)
    if last_layer_constraint > 0 and L > 1:
        constrained_layers.append(L-1)
    
    while unassigned:
        # Flatten saliencies from unassigned layers
        S = f_flatten(saliency, unassigned)
        
        # Compute global threshold
        u_th = f_th(S, n_target)
        
        # Apply global threshold to all layers
        M = f_maskglobal(saliency, unassigned, u_th)
        
        const_flag = 0
        layers_to_remove = []
        
        # Check constraint only for first and last layers
        for i in unassigned:
            if i in constrained_layers and n_target_per_layer[i] is not None:
                mask_sum = M[i].sum().item()
                if mask_sum > n_target_per_layer[i]:
                    # Generate local mask for this constrained layer
                    final_masks[i] = f_masklocal(saliency[i], n_target_per_layer[i])
                    layers_to_remove.append(i)
                    n_target -= n_target_per_layer[i]
                    const_flag = 1
        
        # Remove assigned layers from unassigned list
        for i in layers_to_remove:
            unassigned.remove(i)
        
        # If no constraint violations, use the globally constructed masks
        if const_flag == 0:
            for i in unassigned:
                final_masks[i] = M[i]
            break
    
    return final_masks

# Legacy functions removed - functionality integrated into main iterative_pruning function
# The following functions have been replaced with improved layer constraint handling:
# - get_mask2, get_mask3, get_mask4 replaced by get_mask_layer_constraint
# - iterative_pruning_orig removed (functionality merged into main function)
# - Utility functions for FORCE saliency, gradient norm, etc. removed (unused)
def iterative_pruning(ori_net: nn.Module, train_dataloader: DataLoader, device: torch.device, pruning_factor: float = 0.1,
                      prune_method: int = 3, num_steps: int = 10,
                      mode: str = 'exp', num_batches: int = 1, last_layer_constraint: float = -1, first_layer_constraint: float = -1, layer_constraint: float = -1) -> torch.Tensor:
    """
    Function to gradually remove weights from a network, recomputing the saliency at each step.
    pruning_factor: Fraction of remaining weights (globally) after pruning.
    prune_method: Which method to use to prune the layers, refer to https://arxiv.org/abs/2006.09081.
                   1: Use Iter SNIP.
                   2: Use GRASP-It.
                   3: Use FORCE (default).
    num_steps: Number of iterations to do when pruning progressively (should be >= 1).
    mode: Mode of choosing the sparsity decay schedule. One of 'exp', 'linear'
    num_batches: Number of batches to be used to approximate the gradients (should be -1 or >= 1).
                 When set to -1, uses the whole training set.
    last_layer_constraint: Constraint for last layer (was fc_threshold)
    first_layer_constraint: Constraint for first layer (was conv_threshold)
    layer_constraint: Maximum fraction of weights each layer can keep. If > 0, applies per-layer constraint.
    Returns a flattened binary tensor which corresponds to the final pruning mask.
    """
    # Let's create a copy of the network to make sure we don't affect the training later
    net = copy.deepcopy(ori_net) # Copy the original network

    if prune_method == 3:
        # If we want to apply FORCE we need to save the original (dense) weights
        # to compute the saliency of sparsified connections.
        original_weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                original_weights.append(layer.weight.detach()) #Detach the weight values
    else:
        original_weights = None

    # Choose a decay mode for sparsity (exponential should be used unless you know what you
    # are doing)
    if mode == 'linear':
        pruning_steps = [1 - ((x + 1) * (1 - pruning_factor) / num_steps) for x in range(num_steps)]

    elif mode == 'exp':
        pruning_steps = [np.exp(0 - ((x + 1) * (0 - np.log(pruning_factor)) / num_steps)) for x in range(num_steps)]

    mask = None
    hook_handlers = None
    modulo = num_steps // 4 if num_steps >= 10 else 1
    constraint_scale = layer_constraint / pruning_factor if pruning_factor > 0 else 1.0
    constraint_scale_first = first_layer_constraint / pruning_factor if first_layer_constraint > 0 and pruning_factor > 0 else 0
    constraint_scale_last = last_layer_constraint / pruning_factor if last_layer_constraint > 0 and pruning_factor > 0 else 0
    ############### Iterative pruning starts here ##############
    for step, perc in enumerate(pruning_steps):
        saliency = get_average_saliencies(net, train_dataloader, device,
                                          prune_method=prune_method,
                                          num_batches=num_batches,
                                          original_weights=original_weights) # Returns the saliency metric
        torch.cuda.empty_cache()

        # Make sure all saliencies of previously deleted weights is minimum so they do not
        # get picked again.
        if mask is not None and prune_method < 3:
            min_saliency = get_minimum_saliency(saliency)
            for ii in range(len(saliency)):
                saliency[ii][mask[ii] == 0.] = min_saliency

        if hook_handlers is not None:
            for h in hook_handlers:
                h.remove()
        
        # Use layer constraint if specified, otherwise use global pruning
        if layer_constraint > 0 and (first_layer_constraint > 0 or last_layer_constraint > 0):
            print("Warning: Both layer_constraint and first/last layer constraints are set. Using layer_constraint only.")

        if layer_constraint > 0:
            layer_constraint_adjusted = min(1.0, constraint_scale * perc)
            mask = get_mask_layer_constraint(saliency, perc, layer_constraint_adjusted)
        elif first_layer_constraint > 0 or last_layer_constraint > 0:
            first_layer_constraint_adjusted = min(1.0, constraint_scale_first * perc) if first_layer_constraint > 0 else -1
            last_layer_constraint_adjusted = min(1.0, constraint_scale_last * perc) if last_layer_constraint > 0 else -1
            mask = get_mask_layer_constraint_first_last(saliency, perc, first_layer_constraint_adjusted, last_layer_constraint_adjusted)
        else:
            mask = get_mask(saliency, perc)

        if prune_method == 3:
            net = copy.deepcopy(ori_net)
            apply_prune_mask(net, mask, apply_hooks=False)
        else:
            hook_handlers = apply_prune_mask(net, mask, apply_hooks=True)
            
        # Clean up intermediate objects to save memory
        if 'saliency' in locals():
            del saliency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if (step+1) % modulo == 0 and step != 0:
            p = check_global_pruning(mask)
            print(f'Global pruning {round(float(p),5)}','{}% Pruning complete'.format((step+1)/num_steps *100))
    
    # Safety check: ensure mask is valid
    if mask is None or not mask:
        print(f"Warning: mask is None or empty, creating sparse masks with pruning_factor={pruning_factor}")
        mask = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer_params = layer.weight.numel()
                keep_params = max(1, int(layer_params * pruning_factor))
                sparse_mask = torch.zeros_like(layer.weight)
                flat_mask = sparse_mask.flatten()
                indices = torch.randperm(len(flat_mask))[:keep_params]
                flat_mask[indices] = 1.0
                mask.append(flat_mask.view_as(layer.weight))

    # Create full masks for all layers
    c = 0
    mask_full = []
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_full.append(torch.ones_like(layer.weight))
            mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer, nn.Linear):
            if c < len(mask):
                mask_full.append(mask[c])
                c += 1
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer, nn.Conv2d):
            if c < len(mask):
                mask_full.append(mask[c])
                c += 1
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
    
    flat_mask = torch.cat([torch.flatten(x) for x in mask_full])
    
    # Cleanup
    cleanup_gpu_memory()
    
    return flat_mask


# iterative_pruning_orig function removed - functionality integrated into main iterative_pruning function


def check_global_pruning(mask: List[torch.Tensor]) -> float:
    """Compute fraction of unpruned weights in a mask"""
    flattened_mask = torch.cat([torch.flatten(x) for x in mask])
    return flattened_mask.mean().item()


def get_minimum_saliency(saliency: List[torch.Tensor]) -> float:
    """Compute minimum value of saliency globally"""
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.min().item()


def get_maximum_saliency(saliency: List[torch.Tensor]) -> float:
    """Compute maximum value of saliency globally"""
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.max().item()


####################################################################
######################    UTILS    #################################
####################################################################

# Legacy utility functions - kept for compatibility but marked as deprecated
def get_force_saliency(net: nn.Module, mask: List[torch.Tensor], train_dataloader: DataLoader, device: torch.device, num_batches: int) -> float:
    """
    Given a dense network and a pruning mask, compute the FORCE saliency.
    DEPRECATED: Use get_average_saliencies instead.
    """
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask, apply_hooks=True)
    saliencies = get_average_saliencies(net, train_dataloader, device,
                                        prune_method=3, num_batches=num_batches)
    torch.cuda.empty_cache()
    s = sum_unmasked_saliency(saliencies, mask)
    torch.cuda.empty_cache()
    return s


def sum_unmasked_saliency(variable: List[torch.Tensor], mask: List[torch.Tensor]) -> float:
    """Util to sum all unmasked (mask==1) components"""
    V = 0.0
    for v, m in zip(variable, mask):
        V += v[m > 0].sum().item()
    return V


def get_gradient_norm(net: nn.Module, mask: List[torch.Tensor], train_dataloader: DataLoader, device: torch.device, num_batches: int) -> float:
    """Given a dense network, compute the gradient norm after applying the pruning mask."""
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask)
    gradients = get_average_gradients(net, train_dataloader, device, num_batches)
    torch.cuda.empty_cache()
    norm = 0.0
    for g in gradients:
        norm += (g ** 2).sum().detach().cpu().numpy().item()
    return norm