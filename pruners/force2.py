import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gc
from typing import List, Tuple, Union, Optional, Dict, Any
from torch.utils.data import DataLoader
from utils import *
prune_method_mapper = {
        'force_var': 0,
        'force_var_weight': 1,
        'force_var_grad': 2,
        'force_var_mixed': 3,
    }


def get_prune_mask_var(args: Any, net_ps: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    net = copy.deepcopy(net_ps).to(device)  # Copy the original network
    try:
        method = prune_method_mapper[args.prune_method]  # Default to 0 if not found
    except:
        raise ValueError(f"Unknown pruning method: {args.prune_method}, use one of {list(prune_method_mapper.keys())}")
    
    keep_masks = iterative_pruning(net, loader, device, args.pruning_factor,
                                   prune_method=method,
                                   num_steps=args.num_steps,
                                   mode=args.mode, num_batches=args.num_batches,
                                   last_layer_constraint=getattr(args, 'last_layer_constraint', -1),
                                   first_layer_constraint=getattr(args, 'first_layer_constraint', -1),
                                   layer_constraint=getattr(args, 'layer_constraint', args.min_threshold),
                                   keep_original_weights=args.keep_orig_weights)
    
    # Immediate cleanup after getting the mask - we don't need anything else
    # More aggressive cleanup to prevent GPU memory leaks
    
    # First, explicitly move network to CPU to free GPU memory
    if hasattr(net, 'cpu'):
        net.cpu()
    
    # Clear any remaining gradients
    if hasattr(net, 'zero_grad'):
        net.zero_grad()
    
    # Delete the copied network
    del net
    
    # Additional cleanup for any remaining variables
    import sys
    frame = sys._getframe()
    local_vars = list(frame.f_locals.keys())
    for var_name in local_vars:
        if var_name not in ['keep_masks', 'args', 'device', 'loader']:  # Keep only essential variables
            try:
                del frame.f_locals[var_name]
            except:
                pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Try to reset CUDA context if possible
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    # Force garbage collection multiple times
    gc.collect()
    gc.collect()  # Sometimes needed twice
    
    # Final aggressive cleanup to prevent persistent processes
    cleanup_gpu_memory()
    
    return keep_masks
    

def cleanup_gpu_memory() -> None:
    """Comprehensive GPU memory cleanup for multiprocessing environments"""
    # First, clear all CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
        except:
            pass
    
    # Force multiple garbage collections
    import gc
    gc.collect()
    gc.collect()
    gc.collect()  # Multiple calls sometimes needed
    
    # Try to force process memory cleanup
    try:
        import os
        # Get current process ID
        pid = os.getpid()
        
        # Try psutil if available for more aggressive cleanup
        try:
            import psutil
            current_process = psutil.Process(pid)
            # Force memory reclaim
            _ = current_process.memory_info()
            # Clear any child processes that might be holding resources
            for child in current_process.children(recursive=True):
                try:
                    child.terminate()
                    child.wait(timeout=1)
                except:
                    pass
        except ImportError:
            # psutil not available, use basic OS cleanup
            try:
                os.sync() if hasattr(os, 'sync') else None
            except:
                pass
    except:
        pass
    
    # Additional CUDA context cleanup attempt
    if torch.cuda.is_available():
        try:
            # Try to reset the CUDA context completely
            torch.cuda.empty_cache()
            # Force synchronization one more time
            torch.cuda.synchronize()
        except:
            pass


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
        #print(layer.weight.shape,keep_mask.shape, layer.weight.shape == keep_mask.shape)
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

def get_average_gradients(net: nn.Module, train_dataloader: DataLoader, device: torch.device, num_batches: int = -1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Function to compute gradients and average them over several batches.
    num_batches: Number of batches to be used to approximate the gradients.
                 When set to -1, uses the whole training set.
    Returns a list of tensors, with gradients for each prunable layer.
    """

    # Prepare list to store gradients18:20
    gradients = []
    grad_vars = []
    for layer in net.modules():
        # Select only prunable layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gradients.append(0)
            grad_vars.append([])

    # Take a whole epoch
    count_batch = 0
    #sign_vec = torch.zeros(count_parameters(net),device=device)
    if isinstance(train_dataloader, list):
        for loader in train_dataloader:
            for batch_idx in range(len(loader)):
                inputs, targets = next(iter(loader))
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute gradients (but don't apply them)
                net.zero_grad()
                outputs = net(inputs)
                loss = F.nll_loss(outputs, targets)
                loss.backward()
                counter = 0  # Sum gradients over layers
                for layer in net.modules():
                    # Select only prunable layers
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        gradients[counter] += layer.weight.grad
                        grad_vars[counter].append(layer.weight.grad)
                        counter += 1
                count_batch += 1
                if batch_idx == num_batches - 1:
                    break
                #flat_grat = get_grad_flattened(net,device)
                #sign_vec+=flat_grat.sign()

                # Store gradients
                counter = 0
    else:
        for batch_idx in range(len(train_dataloader)):
            inputs, targets = next(iter(train_dataloader))
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute gradients (but don't apply them)
            net.zero_grad()
            outputs = net(inputs)
            loss = F.nll_loss(outputs, targets)
            loss.backward()
            #flat_grat = get_grad_flattened(net,device)
            #sign_vec+=flat_grat.sign()


            # Store gradients
            counter = 0 # Sum gradients over layers
            for layer in net.modules():
                # Select only prunable layers
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    gradients[counter] += layer.weight.grad
                    grad_vars[counter].append(layer.weight.grad)
                    counter += 1
            count_batch += 1
            if batch_idx == num_batches - 1:
                break
    avg_gradients = [x / count_batch for x in gradients]
    gradient_vars = []
    for layer in grad_vars:
        shape = layer[0].shape
        flats = [l.flatten() for l in layer]
        stacked_gradients = torch.stack(flats, 1)
        layer_vars = torch.std(stacked_gradients, 1)
        #layer_vars = torch.var(batch_stack, dim=1, keepdim=False)
        layer_vars.view(shape)
        gradient_vars.append(layer_vars.view(shape))

    return avg_gradients,gradient_vars

######################## Pruning with saliency metric ##################################
def get_average_saliencies(net: nn.Module, train_dataloader: DataLoader, device: torch.device, prune_method: int = 3, num_batches: int = -1,
                           original_weights: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """
    Get saliencies with averaged gradients.
    num_batches: Number of batches to be used to approximate the gradients.
                 When set to -1, uses the whole training set.
    prune_method: Which method to use to prune the layers, refer to https://arxiv.org/abs/2006.09081.
                   0: Use Variance only.
                   1: Use Variance * weight.
                   2: Use Variance * gradient.
                   3: Use variance * gradient * weight (all)
    Returns a list of tensors with saliencies for each weight.
    """

    def pruning_criteria(method):
        if method == 0:
            # Variance only
            result = layer_var_grad ** 2   # Custom gradient norm approximation
        elif method == 1:
            # Variance * weight
            result = torch.abs(layer_weight * layer_var_grad) # ------> Saliency metric is hadamard product of weight and gradient variance
        elif method == 2:
            # Variance * gradient
            result = torch.abs(layer_weight_grad * layer_var_grad) # ------> Saliency metric is hadamard product of weight and gradient
        elif method == 3:
            # method 18 force + layer variance * grad
            result = torch.abs(layer_weight * layer_weight_grad * layer_var_grad) # ------> Saliency metric is hadamard product of weight and gradient
        return result

    gradients, variance  = get_average_gradients(net, train_dataloader, device, num_batches) #Compute average gradient

    saliency = []
    idx = 0 #index of the layer
    for layer in net.modules(): # Prune only convolutional and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if original_weights is None:
                layer_weight = layer.weight # Use the original weights for the FORCE algorithm
            else: # In other algorithms use the current weight
                layer_weight = original_weights[idx]
            layer_weight_grad = gradients[idx] # Recall the average gradient values
            layer_var_grad = variance[idx]
            idx += 1
            saliency.append(pruning_criteria(prune_method))
    
    # Clean up after saliency computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
            # Replace NaN/Inf with corresponding values
            s = numeric_fix(s)
        else:
            clean_s = s
        clean_saliency.append(clean_s)
    
    # Add small random noise to break ties (with fixed seed for reproducibility)
    torch.manual_seed(0)
    noisy_saliency = [s + torch.randn_like(s) * 1e-10 for s in clean_saliency]

    all_scores = torch.cat([torch.flatten(x) for x in noisy_saliency])
    total_params = len(all_scores)
    num_params_to_keep = round(total_params * pruning_factor)  # Use round instead of int to avoid truncation bias
    
    # Ensure we keep at least 1 parameter for extreme sparsity
    num_params_to_keep = max(1, num_params_to_keep)
    num_params_to_keep = min(num_params_to_keep, total_params)
    
    # print(f"Pruning: keeping {num_params_to_keep}/{total_params} parameters ({num_params_to_keep/total_params:.6f})")
    
    if num_params_to_keep <= 0:
        # If no parameters to keep, return all-zero masks
        return [torch.zeros_like(s) for s in saliency]
    
    if num_params_to_keep >= len(all_scores):
        # If keeping all parameters, return all-one masks
        return [torch.ones_like(s) for s in saliency]
    
    threshold, indices = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    
    # Create a more precise mask that handles ties correctly
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
    
    # Verify the total number of kept parameters
    total_kept = sum(mask.sum().item() for mask in prune_masks)
    actual_ratio = total_kept / total_params
    # print(f"Actual pruning result: {total_kept}/{total_params} parameters ({actual_ratio:.6f})")
    
    return prune_masks

def f_flatten(saliency_list: List[torch.Tensor], unassigned_indices: List[int]) -> torch.Tensor:
    """
    Flatten Saliencies from Unassigned Layers
    
    Concatenate saliency tensors from unassigned layers into a single flattened vector.
    
    Args:
        saliency_list (List[torch.Tensor]): Set of saliency tensors S = {s_1, ..., s_L}
        unassigned_indices (List[int]): Unassigned layer indices A
        
    Returns:
        torch.Tensor: Flattened saliency vector S
        
    Algorithm:
        Initialize S = ∅
        For each i in A:
            s_flat = flatten(s_i)
            S = concatenate(S, s_flat)
    """
    flattened_scores = []
    for i in unassigned_indices:
        flattened_scores.append(torch.flatten(saliency_list[i]))
    return torch.cat(flattened_scores) if flattened_scores else torch.tensor([])

def f_th(flattened_scores: torch.Tensor, n_target: int) -> float:
    """
    Global Threshold Computation
    
    Compute global threshold value for target number of parameters.
    
    Args:
        flattened_scores (torch.Tensor): Flattened saliency vector S
        n_target (int): Target number of parameters to keep
        
    Returns:
        float: Global threshold u_th
    """
    if len(flattened_scores) == 0 or n_target <= 0:
        return float('inf')
    
    # Ensure n_target is an integer
    n_target = int(n_target)
    n_target = min(n_target, len(flattened_scores))
    if n_target >= len(flattened_scores):
        return float('-inf')
    
    threshold, _ = torch.topk(flattened_scores, n_target, sorted=True)
    return threshold[-1].item()

def f_maskglobal(S: List[torch.Tensor], A: List[int], u_th: float) -> List[torch.Tensor]:
    """
    Global Mask Generation
    
    Apply global threshold to create binary masks.
    For the layers where a mask has been already assigned we return all zeros mask.
    
    Parameters:
        S (List[torch.Tensor]): List of saliency values, each vector for a certain layer.
        A (List[int]): Layer indices (unassigned layers)
        u_th (float): Global threshold value
        
    Returns:
        List[torch.Tensor]: Set of masks M = {m_1, ..., m_L} where L = len(S).
        
    Algorithm:
        For each layer index i in range(L):
            if i in A:
                m_i = (s_i >= u_th)  # Apply global threshold to unassigned layers
            else:
                m_i = zeros_like(s_i)  # Placeholder for assigned layers
    """
    masks = []
    unassigned_set = set(A)  # Convert to set for faster lookup
    
    for i in range(len(S)):
        if i in unassigned_set:
            mask = (S[i] >= u_th).float()
        else:
            # Create placeholder mask for assigned layers (won't be used)
            mask = torch.zeros_like(S[i])
        masks.append(mask)
    return masks

def f_masklocal(saliency: torch.Tensor, n_target: int) -> torch.Tensor:
    """
    Local Mask Generation
    
    Create binary mask for single layer using local top-k selection.
    
    Args:
        saliency (torch.Tensor): Saliency tensor s_i for layer i
        n_target (int): Target number of parameters n_target^(i) to keep
        
    Returns:
        torch.Tensor: Binary mask c_i
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

def get_mask_layer_constraint_new(saliency: List[torch.Tensor], pruning_factor: float, layer_constraint: float) -> List[torch.Tensor]:
    """
    Layer-Constrained Mask Generation
    
    Args:
        saliency (List[torch.Tensor]): Set of saliency measures S = {s_1, ..., s_L}
        pruning_factor (float): Target sparsity
        layer_constraint (float): For each layer, we have constraint on the number of ones in the corresponding mask.
        
    Returns:
        List[torch.Tensor]: Set of binary masks C = {c_1, ..., c_L}
        
    This algorithm ensures both global sparsity target and layer-wise constraint.
    """
    if not saliency or len(saliency) == 0:
        return []
    
    # Validate inputs
    if pruning_factor <= 0:
        return [torch.zeros_like(s) for s in saliency]
    if pruning_factor >= 1.0:
        return [torch.ones_like(s) for s in saliency]
    if layer_constraint <= 0:
        return [torch.zeros_like(s) for s in saliency]
    
    # Calculate total parameters and target
    total_params = sum(s.numel() for s in saliency)
    n_target = math.ceil(total_params * pruning_factor)  # Use round instead of int to avoid truncation bias
    
    # Calculate n_target for each layer based on layer_constraint
    n_target_per_layer = [math.ceil(s.numel() * layer_constraint) for s in saliency]  # Use round here too
    
    # Initialize outputs
    L = len(saliency)
    final_masks: List[Optional[torch.Tensor]] = [None] * L
    unassigned = list(range(L))  # A = [L] # initially no mask is assigned yet
    
    while unassigned:
        # Flatten saliencies from unassigned layers
        S = f_flatten(saliency, unassigned)
        
        # Compute global threshold
        u_th = f_th(S, n_target)
        
        # Apply global threshold to all layers.
        M = f_maskglobal(saliency, unassigned, u_th)
        
        const_flag = 0
        layers_to_remove = []
        
        # Check clayer-wise constraint
        for i in unassigned:
            mask_sum = M[i].sum().item()
            if mask_sum > n_target_per_layer[i]: # layer-constraint is violated               
                final_masks[i] = f_masklocal(saliency[i], n_target_per_layer[i]) # Apply local thresholding to keep up with n_target_per_layer[i]
                layers_to_remove.append(i)
                n_target -= n_target_per_layer[i] # Revise the global target
                const_flag = 1 # Flag here signals the constraint violation and need for an additional global masking.
        
        # Remove assigned layers from unassigned list
        for i in layers_to_remove:
            unassigned.remove(i)
        
        # If no constraint violations, use the globally constructed masks for the remaining layers.
        if const_flag == 0:
            for i in unassigned:
                final_masks[i] = M[i]
            break
    
    
    # Convert to proper type for return
    #result_masks: List[torch.Tensor] = [mask for mask in final_masks if mask is not None]
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


def iterative_pruning(ori_net: nn.Module, train_dataloader: DataLoader, device: torch.device, pruning_factor: float = 0.1,
                      prune_method: int = 3, num_steps: int = 10,
                      mode: str = 'exp', num_batches: int = 1, last_layer_constraint: float = -1, first_layer_constraint: float = -1,
                      layer_constraint: float = -1, keep_original_weights: bool = False) -> torch.Tensor:
    """
    Function to gradually remove weights from a network, recomputing the saliency at each step.
    pruning_factor: Fraction of remaining weights (globally) after pruning.
    prune_method: Which method to use to prune the layers. Refer to dict on top
    num_steps: Number of iterations to do when pruning progressively (should be >= 1).
    mode: Mode of choosing the sparsity decay schedule. One of 'exp', 'linear'
    num_batches: Number of batches to be used to approximate the gradients (should be -1 or >= 1).
                 When set to -1, uses the whole training set.
    last_layer_constraint: Constraint for last layer (maximum fraction to keep)
    first_layer_constraint: Constraint for first layer (maximum fraction to keep)
    layer_constraint: Maximum fraction of weights each layer can keep. If > 0, applies per-layer constraint.
                     If <= 0, uses global pruning without layer constraints.
    keep_original_weights: Whether to keep original weights for FORCE method.
    Returns a list of binary tensors which correspond to the final pruning mask.
    """
    # Let's create a copy of the network to make sure we don't affect the training later
    net = copy.deepcopy(ori_net) # Copy the original network

    if keep_original_weights:
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
    for step,perc in enumerate(pruning_steps):
        saliency = []
        saliency = get_average_saliencies(net, train_dataloader, device,
                                          prune_method=prune_method,
                                          num_batches=num_batches,
                                          original_weights=original_weights) # Returns the saliency metric
        torch.cuda.empty_cache()

        # Make sure all saliencies of previously deleted weights is minimum so they do not
        # get picked again.
        if mask is not None and not keep_original_weights:
            min_saliency = get_minimum_saliency(saliency)
            # Use a very negative value to ensure pruned weights stay pruned
            for ii in range(len(saliency)):
                saliency[ii][mask[ii] == 0.] = min_saliency - 1e6

        if hook_handlers is not None:
            for h in hook_handlers:
                h.remove()
        
        # Use layer constraint if specified, otherwise use global pruning
        if layer_constraint > 0 and (first_layer_constraint > 0 or last_layer_constraint > 0):
            print("Warning: Both layer_constraint and first/last layer constraints are set. Using layer_constraint only.")

        print(layer_constraint, first_layer_constraint, last_layer_constraint)
        if layer_constraint > 0:
            layer_constraint_adjusted = min(1.0, constraint_scale * perc)
            mask = get_mask_layer_constraint_new(saliency, perc, layer_constraint_adjusted)
        elif first_layer_constraint > 0 or last_layer_constraint > 0:
            first_layer_constraint_adjusted = min(1.0, constraint_scale_first * perc) if first_layer_constraint > 0 else -1
            last_layer_constraint_adjusted = min(1.0, constraint_scale_last * perc) if last_layer_constraint > 0 else -1
            print(f"Step {step+1}/{num_steps}: First layer constraint adjusted to {first_layer_constraint_adjusted}, Last layer constraint adjusted to {last_layer_constraint_adjusted}")
            mask = get_mask_layer_constraint_first_last(saliency, perc, first_layer_constraint_adjusted, last_layer_constraint_adjusted)
        else:
            mask = get_mask(saliency, perc)


        if keep_original_weights:
            net = copy.deepcopy(ori_net)
            apply_prune_mask(net, mask, apply_hooks=False)
        else:
            hook_handlers = apply_prune_mask(net, mask, apply_hooks=True)

        # Clean up intermediate objects to save memory
        if 'saliency' in locals():
            del saliency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (step+1) % modulo ==0 and step != 0:
            p = check_global_pruning(mask)
            print(f'Global pruning {round(float(p),5)}','{}% Pruning complete'.format((step+1)/num_steps *100))
    
    # No need for get_mask4 since layer constraints are already applied in the masking functions
    # Safety check: ensure mask is valid
    if mask is None or not mask:
        print(f"Warning: mask is None or empty, creating sparse masks with pruning_factor={pruning_factor}")
        # Create proper sparse masks instead of all-ones
        mask = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer_weights = layer.weight.flatten()
                num_keep = max(1, int(layer_weights.numel() * pruning_factor))  # Keep at least 1 weight
                if num_keep >= layer_weights.numel():
                    # Keep all weights in this layer
                    layer_mask = torch.ones_like(layer.weight)
                else:
                    # Keep only the top weights
                    _, indices = torch.topk(torch.abs(layer_weights), num_keep)
                    layer_mask = torch.zeros_like(layer_weights)
                    layer_mask[indices] = 1
                    layer_mask = layer_mask.view_as(layer.weight)
                mask.append(layer_mask)

    # Create full masks for all layers, critical layers included by default
    c = 0
    mask_full = []
    for layer in net.modules():
        if isinstance(layer,nn.BatchNorm2d):
            mask_full.append(torch.ones_like(layer.weight))
            mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer,nn.Linear):
            mask_full.append(mask[c])
            c += 1
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer,nn.Conv2d):
            mask_full.append(mask[c])
            c += 1
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
    flat_mask = torch.cat([torch.flatten(x) for x in mask_full])
    
    # Comprehensive cleanup before returning - remove all hooks and free GPU memory
    if 'hook_handlers' in locals() and hook_handlers is not None:
        for h in hook_handlers:
            try:
                h.remove()
            except:
                pass
        del hook_handlers
    
    # Clean up large objects
    if 'net' in locals():
        del net
    if 'original_weights' in locals() and original_weights is not None:
        del original_weights
    if 'mask' in locals() and isinstance(mask, list):
        del mask
    if 'mask_full' in locals():
        del mask_full
    if 'saliency' in locals():
        del saliency
    
    # Force CUDA cleanup with additional steps to prevent persistent PIDs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Try additional CUDA cleanup
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
        except:
            pass
    
    # Multiple garbage collections to ensure everything is cleaned up
    gc.collect()
    gc.collect()  # Sometimes multiple calls are needed
    
    # Additional process-level cleanup attempt
    try:
        import psutil
        import os
        current_process = psutil.Process(os.getpid())
        # Trigger memory cleanup at process level
        _ = current_process.memory_info()
    except ImportError:
        # psutil not available, use alternative cleanup
        try:
            import os
            # Force process memory sync
            os.sync() if hasattr(os, 'sync') else None
        except:
            pass
    except:
        pass
    
    # Final cleanup attempt - clear all remaining CUDA state
    if torch.cuda.is_available():
        try:
            # Clear all CUDA caches one more time
            torch.cuda.empty_cache()
            # Try to reset memory stats
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    # Final aggressive cleanup to prevent persistent processes
    cleanup_gpu_memory()
    
    return flat_mask


def check_global_pruning(mask: List[torch.Tensor]) -> float:
    "Compute fraction of unpruned weights in a mask"
    flattened_mask = torch.cat([torch.flatten(x) for x in mask])
    return flattened_mask.mean().item()


def get_minimum_saliency(saliency: List[torch.Tensor]) -> float:
    "Compute minimum value of saliency globally"
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.min().item()


def numeric_fix(tensor: torch.Tensor) -> torch.Tensor:
    "Replace inf values in a tensor with a large finite number."
    tensor[torch.isinf(tensor)] = 1e9 
    tensor[torch.isnan(tensor)] = 0.0  # Replace NaN with zero
    return tensor
