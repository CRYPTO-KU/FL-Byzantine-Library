import torch
import torch.nn as nn
import numpy as np
import copy
from utils import *


def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP (original implementation).
    This is the core LAMP algorithm from the paper.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    
    return new_scores.view(scores.shape)


def _get_weights(model):
    """Get weight tensors from prunable layers"""
    weights = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weights.append(m.weight)
    return weights


def _get_modules(model):
    """Get prunable modules"""
    modules = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            modules.append(m)
    return modules


def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    For models without explicit masks, assume all weights are unmasked.
    """
    modules = _get_modules(model)
    unmaskeds = []
    for m in modules:
        # Check if layer has a weight_mask attribute (from torch.nn.utils.prune)
        if hasattr(m, 'weight_mask'):
            unmaskeds.append(m.weight_mask.sum().item())
        else:
            unmaskeds.append(m.weight.numel())
    return torch.FloatTensor(unmaskeds)


def _compute_lamp_amounts(model, amount):
    """
    Compute layer-wise pruning amounts using the original LAMP algorithm.
    This is the core implementation from the original paper.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))
    
    # Compute normalized scores for each layer (w^2 normalized)
    flattened_scores = [_normalize_scores(w**2).view(-1) for w in _get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_surv)
    threshold = topks[-1]
    
    # Count how many weights survive in each layer
    final_survs = [torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() 
                   for score in flattened_scores]
    
    # Compute pruning amounts
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmaskeds[idx]))
    
    return amounts


class LayerAdaptiveMagnitudePruning:
    """
    Implementation of Layer-adaptive Sparsity for the Magnitude-based Pruning (LAMP)
    
    This is the corrected implementation following the original paper:
    "Layerwise Sparsity for Magnitude-based Pruning" (ICLR 2021)
    """

    def __init__(self, target_sparsity=0.9):
        """
        Initialize LAMP pruning
        
        Args:
            target_sparsity: Global target sparsity level
        """
        self.target_sparsity = target_sparsity
        
        # Statistics tracking
        self.layer_stats = []
        self.layer_sparsities = []
        
    def compute_layer_statistics(self, net):
        """
        Compute statistics for each layer
        
        Returns:
            layer_stats: List of dictionaries containing layer statistics
        """
        layer_stats = []
        
        for name, layer in net.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights = layer.weight.data
                
                # Basic statistics
                magnitude = torch.abs(weights)
                stats = {
                    'name': name,
                    'type': 'conv' if isinstance(layer, nn.Conv2d) else 'linear',
                    'shape': weights.shape,
                    'num_params': weights.numel(),
                    'magnitude_mean': magnitude.mean().item(),
                    'magnitude_std': magnitude.std().item(),
                    'magnitude_min': magnitude.min().item(),
                    'magnitude_max': magnitude.max().item(),
                }
                
                layer_stats.append(stats)
        
        self.layer_stats = layer_stats
        return layer_stats

    def generate_lamp_masks(self, net):
        """
        Generate pruning masks using the original LAMP algorithm
        
        Args:
            net: Neural network to prune
            
        Returns:
            masks: List of binary masks for each layer
        """
        # Compute layer statistics for tracking
        layer_stats = self.compute_layer_statistics(net)
        
        # Use the original LAMP algorithm to compute layer-wise amounts
        amounts = _compute_lamp_amounts(net, self.target_sparsity)
        
        print(f"[LAMP] Target global sparsity: {self.target_sparsity:.3f}")
        print(f"[LAMP] Layer sparsities: {[f'{a:.3f}' for a in amounts]}")
        
        # Generate masks based on computed amounts
        masks = []
        layer_idx = 0
        
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights = layer.weight.data
                target_amount = amounts[layer_idx]
                
                # Magnitude-based pruning within each layer
                magnitude = torch.abs(weights)
                flat_magnitude = magnitude.flatten()
                
                num_weights = len(flat_magnitude)
                num_keep = max(1, int(num_weights * (1 - target_amount)))
                
                if num_keep < num_weights:
                    threshold, _ = torch.topk(flat_magnitude, num_keep, sorted=True)
                    acceptable_magnitude = threshold[-1]
                    mask = (magnitude >= acceptable_magnitude).float()
                else:
                    mask = torch.ones_like(magnitude)
                
                masks.append(mask)
                
                # Log layer pruning info
                actual_sparsity = 1 - mask.float().mean().item()
                print(f"[LAMP] Layer {layer_idx} ({layer_stats[layer_idx]['name']}): "
                      f"target={target_amount:.3f}, actual={actual_sparsity:.3f}")
                
                layer_idx += 1
        
        # Store actual sparsities
        self.layer_sparsities = [1 - mask.float().mean().item() for mask in masks]
        
        # Compute actual global sparsity
        total_weights = sum(mask.numel() for mask in masks)
        remaining_weights = sum(mask.sum().item() for mask in masks)
        actual_global_sparsity = 1 - (remaining_weights / total_weights)
        print(f"[LAMP] Actual global sparsity: {actual_global_sparsity:.3f}")
        
        return masks

    def get_lamp_summary(self):
        """Get summary of LAMP pruning decisions"""
        if not self.layer_stats:
            return "No LAMP pruning performed yet"
        
        summary = {
            'target_sparsity': self.target_sparsity,
            'num_layers': len(self.layer_stats),
            'layer_sparsities': self.layer_sparsities,
            'layer_stats': self.layer_stats,
        }
        
        return summary


def get_lamp_mask(args, net, loader, device):
    """
    Main interface function for LAMP compatible with existing codebase
    
    Args:
        args: Arguments containing LAMP parameters
        net: Neural network to prune
        loader: Training data loader (not used in LAMP, kept for compatibility)
        device: Device to run on
        
    Returns:
        Flattened binary mask tensor
    """
    # Initialize LAMP pruner
    lamp = LayerAdaptiveMagnitudePruning(
        target_sparsity=1 - args.pruning_factor  # Convert to sparsity
    )
    
    # Generate LAMP masks
    masks = lamp.generate_lamp_masks(net)
    
    # Convert to full network mask format (compatible with your existing code)
    mask_full = []
    mask_idx = 0
    
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            # BatchNorm layers get zero masks for weights and biases
            mask_full.append(torch.zeros_like(layer.weight))
            mask_full.append(torch.zeros_like(layer.bias))
        elif isinstance(layer, nn.Linear):
            mask_full.append(masks[mask_idx])
            mask_idx += 1
            if layer.bias is not None:
                mask_full.append(torch.zeros_like(layer.bias))
        elif isinstance(layer, nn.Conv2d):
            mask_full.append(masks[mask_idx])
            mask_idx += 1
            if layer.bias is not None:
                mask_full.append(torch.zeros_like(layer.bias))
    
    # Flatten and return
    flat_mask = torch.cat([torch.flatten(x) for x in mask_full])
    return flat_mask


def iterative_lamp_pruning(net, target_sparsity, num_steps=10, mode='exp'):
    """
    Iterative LAMP pruning with adaptive layer sparsities at each step
    
    Args:
        net: Neural network to prune
        target_sparsity: Final target sparsity
        num_steps: Number of iterative pruning steps
        mode: Pruning schedule ('exp' or 'linear')
        
    Returns:
        Final flattened mask
    """
    net_copy = copy.deepcopy(net)
    
    # Create pruning schedule
    if mode == 'linear':
        sparsity_steps = [target_sparsity * (i + 1) / num_steps for i in range(num_steps)]
    elif mode == 'exp':
        sparsity_steps = [1 - (1 - target_sparsity) ** ((i + 1) / num_steps) for i in range(num_steps)]
    else:
        raise ValueError(f"Unknown pruning mode: {mode}")
    
    print(f"[Iterative LAMP] Sparsity schedule: {[f'{s:.3f}' for s in sparsity_steps]}")
    
    # Apply masks iteratively
    for step, step_sparsity in enumerate(sparsity_steps):
        print(f"\n[Iterative LAMP] Step {step + 1}/{num_steps}, target sparsity: {step_sparsity:.3f}")
        
        # Initialize LAMP for this step
        lamp = LayerAdaptiveMagnitudePruning(target_sparsity=step_sparsity)
        
        # Generate masks
        masks = lamp.generate_lamp_masks(net_copy)
        
        # Apply masks to network (zero out pruned weights)
        mask_idx = 0
        for layer in net_copy.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight.data *= masks[mask_idx]
                mask_idx += 1
    
    # Generate final mask format
    mask_full = []
    mask_idx = 0
    
    for layer in net_copy.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_full.append(torch.zeros_like(layer.weight))
            mask_full.append(torch.zeros_like(layer.bias))
        elif isinstance(layer, nn.Linear):
            # Create mask based on current weights (non-zero = 1, zero = 0)
            weight_mask = (layer.weight.data != 0).float()
            mask_full.append(weight_mask)
            mask_idx += 1
            if layer.bias is not None:
                mask_full.append(torch.zeros_like(layer.bias))
        elif isinstance(layer, nn.Conv2d):
            weight_mask = (layer.weight.data != 0).float()
            mask_full.append(weight_mask)
            mask_idx += 1
            if layer.bias is not None:
                mask_full.append(torch.zeros_like(layer.bias))
    
    # Flatten and return
    flat_mask = torch.cat([torch.flatten(x) for x in mask_full])
    return flat_mask


# Utility functions for analysis

def analyze_lamp_distribution(net, target_sparsity=0.9):
    """
    Analyze how LAMP would distribute sparsity across layers
    
    Args:
        net: Neural network
        target_sparsity: Target sparsity to analyze
        
    Returns:
        Analysis results
    """
    lamp = LayerAdaptiveMagnitudePruning(target_sparsity=target_sparsity)
    layer_stats = lamp.compute_layer_statistics(net)
    amounts = _compute_lamp_amounts(net, target_sparsity)
    
    analysis = {
        'layer_names': [stats['name'] for stats in layer_stats],
        'layer_types': [stats['type'] for stats in layer_stats],
        'layer_params': [stats['num_params'] for stats in layer_stats],
        'layer_sparsities': amounts,
        'magnitude_stats': {
            'means': [stats['magnitude_mean'] for stats in layer_stats],
            'stds': [stats['magnitude_std'] for stats in layer_stats],
        }
    }
    
    return analysis
