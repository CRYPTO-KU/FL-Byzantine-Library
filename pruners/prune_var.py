import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import copy

class VariancePruner:
    """
    Variance-based pruning that tracks parameter variance across multiple iterations
    and generates masks based on locations with highest variance.
    """
    
    def __init__(self, model, loaders, pruning_scale, max_threshold, iterations=10):
        """
        Initialize variance pruner.
        
        Args:
            model: PyTorch model to prune
            loaders: List of data loaders for variance calculation
            pruning_scale: Fraction of parameters to prune (0.0 to 1.0)
            max_threshold: Maximum fraction of parameters to prune from any single layer
            iterations: Number of iterations to track variance (T)
        """
        self.model = model
        self.loaders = loaders
        self.pruning_scale = pruning_scale
        self.max_threshold = max_threshold
        self.iterations = iterations
        self.variance_counts = defaultdict(lambda: torch.zeros_like(next(iter(model.parameters()))))
        self.device = next(model.parameters()).device
        
    def _get_parameter_dict(self):
        """Get dictionary of model parameters."""
        param_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:  # Only consider weight matrices
                param_dict[name] = param
        return param_dict
    
    def _calculate_variance_for_iteration(self, loader):
        """Calculate parameter variance for one iteration using given loader."""
        self.model.train()  # Set to training mode for weight updates
        param_dict = self._get_parameter_dict()
        param_gradients = {name: [] for name in param_dict.keys()}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Collect gradients and update weights
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= 10:  # Limit batches per iteration for efficiency
                break
                
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Store gradients before updating weights
            for name, param in param_dict.items():
                if param.grad is not None:
                    param_gradients[name].append(param.grad.clone().detach())
            
            # Update weights
            optimizer.step()
        
        # Calculate variance for each parameter
        param_variances = {}
        for name, grads in param_gradients.items():
            if grads:
                grad_stack = torch.stack(grads)
                variance = torch.var(grad_stack, dim=0)
                param_variances[name] = variance
            else:
                param_variances[name] = torch.zeros_like(param_dict[name])
                
        return param_variances
    
    def _update_variance_counts(self, param_variances):
        """Update variance counts by adding +1 to top variance locations."""
        for name, variance in param_variances.items():
            # Find top 10% variance locations for this iteration
            flat_variance = variance.flatten()
            k = max(1, int(0.1 * len(flat_variance)))
            _, top_indices = torch.topk(flat_variance, k)
            
            # Convert back to original shape and add +1 to top locations
            mask = torch.zeros_like(flat_variance)
            mask[top_indices] = 1.0
            mask = mask.reshape(variance.shape)
            
            if name not in self.variance_counts:
                self.variance_counts[name] = torch.zeros_like(variance)
            self.variance_counts[name] += mask
    
    def _generate_layer_masks(self):
        """Generate masks for each layer based on variance counts."""
        param_dict = self._get_parameter_dict()
        layer_masks = {}
        
        # Calculate total parameters and target pruning amount
        total_params = sum(param.numel() for param in param_dict.values())
        target_pruned = int(total_params * self.pruning_scale)
        
        # Sort layers by total variance count (priority for pruning)
        layer_priorities = {}
        for name, counts in self.variance_counts.items():
            layer_priorities[name] = torch.sum(counts).item()
        
        # Distribute pruning across layers with max threshold constraint
        layer_prune_counts = {}
        remaining_to_prune = target_pruned
        
        for name in sorted(layer_priorities.keys(), key=lambda x: layer_priorities[x], reverse=True):
            param = param_dict[name]
            layer_size = param.numel()
            max_prune_this_layer = int(layer_size * self.max_threshold)
            
            # Determine how much to prune from this layer
            prune_this_layer = min(max_prune_this_layer, remaining_to_prune)
            layer_prune_counts[name] = prune_this_layer
            remaining_to_prune -= prune_this_layer
            
            if remaining_to_prune <= 0:
                break
        
        # Generate masks based on top-k variance counts for each layer
        for name, param in param_dict.items():
            if name in layer_prune_counts and layer_prune_counts[name] > 0:
                counts = self.variance_counts[name]
                flat_counts = counts.flatten()
                k = layer_prune_counts[name]
                
                if k > 0:
                    _, top_indices = torch.topk(flat_counts, min(k, len(flat_counts)))
                    mask = torch.ones_like(flat_counts)
                    mask[top_indices] = 0.0  # 0 means pruned
                    mask = mask.reshape(param.shape)
                else:
                    mask = torch.ones_like(param)
            else:
                mask = torch.ones_like(param)
            
            layer_masks[name] = mask
            
        return layer_masks
    
    def generate_mask(self):
        """
        Main method to generate pruning mask based on variance tracking.
        
        Returns:
            dict: Dictionary mapping parameter names to binary masks
        """
        print(f"Starting variance-based pruning with {self.iterations} iterations...")
        
        # Reset variance counts
        self.variance_counts.clear()
        
        # Track variance over T iterations
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            
            # Use different loaders in round-robin fashion
            loader_idx = iteration % len(self.loaders)
            current_loader = self.loaders[loader_idx]
            
            # Calculate variance for this iteration
            param_variances = self._calculate_variance_for_iteration(current_loader)
            
            # Update variance counts
            self._update_variance_counts(param_variances)
        
        # Generate final masks based on accumulated variance counts
        masks = self._generate_layer_masks()
        
        # Print pruning statistics
        total_params = 0
        pruned_params = 0
        for name, mask in masks.items():
            layer_total = mask.numel()
            layer_pruned = (mask == 0).sum().item()
            total_params += layer_total
            pruned_params += layer_pruned
            print(f"Layer {name}: {layer_pruned}/{layer_total} "
                  f"({100*layer_pruned/layer_total:.2f}%) pruned")
        
        print(f"Total pruning: {pruned_params}/{total_params} "
              f"({100*pruned_params/total_params:.2f}%)")
        
        return masks
    
    def apply_mask(self, masks):
        """Apply generated masks to model parameters."""
        param_dict = self._get_parameter_dict()
        for name, mask in masks.items():
            if name in param_dict:
                param_dict[name].data *= mask.to(param_dict[name].device)


def create_variance_pruner(model, loaders, pruning_scale, max_threshold, iterations=10):
    """
    Factory function to create a variance pruner.
    
    Args:
        model: PyTorch model to prune
        loaders: List of data loaders for variance calculation
        pruning_scale: Fraction of parameters to prune (0.0 to 1.0)
        max_threshold: Maximum fraction of parameters to prune from any single layer
        iterations: Number of iterations to track variance (T)
    
    Returns:
        VariancePruner: Configured variance pruner instance
    """
    return VariancePruner(model, loaders, pruning_scale, max_threshold, iterations)