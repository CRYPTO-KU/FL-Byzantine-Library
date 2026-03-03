import torch
import torch.nn as nn
import numpy as np
import copy
from utils import *


class EarlyStructuralPruning:
    """
    Implementation of Early Structural Pruning (EPI) from NVIDIA paper:
    "When to Prune? A Policy towards Early Structural Pruning"
    
    This class implements the policy-based approach to determine optimal pruning timing
    during training to achieve better performance with reduced computational cost.
    """
    
    def __init__(self, target_sparsity=0.9, patience=5, min_delta=0.001, 
                 warmup_epochs=10, pruning_frequency=5, structured=False, immediate_pruning=False,
                 num_steps=10, mode='exp'):
        """
        Initialize Early Structural Pruning
        
        Args:
            target_sparsity: Final target sparsity level (0.9 = 90% sparse)
            patience: Number of epochs to wait for improvement before pruning
            min_delta: Minimum change in loss to be considered as improvement
            warmup_epochs: Number of epochs before starting pruning consideration
            pruning_frequency: How often to check for pruning opportunities
            structured: Whether to use structured (channel-wise) or unstructured pruning
            immediate_pruning: If True, apply target sparsity immediately instead of gradual schedule
            num_steps: Number of iterative pruning steps for exponential/linear schedule
            mode: Mode of choosing the sparsity decay schedule. One of 'exp', 'linear', 'adaptive'
        """
        self.target_sparsity = target_sparsity
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.pruning_frequency = pruning_frequency
        self.structured = structured
        self.immediate_pruning = immediate_pruning
        self.num_steps = num_steps
        self.mode = mode
        
        # State tracking
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_sparsity = 0.0
        self.pruning_history = []
        self.loss_history = []
        self.should_prune_next = False
        self.pruning_intervals = []
        
    def update_loss_history(self, current_loss, epoch):
        """Update loss history and determine if pruning should occur"""
        self.loss_history.append(current_loss)
        
        # Skip warmup period
        if epoch < self.warmup_epochs:
            return False
            
        # Check if we should evaluate pruning
        if epoch % self.pruning_frequency != 0:
            return False
            
        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        # Determine if we should prune
        should_prune = (self.epochs_without_improvement >= self.patience and 
                       self.current_sparsity < self.target_sparsity)
        
        if should_prune:
            self.epochs_without_improvement = 0  # Reset counter
            self.pruning_intervals.append(epoch)
            
        return should_prune
    
    def get_adaptive_sparsity(self, epoch, total_epochs):
        """
        Calculate adaptive sparsity based on training progress and loss plateau
        Supports multiple scheduling modes: exponential, linear, and adaptive
        """
        # If immediate pruning is enabled, return target sparsity directly
        if self.immediate_pruning:
            return self.target_sparsity
        
        # Handle exponential and linear schedules similar to FORCE
        if self.mode == 'linear':
            # Linear decay schedule
            progress = min(epoch / total_epochs, 1.0)
            sparsity_steps = [1 - ((x + 1) * (1 - self.target_sparsity) / self.num_steps) for x in range(self.num_steps)]
            step_index = int(progress * (self.num_steps - 1))
            base_sparsity = sparsity_steps[step_index]
            
        elif self.mode == 'exp':
            # Exponential decay schedule (similar to FORCE)
            progress = min(epoch / total_epochs, 1.0)
            sparsity_steps = [np.exp(0 - ((x + 1) * (0 - np.log(self.target_sparsity)) / self.num_steps)) for x in range(self.num_steps)]
            step_index = int(progress * (self.num_steps - 1))
            base_sparsity = sparsity_steps[step_index]
            
        elif self.mode == 'adaptive':
            # Original adaptive schedule based on polynomial decay
            progress = min(epoch / total_epochs, 1.0)
            base_sparsity = self.target_sparsity * (progress ** 3)
        else:
            # Default to exponential if mode is not recognized
            progress = min(epoch / total_epochs, 1.0)
            sparsity_steps = [np.exp(0 - ((x + 1) * (0 - np.log(self.target_sparsity)) / self.num_steps)) for x in range(self.num_steps)]
            step_index = int(progress * (self.num_steps - 1))
            base_sparsity = sparsity_steps[step_index]
            
        # Apply adaptive component for 'adaptive' mode only
        if self.mode == 'adaptive' and len(self.loss_history) >= self.patience:
            recent_losses = self.loss_history[-self.patience:]
            loss_variance = np.var(recent_losses)
            
            # If loss is plateauing (low variance), be more aggressive
            if loss_variance < self.min_delta:
                plateau_factor = 1.2
            else:
                plateau_factor = 1.0
                
            adaptive_sparsity = min(base_sparsity * plateau_factor, self.target_sparsity)
        else:
            adaptive_sparsity = base_sparsity
            
        return adaptive_sparsity
    
    def compute_weight_importance(self, net, train_dataloader, device, num_batches=5):
        """
        Compute weight importance scores using gradient-based metrics
        Similar to the saliency computation in your existing code
        """
        # Copy network to avoid affecting original
        net_copy = copy.deepcopy(net)
        
        # Compute gradients
        gradients = []
        gradient_vars = []
        
        for layer in net_copy.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gradients.append(0)
                gradient_vars.append([])
        
        count_batch = 0
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            if count_batch >= num_batches and num_batches > 0:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            net_copy.zero_grad()
            outputs = net_copy(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            counter = 0
            for layer in net_copy.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if layer.weight.grad is not None:
                        gradients[counter] += layer.weight.grad
                        gradient_vars[counter].append(layer.weight.grad.clone())
                    counter += 1
            count_batch += 1
        
        # Average gradients
        avg_gradients = [g / count_batch for g in gradients]
        
        # Compute gradient variance
        variance = []
        for layer_grads in gradient_vars:
            if len(layer_grads) > 1:
                stacked = torch.stack([g.flatten() for g in layer_grads], dim=1)
                layer_var = torch.var(stacked, dim=1).view(layer_grads[0].shape)
            else:
                layer_var = torch.zeros_like(layer_grads[0])
            variance.append(layer_var)
        
        # Compute importance scores (weight * gradient * variance)
        importance_scores = []
        idx = 0
        for layer in net_copy.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weight = layer.weight
                grad = avg_gradients[idx]
                var = variance[idx]
                
                # EPI-specific importance: weight magnitude * gradient * variance
                importance = torch.abs(weight * grad * var)
                importance_scores.append(importance)
                idx += 1
                
        return importance_scores
    
    def get_structured_mask(self, importance_scores, sparsity_level):
        """
        Generate structured pruning mask (channel-wise for conv, neuron-wise for linear)
        """
        masks = []
        
        for importance in importance_scores:
            if len(importance.shape) == 4:  # Conv2d
                # Channel-wise pruning: compute importance per output channel
                channel_importance = importance.sum(dim=(1, 2, 3))
                num_channels = len(channel_importance)
                num_keep = int(num_channels * (1 - sparsity_level))
                
                if num_keep > 0:
                    _, indices = torch.topk(channel_importance, num_keep)
                    mask = torch.zeros_like(importance)
                    mask[indices] = 1
                else:
                    mask = torch.zeros_like(importance)
                    
            elif len(importance.shape) == 2:  # Linear
                # Neuron-wise pruning: compute importance per output neuron
                neuron_importance = importance.sum(dim=1)
                num_neurons = len(neuron_importance)
                num_keep = int(num_neurons * (1 - sparsity_level))
                
                if num_keep > 0:
                    _, indices = torch.topk(neuron_importance, num_keep)
                    mask = torch.zeros_like(importance)
                    mask[indices] = 1
                else:
                    mask = torch.zeros_like(importance)
            else:
                # Fallback to unstructured
                flat_importance = importance.flatten()
                num_weights = len(flat_importance)
                num_keep = int(num_weights * (1 - sparsity_level))
                
                if num_keep > 0:
                    _, indices = torch.topk(flat_importance, num_keep)
                    mask = torch.zeros_like(flat_importance)
                    mask[indices] = 1
                    mask = mask.view(importance.shape)
                else:
                    mask = torch.zeros_like(importance)
                    
            masks.append(mask)
            
        return masks
    
    def get_unstructured_mask(self, importance_scores, sparsity_level):
        """
        Generate unstructured pruning mask using importance scores
        Compatible with your existing masking functions
        """
        return self._get_global_mask(importance_scores, sparsity_level)
    
    def _get_global_mask(self, importance_scores, sparsity_level):
        """Global unstructured masking"""
        # Flatten all importance scores
        all_scores = torch.cat([torch.flatten(score) for score in importance_scores])
        num_weights = len(all_scores)
        num_keep = int(num_weights * (1 - sparsity_level))
        
        if num_keep > 0:
            threshold, _ = torch.topk(all_scores, num_keep, sorted=True)
            acceptable_score = threshold[-1]
        else:
            acceptable_score = float('inf')
        
        masks = []
        for score in importance_scores:
            mask = (score >= acceptable_score).float()
            masks.append(mask)
            
        return masks

    def _apply_masks_to_network(self, net, masks):
        """
        Apply pruning masks to network weights
        """
        mask_idx = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if mask_idx < len(masks):
                    # Set masked weights to zero
                    layer.weight.data = layer.weight.data * masks[mask_idx]
                    mask_idx += 1

    def prune_network(self, net, train_dataloader, device, epoch, total_epochs):
        """
        Main pruning function that implements the EPI policy
        
        Returns:
            masks: List of pruning masks for each layer
            pruned: Boolean indicating if pruning was performed
        """
        # Determine target sparsity for this epoch
        target_sparsity = self.get_adaptive_sparsity(epoch, total_epochs)
        
        # Only prune if we need to increase sparsity
        if target_sparsity <= self.current_sparsity:
            return None, False
            
        print(f"[EPI] Pruning at epoch {epoch}, target sparsity: {target_sparsity:.3f}")
        
        # Compute weight importance
        importance_scores = self.compute_weight_importance(net, train_dataloader, device)
        
        # Generate masks based on pruning type
        if self.structured:
            masks = self.get_structured_mask(importance_scores, target_sparsity)
        else:
            masks = self.get_unstructured_mask(importance_scores, target_sparsity)
        
        # Update current sparsity
        total_weights = sum(mask.numel() for mask in masks)
        remaining_weights = sum(mask.sum().item() for mask in masks)
        self.current_sparsity = 1 - (remaining_weights / total_weights)
        
        # Record pruning event
        self.pruning_history.append({
            'epoch': epoch,
            'target_sparsity': target_sparsity,
            'actual_sparsity': self.current_sparsity,
            'num_pruned_weights': total_weights - remaining_weights
        })
        
        print(f"[EPI] Achieved sparsity: {self.current_sparsity:.3f}")
        
        return masks, True


def get_epi_mask_iterative(args, net, loader, device):
    """
    Iterative EPI pruning function similar to FORCE's iterative_pruning
    Performs gradual pruning over multiple steps using exponential/linear schedule
    
    Args:
        args: Arguments containing pruning parameters
              args.pruning_factor should be the fraction of weights to KEEP
              args.num_steps: Number of iterative pruning steps  
              args.mode: 'exp' for exponential, 'linear' for linear schedule
        net: Neural network to prune
        loader: Training data loader  
        device: Device to run on
        
    Returns:
        Flattened binary mask tensor
    """
    # Initialize EPI pruner for iterative pruning
    epi = EarlyStructuralPruning(
        target_sparsity=1.0 - args.pruning_factor,  # Convert density to sparsity
        patience=getattr(args, 'epi_patience', 5),
        min_delta=getattr(args, 'epi_min_delta', 0.001),
        warmup_epochs=getattr(args, 'epi_warmup', 10), 
        pruning_frequency=getattr(args, 'epi_frequency', 5),
        structured=getattr(args, 'epi_structured', False),
        immediate_pruning=False,  # Use gradual schedule for iterative pruning
        num_steps=getattr(args, 'num_steps', 10),
        mode=getattr(args, 'mode', 'exp')
    )
    
    # Copy network to avoid affecting original
    net_copy = copy.deepcopy(net)
    
    # Choose decay schedule similar to FORCE
    num_steps = getattr(args, 'num_steps', 10)
    mode = getattr(args, 'mode', 'exp')
    pruning_factor = args.pruning_factor
    
    if mode == 'linear':
        pruning_steps = [1 - ((x + 1) * (1 - pruning_factor) / num_steps) for x in range(num_steps)]
    elif mode == 'exp':
        pruning_steps = [np.exp(0 - ((x + 1) * (0 - np.log(pruning_factor)) / num_steps)) for x in range(num_steps)]
    else:
        # Default to exponential
        pruning_steps = [np.exp(0 - ((x + 1) * (0 - np.log(pruning_factor)) / num_steps)) for x in range(num_steps)]
    
    masks = None
    
    # Iterative pruning loop
    for step, sparsity_target in enumerate(pruning_steps):
        current_sparsity = 1.0 - sparsity_target
        
        # Compute importance scores
        importance_scores = epi.compute_weight_importance(net_copy, loader, device, 
                                                        num_batches=getattr(args, 'num_batches', 5))
        
        # Generate masks 
        if getattr(args, 'epi_structured', False):
            masks = epi.get_structured_mask(importance_scores, current_sparsity)
        else:
            masks = epi.get_unstructured_mask(importance_scores, current_sparsity)
            
        # Apply masks to network for next iteration
        epi._apply_masks_to_network(net_copy, masks)
        
        if (step + 1) % max(1, num_steps // 5) == 0:
            total_weights = sum(mask.numel() for mask in masks)
            remaining_weights = sum(mask.sum().item() for mask in masks) 
            actual_sparsity = 1 - (remaining_weights / total_weights)
            print(f'[EPI Iterative] Step {step+1}/{num_steps}, Target: {current_sparsity:.3f}, '
                  f'Actual: {actual_sparsity:.3f}')
    
    if masks is None:
        # Return all-ones mask (no pruning)
        total_params = sum(p.numel() for p in net.parameters())
        return torch.ones(total_params, device=device)
    
    # Convert layer masks to full network mask (compatible with existing format)
    mask_full = []
    mask_idx = 0
    
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
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


def get_epi_mask(args, net, loader, device):
    """
    Main interface function compatible with your existing codebase
    
    Args:
        args: Arguments containing EPI parameters
              args.pruning_factor should be the fraction of weights to KEEP (e.g., 0.01 = keep 1% of weights)
        net: Neural network to prune
        loader: Training data loader
        device: Device to run on
        
    Returns:
        Flattened binary mask tensor
        
    Note:
        This function applies the target sparsity immediately (not gradually over epochs)
        to be compatible with single-shot pruning workflows.
    """
    # Initialize EPI pruner
    epi = EarlyStructuralPruning(
        target_sparsity=1.0 - args.pruning_factor,  # Convert density to sparsity
        patience=getattr(args, 'epi_patience', 5),
        min_delta=getattr(args, 'epi_min_delta', 0.001),
        warmup_epochs=getattr(args, 'epi_warmup', 10),
        pruning_frequency=getattr(args, 'epi_frequency', 5),
        structured=getattr(args, 'epi_structured', False),
        immediate_pruning=False,  # Apply target sparsity immediately for mask generation
        num_steps=getattr(args, 'num_steps', 10),
        mode=getattr(args, 'mode', 'exp')  # Default to exponential schedule
    )

    # For mask generation, epoch doesn't matter since immediate_pruning=False
    epoch = getattr(args, 'current_epoch', 50)
    total_epochs = getattr(args, 'total_epochs', 100)
    
    # Get masks
    masks, pruned = epi.prune_network(net, loader, device, epoch, total_epochs)
    
    if not pruned or masks is None:
        # Return all-ones mask (no pruning)
        total_params = sum(p.numel() for p in net.parameters())
        return torch.ones(total_params, device=device)
    
    # Convert layer masks to full network mask (compatible with your format)
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


# Additional utility functions for integration

def epi_training_step(epi_pruner, net, train_loader, device, epoch, total_epochs, current_loss):
    """
    Function to be called during training to update EPI state and potentially prune
    
    Args:
        epi_pruner: EarlyStructuralPruning instance
        net: Current network
        train_loader: Training data loader
        device: Device
        epoch: Current epoch
        total_epochs: Total training epochs
        current_loss: Current training/validation loss
        
    Returns:
        masks: Pruning masks if pruning occurred, None otherwise
        pruned: Boolean indicating if pruning was performed
    """
    # Update loss history and check if pruning should occur
    should_prune = epi_pruner.update_loss_history(current_loss, epoch)
    
    if should_prune:
        return epi_pruner.prune_network(net, train_loader, device, epoch, total_epochs)
    else:
        return None, False


def get_epi_summary(epi_pruner):
    """Get summary of EPI pruning decisions and timeline"""
    summary = {
        'total_pruning_events': len(epi_pruner.pruning_history),
        'final_sparsity': epi_pruner.current_sparsity,
        'target_sparsity': epi_pruner.target_sparsity,
        'pruning_intervals': epi_pruner.pruning_intervals,
        'pruning_history': epi_pruner.pruning_history
    }
    return summary
