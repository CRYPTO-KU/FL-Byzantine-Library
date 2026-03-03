import torch
import torch.nn as nn
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from copy import deepcopy


def get_mask_basic(args,net_ps,loader,device):
    target_sparsity = 1-args.pruning_factor
    num_iterations = args.num_steps
    # Convert single loader to list as expected by the function
    loaders = [loader] if not isinstance(loader, list) else loader
    mask = get_gradient_statistics_mask(net_ps,loaders,device,target_sparsity,num_iterations)
    return mask

# Try to import from parent utils, provide fallback if not available
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_grad_flattened, count_parameters
except ImportError:
    print("Warning: Could not import from utils, using fallback implementations")
    
    def get_grad_flattened(model: nn.Module, device: torch.device) -> torch.Tensor:
        """Fallback implementation to get flattened gradients."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros_like(param).view(-1))
        return torch.cat(grads).to(device)
    
    def count_parameters(model: nn.Module) -> int:
        """Fallback implementation to count parameters."""
        return sum(p.numel() for p in model.parameters())
    


class GradientStatisticsPruner:
    """
    Iterative network pruning algorithm that combines model weights and gradient statistics.
    
    This algorithm:
    1. Iteratively prunes the network in exponential sparsity steps
    2. Uses multiple data loaders to compute gradient statistics
    3. Combines weight magnitudes with gradient standard deviations for pruning decisions
    4. Returns locations of remaining weights after reaching target sparsity
    """
    
    def __init__(self, 
                 target_sparsity: float = 0.9,
                 num_iterations: int = 10,
                 weight_importance_ratio: float = 0.5,
                 exponential_base: float = 2.0,
                 importance_method: str = 'weight_std',
                 layer_collapse_threshold: float = 0.01,
                 enable_recovery: bool = True):
        """
        Initialize the pruning algorithm.
        
        Args:
            target_sparsity: Final sparsity level (0.9 = 90% pruned)
            num_iterations: Number of pruning iterations
            weight_importance_ratio: Balance between weights and gradients (0.5 = equal weight)
            exponential_base: Base for exponential sparsity progression
            importance_method: Method to compute importance ('weight_std' or 'grad_std')
                - 'weight_std': Combines weight magnitudes with gradient std
                - 'grad_std': Combines gradient magnitudes with gradient std
            layer_collapse_threshold: Minimum fraction of weights to keep per layer (0.05 = 5%)
            enable_recovery: Whether to enable gradient-based weight recovery
        """
        self.target_sparsity = target_sparsity
        self.num_iterations = num_iterations
        self.weight_ratio = weight_importance_ratio
        self.grad_ratio = 1.0 - weight_importance_ratio
        self.exp_base = exponential_base
        self.importance_method = importance_method
        self.layer_collapse_threshold = layer_collapse_threshold
        self.enable_recovery = enable_recovery
        
    def compute_gradient_statistics(self, model: nn.Module, loaders: List[DataLoader], device: torch.device):
        """
        Compute gradient statistics across multiple data loaders.
        
        Args:
            model: Neural network model
            loaders: List of data loaders
            device: Computing device
            
        Returns:
            tuple: (mean_gradients, std_gradients) as flattened tensors
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        all_gradients = []
        
        print(f"Computing gradient statistics from {len(loaders)} loaders...")
        
        for loader_idx, loader in enumerate(loaders):
            # Get a single batch from each loader
            try:
                batch_data, batch_labels = next(iter(loader))
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                # Forward pass
                model.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Extract and store gradients
                grad_flat = get_grad_flattened(model, device=device)
                all_gradients.append(grad_flat.clone())
                
            except StopIteration:
                print(f"Warning: Loader {loader_idx} is empty, skipping...")
                continue
        
        if not all_gradients:
            raise ValueError("No gradients computed - all loaders are empty!")
        
        # Stack gradients and compute statistics
        gradients_tensor = torch.stack(all_gradients)  # Shape: [num_loaders, num_params]
        
        mean_gradients = torch.mean(gradients_tensor, dim=0)
        std_gradients = torch.std(gradients_tensor, dim=0)
        
        print(f"Gradient statistics computed from {len(all_gradients)} batches")
        return mean_gradients, std_gradients
    
    def compute_importance_scores(self, weights: torch.Tensor, grad_mean: torch.Tensor, grad_std: torch.Tensor):
        """
        Compute importance scores using different methods.
        
        Args:
            weights: Flattened model weights
            grad_mean: Mean gradients across loaders
            grad_std: Standard deviation of gradients across loaders
            
        Returns:
            torch.Tensor: Importance scores for each parameter
        """
        if self.importance_method == 'weight_std':
            return self._compute_weight_std_importance(weights, grad_mean, grad_std)
        elif self.importance_method == 'grad_std':
            return self._compute_grad_std_importance(weights, grad_mean, grad_std)
        else:
            raise ValueError(f"Unknown importance method: {self.importance_method}")
    
    def _compute_weight_std_importance(self, weights: torch.Tensor, grad_mean: torch.Tensor, grad_std: torch.Tensor):
        """
        Original method: Combine weight magnitudes with gradient standard deviations.
        
        Args:
            weights: Flattened model weights
            grad_mean: Mean gradients across loaders
            grad_std: Standard deviation of gradients across loaders
            
        Returns:
            torch.Tensor: Importance scores for each parameter
        """
        # Normalize weight magnitudes
        weight_importance = torch.abs(weights)
        weight_importance = weight_importance / (weight_importance.max() + 1e-8)
        
        # Normalize gradient standard deviations (higher std = more important)
        grad_importance = grad_std / (grad_std.max() + 1e-8)
        
        # Combine importance scores
        combined_importance = (self.weight_ratio * weight_importance + 
                             self.grad_ratio * grad_importance)
        
        return combined_importance
    
    def _compute_grad_std_importance(self, weights: torch.Tensor, grad_mean: torch.Tensor, grad_std: torch.Tensor):
        """
        New method: Combine absolute gradient values with gradient standard deviations.
        
        Args:
            weights: Flattened model weights
            grad_mean: Mean gradients across loaders
            grad_std: Standard deviation of gradients across loaders
            
        Returns:
            torch.Tensor: Importance scores for each parameter
        """
        # Normalize absolute gradient magnitudes
        grad_magnitude_importance = torch.abs(grad_mean)
        grad_magnitude_importance = grad_magnitude_importance / (grad_magnitude_importance.max() + 1e-8)
        
        # Normalize gradient standard deviations (higher std = more important)
        grad_std_importance = grad_std / (grad_std.max() + 1e-8)
        
        # Combine importance scores
        # weight_ratio now represents grad_magnitude ratio, grad_ratio represents std ratio
        combined_importance = (self.weight_ratio * grad_magnitude_importance + 
                             self.grad_ratio * grad_std_importance)
        
        return combined_importance
    
    def _get_layer_boundaries(self, model: nn.Module):
        """
        Get the parameter boundaries for each layer in the flattened parameter vector.
        
        Args:
            model: Neural network model
            
        Returns:
            List[tuple]: List of (start_idx, end_idx, layer_name) for each layer
        """
        boundaries = []
        current_idx = 0
        
        for name, param in model.named_parameters():
            param_size = param.numel()
            boundaries.append((current_idx, current_idx + param_size, name))
            current_idx += param_size
        
        return boundaries
    
    def _prevent_layer_collapse(self, mask: torch.Tensor, importance_scores: torch.Tensor, 
                               model: nn.Module, grad_mean: torch.Tensor):
        """
        Prevent layer collapse by ensuring minimum weights per layer and gradient-based recovery.
        
        Args:
            mask: Current pruning mask
            importance_scores: Importance scores for all parameters
            model: Neural network model
            grad_mean: Mean gradients for recovery decisions
            
        Returns:
            torch.Tensor: Updated mask with layer collapse prevention
        """
        if not self.enable_recovery:
            return mask
        
        layer_boundaries = self._get_layer_boundaries(model)
        updated_mask = mask.clone()
        recovery_count = 0
        
        for start_idx, end_idx, layer_name in layer_boundaries:
            # Skip bias terms and batch norm parameters
            if 'bias' in layer_name or 'bn' in layer_name.lower():
                continue
                
            layer_mask = mask[start_idx:end_idx]
            layer_size = end_idx - start_idx
            current_kept = layer_mask.sum().item()
            min_kept = max(1, int(layer_size * self.layer_collapse_threshold))
            
            if current_kept < min_kept:
                # Layer is at risk of collapse, recover some weights
                needed = min_kept - current_kept
                
                # Get pruned weights (mask == 0) in this layer
                pruned_indices = (layer_mask == 0).nonzero(as_tuple=True)[0]
                
                if len(pruned_indices) > 0:
                    # Use gradient magnitude to decide which weights to recover
                    layer_grad_magnitudes = torch.abs(grad_mean[start_idx:end_idx])
                    pruned_grad_magnitudes = layer_grad_magnitudes[pruned_indices]
                    
                    # Select top gradient magnitude weights to recover
                    num_to_recover = min(needed, len(pruned_indices))
                    _, top_grad_indices = torch.topk(pruned_grad_magnitudes, int(num_to_recover))
                    
                    # Recover these weights
                    global_indices = start_idx + pruned_indices[top_grad_indices]
                    updated_mask[global_indices] = 1
                    recovery_count += num_to_recover
                    
                    print(f"Layer collapse prevention: Recovered {num_to_recover} weights in {layer_name}")
        
        if recovery_count > 0:
            print(f"Total weights recovered from collapse: {recovery_count}")
        
        return updated_mask
    
    def _adaptive_sparsity_adjustment(self, target_sparsity: float, mask: torch.Tensor, 
                                    total_params: int, model: nn.Module):
        """
        Adjust target sparsity if layer collapse prevention recovered too many weights.
        
        Args:
            target_sparsity: Original target sparsity
            mask: Current mask after collapse prevention
            total_params: Total number of parameters
            model: Neural network model
            
        Returns:
            float: Adjusted target sparsity
        """
        current_sparsity = 1 - (mask.sum().item() / total_params)
        
        if current_sparsity < target_sparsity:
            # We're below target sparsity due to recovery, adjust if possible
            difference = target_sparsity - current_sparsity
            print(f"Adjusted sparsity due to collapse prevention: {current_sparsity:.1%} "
                  f"(target was {target_sparsity:.1%})")
            return current_sparsity
        
        return target_sparsity

    def get_exponential_sparsity_schedule(self):
        """
        Generate exponential sparsity schedule.
        
        Returns:
            List[float]: Sparsity levels for each iteration
        """
        sparsities = []
        for i in range(self.num_iterations):
            # Exponential progression: starts slow, accelerates
            progress = (i + 1) / self.num_iterations
            exponential_progress = (self.exp_base ** progress - 1) / (self.exp_base - 1)
            sparsity = exponential_progress * self.target_sparsity
            sparsities.append(min(sparsity, self.target_sparsity))
        
        return sparsities
    
    def prune_network(self, model: nn.Module, loaders: List[DataLoader], device: torch.device):
        """
        Main pruning function that iteratively prunes the network.
        
        Args:
            model: Neural network model to prune
            loaders: List of data loaders for gradient computation
            device: Computing device
            
        Returns:
            torch.Tensor: Binary mask indicating remaining weights (1=keep, 0=prune)
        """
        print(f"Starting iterative pruning to {self.target_sparsity:.1%} sparsity...")
        print(f"Total parameters: {count_parameters(model):,}")
        
        # Initialize mask (all weights kept initially)
        total_params = count_parameters(model)
        current_mask = torch.ones(total_params, device=device)
        
        # Get sparsity schedule
        sparsity_schedule = self.get_exponential_sparsity_schedule()
        
        for iteration in range(self.num_iterations):
            target_sparsity = sparsity_schedule[iteration]
            
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            print(f"Target sparsity: {target_sparsity:.1%}")
            print(f"Using importance method: {self.importance_method}")
            
            # Compute gradient statistics
            mean_grads, std_grads = self.compute_gradient_statistics(model, loaders, device)
            
            # Get current weights (flattened)
            weights_flat = torch.cat([p.data.flatten() for p in model.parameters()])
            
            # Compute importance scores
            importance_scores = self.compute_importance_scores(weights_flat, mean_grads, std_grads)
            
            # Apply current mask to importance scores
            masked_importance = importance_scores * current_mask
            
            # Determine number of weights to keep
            num_params_to_keep = int(total_params * (1 - target_sparsity))
            
            # Get top-k most important parameters
            if num_params_to_keep > 0:
                _, top_indices = torch.topk(masked_importance, num_params_to_keep)
                new_mask = torch.zeros_like(current_mask)
                new_mask[top_indices] = 1
            else:
                new_mask = torch.zeros_like(current_mask)
            
            # Apply layer collapse prevention
            if self.enable_recovery:
                new_mask = self._prevent_layer_collapse(new_mask, importance_scores, model, mean_grads)
                
                # Adjust target sparsity if needed due to recovery
                target_sparsity = self._adaptive_sparsity_adjustment(target_sparsity, new_mask, 
                                                                   total_params, model)
            
            # Update mask
            current_mask = new_mask
            
            # Report progress
            current_sparsity = 1 - (current_mask.sum().item() / total_params)
            remaining_params = current_mask.sum().item()
            
            print(f"Current sparsity: {current_sparsity:.1%}")
            print(f"Remaining parameters: {remaining_params:,}")
            
            # Apply mask to model (optional - for gradient computation in next iteration)
            self._apply_mask_to_model(model, current_mask)
        
        final_sparsity = 1 - (current_mask.sum().item() / total_params)
        print(f"\n✅ Pruning completed!")
        print(f"Final sparsity: {final_sparsity:.1%}")
        print(f"Remaining parameters: {current_mask.sum().item():,}")
        
        return current_mask
    
    def _apply_mask_to_model(self, model: nn.Module, mask: torch.Tensor):
        """
        Apply pruning mask to model parameters.
        
        Args:
            model: Neural network model
            mask: Binary mask indicating which weights to keep
        """
        param_idx = 0
        with torch.no_grad():
            for param in model.parameters():
                param_size = param.numel()
                param_mask = mask[param_idx:param_idx + param_size].view(param.shape)
                param.data *= param_mask
                param_idx += param_size


def get_gradient_statistics_mask(model: nn.Module, 
                                loaders: List[DataLoader], 
                                device: torch.device,
                                target_sparsity: float = 0.9,
                                num_iterations: int = 10,
                                weight_importance_ratio: float = 0.5,
                                exponential_base: float = 5.0,
                                importance_method: str = 'weight_std',
                                layer_collapse_threshold: float = 0.01,
                                enable_recovery: bool = True):
    """
    Convenience function to get pruning mask using gradient statistics.
    
    Args:
        model: Neural network model to prune
        loaders: List of data loaders for gradient computation
        device: Computing device
        target_sparsity: Final sparsity level (0.9 = 90% pruned)
        num_iterations: Number of pruning iterations
        weight_importance_ratio: Balance between first and second component (0.5 = equal)
        exponential_base: Base for exponential sparsity progression
        importance_method: Method to compute importance ('weight_std' or 'grad_std')
            - 'weight_std': Combines weight magnitudes with gradient std
            - 'grad_std': Combines gradient magnitudes with gradient std
        layer_collapse_threshold: Minimum fraction of weights to keep per layer (default: 0.05 = 5%)
        enable_recovery: Whether to enable gradient-based weight recovery for layer collapse prevention
        
    Returns:
        torch.Tensor: Binary mask indicating remaining weights (1=keep, 0=prune)
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = deepcopy(model)
    
    # Initialize pruner
    pruner = GradientStatisticsPruner(
        target_sparsity=target_sparsity,
        num_iterations=num_iterations,
        weight_importance_ratio=weight_importance_ratio,
        exponential_base=exponential_base,
        importance_method=importance_method,
        layer_collapse_threshold=layer_collapse_threshold,
        enable_recovery=enable_recovery
    )
    
    # Perform pruning
    mask = pruner.prune_network(model_copy, loaders, device)
    
    return mask


# Example usage function
def example_usage():
    """
    Example of how to use the gradient statistics pruning algorithm.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create example model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create example data loaders
    loaders = []
    for i in range(3):  # 3 different data sources
        x = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        loaders.append(loader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("=== Testing Weight + Std Method ===")
    # Get pruning mask using weight + std method
    mask1 = get_gradient_statistics_mask(
        model=model,
        loaders=loaders,
        device=device,
        target_sparsity=0.8,  # 80% sparsity
        num_iterations=5,
        weight_importance_ratio=0.6,  # 60% weights, 40% gradients
        exponential_base=2.0,
        importance_method='weight_std'
    )
    
    print(f"Weight+Std - Mask shape: {mask1.shape}")
    print(f"Weight+Std - Remaining weights: {mask1.sum().item()}")
    
    print("\n=== Testing Gradient + Std Method ===")
    # Get pruning mask using gradient + std method
    mask2 = get_gradient_statistics_mask(
        model=model,
        loaders=loaders,
        device=device,
        target_sparsity=0.8,  # 80% sparsity
        num_iterations=5,
        weight_importance_ratio=0.6,  # 60% grad magnitude, 40% grad std
        exponential_base=2.0,
        importance_method='grad_std'
    )
    
    print(f"Grad+Std - Mask shape: {mask2.shape}")
    print(f"Grad+Std - Remaining weights: {mask2.sum().item()}")
    
    # Compare the two methods
    overlap = (mask1 * mask2).sum().item()
    total_kept1 = mask1.sum().item()
    total_kept2 = mask2.sum().item()
    
    print(f"\n=== Method Comparison ===")
    print(f"Overlap: {overlap} weights")
    print(f"Weight+Std kept: {total_kept1} weights")
    print(f"Grad+Std kept: {total_kept2} weights")
    print(f"Overlap percentage: {overlap / min(total_kept1, total_kept2) * 100:.1f}%")
    
    print("\n=== Testing Layer Collapse Prevention ===")
    # Test with layer collapse prevention enabled
    mask3 = get_gradient_statistics_mask(
        model=model,
        loaders=loaders,
        device=device,
        target_sparsity=0.95,  # Very high sparsity to trigger layer collapse prevention
        num_iterations=5,
        weight_importance_ratio=0.5,
        exponential_base=2.0,
        importance_method='grad_std',
        layer_collapse_threshold=0.05,  # Keep at least 5% of weights per layer
        enable_recovery=True
    )
    
    print(f"With Collapse Prevention - Mask shape: {mask3.shape}")
    print(f"With Collapse Prevention - Remaining weights: {mask3.sum().item()}")
    
    # Test without layer collapse prevention for comparison
    mask4 = get_gradient_statistics_mask(
        model=model,
        loaders=loaders,
        device=device,
        target_sparsity=0.95,  # Same high sparsity
        num_iterations=5,
        weight_importance_ratio=0.5,
        exponential_base=2.0,
        importance_method='grad_std',
        enable_recovery=False  # Disable recovery
    )
    
    print(f"Without Collapse Prevention - Mask shape: {mask4.shape}")
    print(f"Without Collapse Prevention - Remaining weights: {mask4.sum().item()}")
    
    print(f"Recovery difference: {mask3.sum().item() - mask4.sum().item()} weights recovered")


if __name__ == "__main__":
    example_usage()
