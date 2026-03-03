import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import *


class GradualExpansionMethod:
    """
    Implementation of GEM (Gradual Expansion Method) from "Rare Gems: Finding Lottery Tickets at Initialization"
    
    This algorithm finds lottery ticket subnetworks at initialization by gradually expanding
    from a minimal sparse network and using gradient-based metrics to identify important
    connections without any training.
    """
    
    def __init__(self, target_density=0.1, expansion_steps=10, initial_sparsity=0.99,
                 expansion_mode='geometric', importance_metric='snip', 
                 num_batches=10, seed_method='random'):
        """
        Initialize GEM
        
        Args:
            target_density: Final target density level (fraction of weights to keep)
            expansion_steps: Number of expansion steps from initial to target density
            initial_sparsity: Starting sparsity level (very sparse)
            expansion_mode: How to expand density ('geometric', 'linear', 'adaptive')
            importance_metric: Metric for connection importance ('snip', 'grasp', 'synflow', 'random')
            num_batches: Number of batches for gradient computation
            seed_method: How to select initial seed network ('random', 'magnitude', 'uniform')
        """
        self.target_density = target_density
        self.expansion_steps = expansion_steps
        self.initial_sparsity = initial_sparsity
        self.expansion_mode = expansion_mode
        self.importance_metric = importance_metric
        self.num_batches = num_batches
        self.seed_method = seed_method
        
        # Tracking
        self.expansion_history = []
        self.importance_scores_history = []
        self.current_mask = None
        
    def create_initial_seed_mask(self, net):
        """
        Create initial sparse seed network
        
        Args:
            net: Neural network
            
        Returns:
            Initial binary masks for each layer
        """
        print(f"[GEM] Creating initial seed mask with {self.initial_sparsity:.3f} sparsity")
        
        masks = []
        
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights = layer.weight.data
                
                if self.seed_method == 'random':
                    # Random seed selection
                    mask = torch.rand_like(weights) > self.initial_sparsity
                    
                elif self.seed_method == 'magnitude':
                    # Magnitude-based seed selection
                    magnitude = torch.abs(weights)
                    flat_magnitude = magnitude.flatten()
                    num_keep = max(1, int(len(flat_magnitude) * (1 - self.initial_sparsity)))
                    
                    if num_keep < len(flat_magnitude):
                        threshold, _ = torch.topk(flat_magnitude, num_keep, sorted=True)
                        mask = magnitude >= threshold[-1]
                    else:
                        mask = torch.ones_like(weights, dtype=torch.bool)
                        
                elif self.seed_method == 'uniform':
                    # Uniform distribution across layers
                    num_weights = weights.numel()
                    num_keep = max(1, int(num_weights * (1 - self.initial_sparsity)))
                    
                    flat_weights = weights.flatten()
                    indices = torch.randperm(len(flat_weights))[:num_keep]
                    
                    mask = torch.zeros_like(flat_weights, dtype=torch.bool)
                    mask[indices] = True
                    mask = mask.view(weights.shape)
                    
                else:
                    raise ValueError(f"Unknown seed method: {self.seed_method}")
                
                masks.append(mask.float())
        
        return masks
    
    def compute_connection_importance(self, net, dataloader, device, current_mask=None):
        """
        Compute importance scores for connections using gradient-based metrics
        
        Args:
            net: Neural network
            dataloader: Training data loader
            device: Device to run on
            current_mask: Current mask (for masked gradient computation)
            
        Returns:
            Importance scores for each layer
        """
        net_copy = copy.deepcopy(net).to(device)
        
        # Apply current mask if provided
        if current_mask is not None:
            self.apply_mask_to_network(net_copy, current_mask)
        
        if self.importance_metric == 'snip':
            return self._compute_snip_importance(net_copy, dataloader, device)
        elif self.importance_metric == 'grasp':
            return self._compute_grasp_importance(net_copy, dataloader, device)
        elif self.importance_metric == 'synflow':
            return self._compute_synflow_importance(net_copy, device)
        elif self.importance_metric == 'random':
            return self._compute_random_importance(net_copy)
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
    
    def _compute_snip_importance(self, net, dataloader, device):
        """Compute SNIP importance scores"""
        importance_scores = []
        
        # Initialize gradients storage
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                importance_scores.append(torch.zeros_like(layer.weight))
        
        net.eval()
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_count >= self.num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            net.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # Collect gradients
            layer_idx = 0
            for layer in net.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    if layer.weight.grad is not None:
                        # SNIP: |weight * gradient|
                        importance = torch.abs(layer.weight * layer.weight.grad)
                        importance_scores[layer_idx] += importance
                    layer_idx += 1
            
            batch_count += 1
        
        # Average over batches
        importance_scores = [score / batch_count for score in importance_scores]
        return importance_scores
    
    def _compute_grasp_importance(self, net, dataloader, device):
        """Compute GraSP importance scores"""
        importance_scores = []
        
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                importance_scores.append(torch.zeros_like(layer.weight))
        
        net.eval()
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_count >= self.num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # First forward pass
            net.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Compute gradients
            gradients = torch.autograd.grad(loss, [p for p in net.parameters() if p.requires_grad], 
                                          create_graph=True)
            
            # Second order gradients (Hessian diagonal approximation)
            grad_norm = torch.tensor(0.0, device=device, requires_grad=True)
            layer_idx = 0
            grad_idx = 0
            
            for layer in net.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    if layer.weight.requires_grad:
                        grad = gradients[grad_idx]
                        grad_norm = grad_norm + (grad ** 2).sum()
                        grad_idx += 1
                    layer_idx += 1
            
            # Compute second order gradients (only if grad_norm > 0)
            if grad_norm.item() > 0:
                hessian_grads = torch.autograd.grad(grad_norm, [p for p in net.parameters() if p.requires_grad], 
                                                   retain_graph=False, allow_unused=True)
            else:
                hessian_grads = [torch.zeros_like(p) for p in net.parameters() if p.requires_grad]
            
            # GraSP importance: |weight * gradient * hessian|
            layer_idx = 0
            hess_idx = 0
            for layer in net.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    if layer.weight.requires_grad:
                        grad = gradients[hess_idx]
                        hess = hessian_grads[hess_idx]
                        importance = torch.abs(layer.weight * grad * hess)
                        importance_scores[layer_idx] += importance
                        hess_idx += 1
                    layer_idx += 1
            
            batch_count += 1
        
        # Average over batches
        importance_scores = [score / batch_count for score in importance_scores]
        return importance_scores
    
    def _compute_synflow_importance(self, net, device):
        """Compute SynFlow importance scores (data-free)"""
        importance_scores = []
        
        # Create synthetic input
        input_shape = None
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d):
                # Assume first conv layer determines input shape
                in_channels = layer.in_channels
                input_shape = (1, in_channels, 32, 32)  # Default assumption
                break
        
        if input_shape is None:
            # Linear network, estimate input size
            for layer in net.modules():
                if isinstance(layer, nn.Linear):
                    input_shape = (1, layer.in_features)
                    break
        
        if input_shape is None:
            raise ValueError("Could not determine input shape for SynFlow")
        
        # Create synthetic input (all ones)
        synthetic_input = torch.ones(input_shape, device=device)
        
        net.zero_grad()
        
        # SynFlow uses absolute values of weights
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight.data = torch.abs(layer.weight.data)
        
        # Forward pass with synthetic data
        output = net(synthetic_input)
        
        # SynFlow objective: sum of all activations
        loss = output.abs().sum()
        loss.backward()
        
        # Collect importance scores
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if layer.weight.grad is not None:
                    # SynFlow: |weight * gradient|
                    importance = torch.abs(layer.weight * layer.weight.grad)
                    importance_scores.append(importance)
                else:
                    importance_scores.append(torch.zeros_like(layer.weight))
        
        return importance_scores
    
    def _compute_random_importance(self, net):
        """Compute random importance scores (baseline)"""
        importance_scores = []
        
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                importance = torch.rand_like(layer.weight)
                importance_scores.append(importance)
        
        return importance_scores
    
    def expand_network(self, current_mask, importance_scores, target_density):
        """
        Expand the network by adding most important connections
        
        Args:
            current_mask: Current binary mask
            importance_scores: Importance scores for each layer
            target_density: Target density for this expansion step
            
        Returns:
            Updated masks
        """
        expanded_masks = []
        
        for layer_idx, (mask, importance) in enumerate(zip(current_mask, importance_scores)):
            current_density = mask.float().mean().item()
            
            if current_density >= target_density:
                # Already at or above target density
                expanded_masks.append(mask)
                continue
            
            # Find currently masked (pruned) connections
            available_connections = (mask == 0).float()
            available_importance = importance * available_connections
            
            # Calculate how many connections to add
            total_weights = mask.numel()
            current_connections = int(current_density * total_weights)
            target_connections = int(target_density * total_weights)
            connections_to_add = target_connections - current_connections
            
            if connections_to_add <= 0:
                expanded_masks.append(mask)
                continue
            
            # Select most important available connections
            flat_importance = available_importance.flatten()
            flat_mask = mask.flatten()
            
            # Get indices of available connections sorted by importance
            available_indices = torch.nonzero(flat_importance.flatten()).squeeze()
            if available_indices.numel() == 0:
                expanded_masks.append(mask)
                continue
            
            if available_indices.numel() == 1:
                available_indices = available_indices.unsqueeze(0)
            
            available_scores = flat_importance[available_indices]
            
            # Select top connections to add
            num_to_add = min(connections_to_add, len(available_scores))
            if num_to_add > 0:
                _, top_indices = torch.topk(available_scores, num_to_add)
                selected_indices = available_indices[top_indices]
                
                # Update mask
                new_flat_mask = flat_mask.clone()
                new_flat_mask[selected_indices] = 1
                new_mask = new_flat_mask.view(mask.shape)
            else:
                new_mask = mask
            
            expanded_masks.append(new_mask)
        
        return expanded_masks
    
    def apply_mask_to_network(self, net, masks):
        """Apply masks to network weights"""
        mask_idx = 0
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight.data *= masks[mask_idx]
                mask_idx += 1
    
    def get_expansion_schedule(self):
        """
        Generate expansion schedule from initial to target density
        
        Returns:
            List of density levels for each expansion step
        """
        initial_density = 1 - self.initial_sparsity
        target_density = self.target_density
        
        if self.expansion_mode == 'linear':
            densities = np.linspace(initial_density, target_density, self.expansion_steps + 1)[1:]
        elif self.expansion_mode == 'geometric':
            if initial_density > 0:
                ratio = (target_density / initial_density) ** (1 / self.expansion_steps)
                densities = [initial_density * (ratio ** i) for i in range(1, self.expansion_steps + 1)]
            else:
                densities = np.linspace(initial_density, target_density, self.expansion_steps + 1)[1:]
        elif self.expansion_mode == 'adaptive':
            # Adaptive schedule: faster expansion initially, slower later
            steps = np.arange(1, self.expansion_steps + 1)
            progress = (steps / self.expansion_steps) ** 0.5  # Square root progression
            densities = initial_density + progress * (target_density - initial_density)
        else:
            raise ValueError(f"Unknown expansion mode: {self.expansion_mode}")
        
        return densities
    
    def find_lottery_ticket(self, net, dataloader, device):
        """
        Main GEM algorithm: find lottery ticket at initialization
        
        Args:
            net: Neural network
            dataloader: Training data loader
            device: Device to run on
            
        Returns:
            Final lottery ticket masks
        """
        print(f"[GEM] Starting lottery ticket search with {self.importance_metric} importance")
        print(f"[GEM] Initial sparsity: {self.initial_sparsity:.3f}, Target density: {self.target_density:.3f}")
        
        # Step 1: Create initial seed network
        current_mask = self.create_initial_seed_mask(net)
        self.current_mask = current_mask
        
        # Step 2: Get expansion schedule
        expansion_schedule = self.get_expansion_schedule()
        print(f"[GEM] Expansion schedule (densities): {[f'{d:.3f}' for d in expansion_schedule]}")
        
        # Step 3: Gradual expansion
        for step, target_density in enumerate(expansion_schedule):
            print(f"\n[GEM] Expansion step {step + 1}/{len(expansion_schedule)}, target density: {target_density:.3f}")
            
            # Compute importance scores for current network
            importance_scores = self.compute_connection_importance(net, dataloader, device, current_mask)
            
            # Expand network
            current_mask = self.expand_network(current_mask, importance_scores, target_density)
            
            # Track progress
            actual_densities = [mask.float().mean().item() for mask in current_mask]
            avg_density = np.mean(actual_densities)
            print(f"[GEM] Achieved average density: {avg_density:.3f}")
            
            # Store history
            self.expansion_history.append({
                'step': step + 1,
                'target_density': target_density,
                'actual_density': avg_density,
                'layer_densities': actual_densities
            })
            self.importance_scores_history.append([score.cpu().clone() for score in importance_scores])
        
        print(f"\n[GEM] Lottery ticket search completed!")
        final_sparsity = 1 - np.mean([mask.float().mean().item() for mask in current_mask])
        print(f"[GEM] Final sparsity: {final_sparsity:.3f}")
        
        return current_mask


def get_gem_mask(args, net, loader, device):
    """
    Main interface function for GEM compatible with existing codebase
    
    Args:
        args: Arguments containing GEM parameters
              args.pruning_factor should be the target density (e.g., 0.05 for 5% of weights remaining)
        net: Neural network to find lottery ticket for
        loader: Training data loader
        device: Device to run on
        
    Returns:
        Flattened binary mask tensor
    """
    # Initialize GEM
    gem = GradualExpansionMethod(
        target_density=args.pruning_factor,  # Use pruning_factor directly as density
        expansion_steps=getattr(args, 'num_steps', 20),
        initial_sparsity=getattr(args, 'gem_initial_sparsity', 0.99),
        expansion_mode=getattr(args, 'gem_expansion_mode', 'geometric'),
        importance_metric=getattr(args, 'gem_importance_metric', 'snip'),
        num_batches=getattr(args, 'num_batches', 3),
        seed_method=getattr(args, 'gem_seed_method', 'random')
    )
    
    # Find lottery ticket
    masks = gem.find_lottery_ticket(net, loader, device)
    
    # Convert to full network mask format
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


def gem_analysis(net, dataloader, device, importance_metrics=['snip', 'grasp', 'synflow'], 
                 expansion_modes=['linear', 'geometric', 'adaptive']):
    """
    Analyze different GEM configurations
    
    Args:
        net: Neural network
        dataloader: Training data loader
        device: Device
        importance_metrics: List of importance metrics to compare
        expansion_modes: List of expansion modes to compare
        
    Returns:
        Analysis results
    """
    results = {}
    
    for metric in importance_metrics:
        for mode in expansion_modes:
            config_name = f"{metric}_{mode}"
            print(f"\n[GEM Analysis] Testing {config_name}")
            
            gem = GradualExpansionMethod(
                target_density=0.1,  # 10% of weights remaining
                expansion_steps=5,  # Fewer steps for analysis
                importance_metric=metric,
                expansion_mode=mode,
                num_batches=5
            )
            
            masks = gem.find_lottery_ticket(net, dataloader, device)
            
            # Compute statistics
            final_sparsity = 1 - np.mean([mask.float().mean().item() for mask in masks])
            layer_sparsities = [1 - mask.float().mean().item() for mask in masks]
            
            results[config_name] = {
                'final_sparsity': final_sparsity,
                'layer_sparsities': layer_sparsities,
                'expansion_history': gem.expansion_history,
                'masks': [mask.cpu() for mask in masks]
            }
    
    return results


def gem_lottery_ticket_verification(net, masks, dataloader, device, num_epochs=10):
    """
    Verify the quality of found lottery ticket by training
    
    Args:
        net: Neural network
        masks: Lottery ticket masks
        dataloader: Training data loader
        device: Device
        num_epochs: Number of training epochs for verification
        
    Returns:
        Training results
    """
    # Create pruned network
    pruned_net = copy.deepcopy(net)
    
    # Apply masks
    mask_idx = 0
    for layer in pruned_net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight.data *= masks[mask_idx]
            
            # Register hook to maintain sparsity during training
            def make_hook(mask):
                def hook(grad):
                    return grad * mask
                return hook
            
            layer.weight.register_hook(make_hook(masks[mask_idx]))
            mask_idx += 1
    
    # Simple training loop for verification
    optimizer = torch.optim.SGD(pruned_net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    pruned_net.train()
    training_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit batches for quick verification
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = pruned_net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        print(f"[GEM Verification] Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    return {
        'training_losses': training_losses,
        'final_loss': training_losses[-1] if training_losses else float('inf'),
        'sparsity': 1 - np.mean([mask.float().mean().item() for mask in masks])
    }
