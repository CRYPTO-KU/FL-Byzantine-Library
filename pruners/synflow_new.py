import torch
import torch.nn as nn
import numpy as np


class SynFlow:
    """
    Vanilla SynFlow implementation based on the original paper.
    "Pruning neural networks without any data by iteratively conserving synaptic flow"
    
    This is a simplified, clean implementation without layer constraints or complex features.
    """
    
    def __init__(self, masked_parameters):
        """
        Initialize SynFlow pruner.
        
        Args:
            masked_parameters: List of (mask, parameter) tuples
        """
        self.masked_parameters = list(masked_parameters)
        self.scores = {}
    
    def score(self, model, loss, dataloader, device):
        """
        Compute SynFlow scores for pruning.
        
        Args:
            model: Neural network model
            loss: Loss function (not used but kept for compatibility)
            dataloader: Data loader (not used but kept for compatibility) 
            device: Device to run computation on
        """
        @torch.no_grad()
        def linearize(model):
            """Convert model weights to absolute values and store signs"""
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            """Restore original signs to model weights"""
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        # Step 1: Linearize the model (convert to absolute values)
        signs = linearize(model)

        # Step 2: Create synthetic input (all ones)
        # Get input shape from first parameter
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)
        
        # Step 3: Forward pass with synthetic input
        output = model(input)
        
        # Step 4: Compute total output and backward pass
        torch.sum(output).backward()
        
        # Step 5: Compute SynFlow scores (gradient * weight)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        # Step 6: Restore original model weights
        nonlinearize(model, signs)


def get_masked_parameters(model):
    """
    Simple function to get masked parameters for regular PyTorch models.
    Returns list of (mask, parameter) tuples for prunable layers.
    """
    masked_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Skip biases (1D)
            # Create a dummy mask (all ones)
            mask = torch.ones_like(param)
            masked_params.append((mask, param))
    return masked_params


def get_synflow_mask(args, net, loader, device):
    """
    Main interface function for SynFlow compatible with existing codebase
    
    Args:
        args: Arguments containing SynFlow parameters (specifically args.pruning_factor)
        net: Neural network to prune
        loader: Data loader for getting input shape (only first batch is used)
        device: Device to run on
        
    Returns:
        Flattened binary mask tensor (1 = keep, 0 = prune)
    """
    # Get masked parameters for SynFlow
    masked_params = get_masked_parameters(net)
    
    # Initialize SynFlow
    synflow = SynFlow(masked_params)
    
    # Compute SynFlow scores
    synflow.score(net, nn.CrossEntropyLoss(), loader, device)
    
    # Apply global threshold pruning
    sparsity = 1 - args.pruning_factor  # Convert density to sparsity
    
    # Collect all scores for global thresholding
    all_scores = []
    score_shapes = []
    param_ids = []
    
    for param_id, score in synflow.scores.items():
        flat_score = score.flatten()
        all_scores.append(flat_score)
        score_shapes.append(score.shape)
        param_ids.append(param_id)
    
    # Concatenate all scores
    global_scores = torch.cat(all_scores)
    total_params = len(global_scores)
    
    # Calculate threshold for pruning
    num_to_prune = int(total_params * sparsity)
    if num_to_prune > 0 and num_to_prune < total_params:
        sorted_scores, _ = torch.sort(global_scores)
        threshold = sorted_scores[num_to_prune]
    elif num_to_prune >= total_params:
        threshold = float('inf')  # Prune everything
    else:
        threshold = float('-inf')  # Keep everything
    
    # Create masks for each parameter
    masks = {}
    param_idx = 0
    
    for i, param_id in enumerate(param_ids):
        score = synflow.scores[param_id]
        # Create mask (1 = keep, 0 = prune)
        mask = (score > threshold).float()
        masks[param_id] = mask
        param_idx += score.numel()
    
    # Convert to full network mask format (compatible with existing codebase)
    mask_full = []
    mask_idx = 0
    
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            # BatchNorm layers: keep weights and biases (not pruned by SynFlow)
            mask_full.append(torch.ones_like(layer.weight))
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer, nn.Linear):
            # Linear layers: apply SynFlow mask to weights
            param_id = id(layer.weight)
            if param_id in masks:
                mask_full.append(masks[param_id])
            else:
                mask_full.append(torch.ones_like(layer.weight))
            mask_idx += 1
            # Biases: keep (not pruned by SynFlow)
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
        elif isinstance(layer, nn.Conv2d):
            # Conv2d layers: apply SynFlow mask to weights
            param_id = id(layer.weight)
            if param_id in masks:
                mask_full.append(masks[param_id])
            else:
                mask_full.append(torch.ones_like(layer.weight))
            mask_idx += 1
            # Biases: keep (not pruned by SynFlow)
            if layer.bias is not None:
                mask_full.append(torch.ones_like(layer.bias))
    
    # Flatten and return
    flat_mask = torch.cat([torch.flatten(x) for x in mask_full])
    return flat_mask
