"""
LASA: Layer-Adaptive Sparsified model Aggregation
From: "Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation"
https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf

Official implementation: https://github.com/JiiahaoXU/LASA/blob/master/algorithms/defense/lasa.py

LASA enhances Byzantine resilience by:
1. Applying gradient clipping to client updates
2. Pre-aggregation sparsification using top-k selection
3. Layer-wise Byzantine detection using norm check and sign check
4. Adaptive aggregation of benign clients only
"""

import torch
import numpy as np
from .base import _BaseAggregator


def topk_sparsification(vector, sparsity_ratio):
    """
    Apply top-k sparsification to a vector
    
    Args:
        vector: Input tensor vector
        sparsity_ratio: Sparsification ratio (0-1)
        
    Returns:
        sparse_vector: Sparsified vector
    """
    k_dim = int(sparsity_ratio * vector.numel())
    sign_vec = vector.sign()
    sparse_update = torch.zeros_like(vector)
    vals, indices = torch.topk(vector.abs(), k_dim)
    sparse_update[indices] = vals
    sparse_update *= sign_vec
    return sparse_update


class LASA(_BaseAggregator):
    """
    Layer-Adaptive Sparsified model Aggregation (LASA)
    
    Official implementation from: https://github.com/JiiahaoXU/LASA
    
    The algorithm consists of:
    1. Gradient clipping using median norm
    2. Pre-aggregation sparsification 
    3. Layer-wise Byzantine detection:
       - Norm check: Detect clients with abnormal gradient norms
       - Sign check: Detect clients with abnormal sign patterns
    4. Aggregation using only benign clients
    
    Args:
        sparsity_ratio: Sparsification ratio for pre-aggregation (default: 0.9)
        lambda_n: Threshold for norm check (default: 1)
        lambda_s: Threshold for sign check (default: 1)
        num_clients: Total number of clients
    """
    def __init__(self, layer_dims, sparsity_ratio=0.9, lambda_n=1, lambda_s=1, num_clients=10):
        super(LASA, self).__init__()

        self.layer_dims = layer_dims
        self.sparsity_ratio = sparsity_ratio
        self.lambda_n = lambda_n  # Norm check threshold
        self.lambda_s = lambda_s  # Sign check threshold
        self.num_clients = num_clients
        
        # Statistics tracking
        self.rounds = 0
        self.detection_stats = {}
        
    def gradient_sanitization(self, local_updates):
        local_updates_ = []
        for i in range(len(local_updates)):
            if local_updates[i].isnan().any():
                continue
            local_updates_.append(local_updates[i])
        flat_all_grads = torch.stack(local_updates, dim=0)
        grad_norm = torch.norm(flat_all_grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (flat_all_grads/grad_norm)*grad_norm_clipped
        #print(grad_norm)
        #print('after clipping')
        #print(torch.norm(grads_clip, dim=1).reshape((-1, 1)))
        return grads_clip
    
    def pre_aggregation_sparsification(self, local_updates):
        """
        Apply pre-aggregation sparsification to all updates
        
        Args:
            local_updates: List of parameter dictionaries
            
        Returns:
            sparse_updates: List of sparsified parameter dictionaries
        """
        sparse_updates = []
        for update in local_updates:
            sparse_update = topk_sparsification(update, self.sparsity_ratio)
            sparse_updates.append(sparse_update)
        
        return sparse_updates

    def byzantine_detection(self, local_updates, start_dim, end_dim):
        """
        Detect Byzantine clients for a specific layer using norm and sign checks
        
        Args:
            local_updates: List of flattened parameter tensors
            start_dim: Starting dimension index for layer
            end_dim: Ending dimension index for layer
            
        Returns:
            benign_indices: List of indices of detected benign clients
        """
        all_set = set(range(len(local_updates)))
        
        # Extract layer parameters using start_dim and end_dim
        layer_flat_params = []
        for cl_update in local_updates:
            flat_param = cl_update[start_dim:end_dim]
            layer_flat_params.append(flat_param)

        if len(layer_flat_params) == 0:
            return list(all_set)
        
        grads = torch.stack(layer_flat_params, dim=0)
        
        # Norm check
        grad_l2norm = torch.norm(grads.float(), dim=1).cpu().numpy()
        norm_med = np.median(grad_l2norm)
        norm_std = np.std(grad_l2norm)
        #print(grad_l2norm)
        #print(norm_med)
        #print(norm_std)

        # Calculate MZ-score for norm check
        norm_scores = []
        for i in range(len(grad_l2norm)):
            if norm_std > 0:
                score = np.abs((grad_l2norm[i] - norm_med) / norm_std)
            else:
                score = 0
            norm_scores.append(score)
        #print(norm_scores)
        # Find benign clients based on norm check
        benign_idx1 = all_set.copy()
        benign_idx1 = benign_idx1.intersection(
            set([int(i) for i in np.argwhere(np.array(norm_scores) < self.lambda_n).flatten()])
        )
        #print(benign_idx1)
        #print(norm_scores)
        # print(np.array(norm_scores) < self.lambda_n)
        # Sign check
        layer_signs = []
        for i, layer_param in enumerate(layer_flat_params):
            sign_sum = torch.sum(torch.sign(layer_param))
            abs_sign_sum = torch.sum(torch.abs(torch.sign(layer_param)))
            
            if abs_sign_sum > 0:
                sign_ratio = 0.5 * (1 + sign_sum / abs_sign_sum * (1 - self.sparsity_ratio))
                layer_signs.append(sign_ratio.item())
            else:
                layer_signs.append(0.5)
        
        benign_idx2 = all_set.copy()
        if len(layer_signs) > 0:
            median_sign = np.median(layer_signs)
            std_sign = np.std(layer_signs)
            
            # Calculate MZ-score for sign check
            sign_scores = []
            for sign in layer_signs:
                if std_sign > 0:
                    score = np.abs((sign - median_sign) / std_sign)
                else:
                    score = 0
                sign_scores.append(score)
            
            benign_idx2 = benign_idx2.intersection(
                set([int(i) for i in np.argwhere(np.array(sign_scores) < self.lambda_s).flatten()])
            )
            #print(sign_scores)
            #print(layer_signs)
            #print(median_sign,std_sign)
            #print(np.array(sign_scores) < self.lambda_s)
        
        # Intersection of both checks
        benign_indices = list(benign_idx2.intersection(benign_idx1))
        
        # If no benign clients found, use all clients
        if len(benign_indices) == 0:
            benign_indices = list(all_set)
        #print(benign_idx2)
        
        # Store detection statistics
        layer_key = f"layer_{start_dim}_{end_dim}"
        self.detection_stats[layer_key] = {
            'norm_scores': norm_scores,
            'sign_scores': sign_scores if len(layer_signs) > 0 else [],
        }
        
        return benign_indices
    
    def __call__(self, inputs):
        """
        Apply LASA aggregation algorithm
        
        Args:
            inputs: List of flattened parameter tensors (client updates)
            
        Returns:
            aggregated_update: Aggregated flattened parameter tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")

        # Step 1: Gradient clipping and sanitization
        clipped_updates = self.gradient_sanitization(inputs)
        
        # Step 2: Pre-aggregation sparsification
        sparse_updates = self.pre_aggregation_sparsification(clipped_updates)
        
        # Step 3: Layer-wise adaptive aggregation
        aggregated_layers = []
        
        # Iterate through each layer using cumulative dimensions
        byz_passed = []
        for i in range(len(self.layer_dims) - 1):
            start_dim = self.layer_dims[i]
            end_dim = self.layer_dims[i + 1]

            # Step 3a: Byzantine detection for this layer
            benign_indices = self.byzantine_detection(sparse_updates, start_dim, end_dim)
            byz_passed.append(1 if 20 in benign_indices else 0)
            # Step 3b: Aggregate using only benign clients (from clipped updates)
            benign_layer_params = []
            for idx in benign_indices:
                if idx < len(clipped_updates):
                    layer_param = clipped_updates[idx][start_dim:end_dim]
                    benign_layer_params.append(layer_param)
            
            if len(benign_layer_params) > 0:
                aggregated_layer = torch.mean(torch.stack(benign_layer_params, dim=0), dim=0)
            else:
                # Fallback to first client if no benign clients
                aggregated_layer = clipped_updates[0][start_dim:end_dim]
                
            aggregated_layers.append(aggregated_layer)
        #print(byz_passed)
        # Concatenate all layer aggregations
        aggregated_update = torch.cat(aggregated_layers, dim=0)
        
        self.rounds += 1
        self.avg_byz_passed = sum(byz_passed) / len(byz_passed) if byz_passed else 0

        return aggregated_update
    
    def get_attack_stats(self):
        """Get attack statistics and detection info"""
        lasa_stats = {
            'LASA-byz-ratio': self.avg_byz_passed,
        }
        
        return lasa_stats

