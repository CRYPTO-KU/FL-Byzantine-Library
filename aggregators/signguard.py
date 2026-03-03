"""
SignGuard: Byzantine-robust aggregation using norm filtering and sign-based clustering
Based on: https://github.com/JianXu95/SignGuard
Modified for tensor-based aggregation without dictionaries
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from .base import _BaseAggregator


class SignGuard(_BaseAggregator):
    """
    SignGuard aggregation method
    
    This method combines:
    1. Norm-based filtering to remove outliers
    2. Sign-based clustering to identify Byzantine clients
    3. Gradient clipping for robustness
    
    Args:
        num_clients: Total number of clients
        sparsity_ratio: Ratio for sparse sign gradient analysis (default: 0.1)
        norm_bounds: Tuple of (lower, upper) bounds for norm filtering (default: (0.1, 3.0))
        clustering_method: 'dbscan' or 'meanshift' (default: 'meanshift')
        iterations: Number of iterations for sign clustering (default: 1)
        eps: DBSCAN epsilon parameter (default: 0.05)
        min_samples: DBSCAN min_samples parameter (default: 2)
    """
    
    def __init__(self, num_clients, sparsity_ratio=0.1, norm_bounds=(0.1, 3.0), 
                 clustering_method='meanshift', iterations=1, eps=0.05, min_samples=2):
        super(SignGuard, self).__init__()
        
        self.num_clients = num_clients
        self.sparsity_ratio = sparsity_ratio
        self.norm_lower_bound, self.norm_upper_bound = norm_bounds
        self.clustering_method = clustering_method
        self.iterations = iterations
        self.eps = eps
        self.min_samples = min_samples
        
        # Statistics tracking
        self.rounds = 0
        self.detection_stats = {}
        
    def norm_based_filtering(self, grads):
        """
        Filter clients based on gradient norms
        
        Args:
            grads: Tensor of shape (num_clients, num_params)
            
        Returns:
            benign_indices: Set of indices passing norm filter
        """
        all_set = set(range(grads.shape[0]))
        
        # Calculate L2 norms
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        
        # Handle NaN values
        if np.any(np.isnan(grad_l2norm)):
            grad_l2norm = np.where(np.isnan(grad_l2norm), 0, grad_l2norm)
        
        # Calculate median norm
        norm_med = np.median(grad_l2norm)
        
        # Filter based on norm bounds
        benign_idx1 = all_set.copy()
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in 
                                                   np.argwhere(grad_l2norm > self.norm_lower_bound * norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in 
                                                   np.argwhere(grad_l2norm < self.norm_upper_bound * norm_med)]))
        
        return benign_idx1, grad_l2norm, norm_med
    
    def sign_based_clustering(self, grads):
        """
        Cluster clients based on sign patterns of sparse gradients
        
        Args:
            grads: Tensor of shape (num_clients, num_params)
            
        Returns:
            benign_indices: Set of indices in the largest cluster
        """
        all_set = set(range(grads.shape[0]))
        benign_idx2 = all_set.copy()
        
        num_param = grads.shape[1]
        num_spars = int(self.sparsity_ratio * num_param)
        
        for it in range(self.iterations):
            # Randomly sample sparse parameters
            idx = torch.randint(0, (num_param - num_spars), size=(1,)).item()
            gradss = grads[:, idx:(idx + num_spars)]
            sign_grads = torch.sign(gradss)
            
            # Calculate sign ratios
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32) / num_spars
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32) / num_spars
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32) / num_spars
            
            # Normalize features
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)
            
            # Stack features
            feat = [pos_feat, zero_feat, neg_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()
            
            # Clustering
            if self.clustering_method == 'dbscan':
                clf_sign = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_cluster > 0:
                    num_class = []
                    for i in range(n_cluster):
                        num_class.append(np.sum(labels == i))
                    benign_class = np.argmax(num_class)
                    benign_idx2 = benign_idx2.intersection(set([int(i) for i in 
                                                               np.argwhere(labels == benign_class)]))
            else:  # meanshift
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=self.num_clients)
                if bandwidth <= 0:
                    bandwidth = 0.1  # Fallback bandwidth
                
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                
                if n_cluster > 0:
                    num_class = []
                    for i in range(n_cluster):
                        num_class.append(np.sum(labels == i))
                    benign_class = np.argmax(num_class)
                    benign_idx2 = benign_idx2.intersection(set([int(i) for i in 
                                                               np.argwhere(labels == benign_class)]))
        
        return benign_idx2
    
    def gradient_clipping(self, grads):
        """
        Apply gradient clipping using median norm
        
        Args:
            grads: Tensor of shape (num_clients, num_params)
            
        Returns:
            clipped_grads: Clipped gradients
        """
        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads / grad_norm) * grad_norm_clipped
        
        return grads_clip
    
    def __call__(self, inputs):
        """
        Apply SignGuard aggregation
        
        Args:
            inputs: List of flattened parameter tensors (client updates)
            
        Returns:
            aggregated_update: Aggregated flattened parameter tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        # Stack inputs into tensor
        grads = torch.stack(inputs, dim=0)
        
        # Step 1: Norm-based filtering
        benign_idx1, grad_l2norm, norm_med = self.norm_based_filtering(grads)
        
        # Step 2: Sign-based clustering
        benign_idx2 = self.sign_based_clustering(grads)
        
        # Step 3: Intersection of both filters
        benign_indices = list(benign_idx2.intersection(benign_idx1))
        
        # If no benign clients found, use all clients
        if len(benign_indices) == 0:
            benign_indices = list(range(len(inputs)))
        
        # Step 4: Gradient clipping (optional, can be applied to all or just benign)
        grads_clipped = self.gradient_clipping(grads)
        
        # Step 5: Aggregate using only benign clients
        global_grad = grads_clipped[benign_indices].mean(dim=0)
        
        # Store detection statistics
        all_set = set(range(len(inputs)))
        self.detection_stats[f'round_{self.rounds}'] = {
            'total_clients': len(inputs),
            'benign_detected': len(benign_indices),
            'norm_filtered_out': len(all_set - benign_idx1),
            'sign_filtered_out': len(all_set - benign_idx2),
            'final_benign_ratio': len(benign_indices) / len(inputs),
            'norm_median': norm_med,
            'benign_indices': benign_indices
        }
        
        self.rounds += 1
        
        return global_grad
    
    def get_attack_stats(self):
        """Get attack statistics and detection info"""
        if not self.detection_stats:
            return {}
        
        total_benign = sum([stats['benign_detected'] for stats in self.detection_stats.values()])
        total_clients = sum([stats['total_clients'] for stats in self.detection_stats.values()])
        avg_benign_ratio = total_benign / total_clients if total_clients > 0 else 0
        
        signguard_stats = {
            'SignGuard-avg-benign-ratio': avg_benign_ratio,
        }
        
        return signguard_stats
