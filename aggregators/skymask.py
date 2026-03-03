"""
SkyMask: Attack-agnostic Robust Federated Learning with Fine-grained Learnable Masks (ECCV 2024)

Based on: https://github.com/KoalaYan/SkyMask

Core idea: Learns per-client scalar masks optimized on server-side clean data,
then uses GMM clustering on mask values to separate benign from malicious clients.
The server trains a reference update and uses it as a "known benign" anchor for
cluster identification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from copy import deepcopy
from .base import _BaseAggregator


class SkyMask(_BaseAggregator):
    """
    SkyMask aggregation with learnable masks and GMM clustering.
    
    Steps per round:
        1. Compute a server reference gradient using clean root dataset 
        2. Initialize learnable mask parameter ω_i for each client (+ server ref)
        3. For each optimization step:
           - Compute masked aggregate: g = Σ(σ(ω_i)·g_i) / Σ(σ(ω_i))
           - Apply to a copy of the model, evaluate on clean data
           - Backprop to update ω parameters
        4. Collect optimized mask values per client
        5. GMM(n_components=2) to cluster clients 
        6. Identify benign cluster (the one containing the server reference)
        7. Average gradients from benign clients only
    
    Args:
        root_dataset: Clean server-side dataset for validation
        net_ps: Reference to the global model
        args: Experiment arguments
        device: Computation device
        mask_lr: Learning rate for mask optimization (default: 0.01)
        mask_epochs: Number of optimization epochs for masks (default: 20)
    """
    
    def __init__(self, root_dataset, net_ps, args, device, mask_lr=0.01, mask_epochs=20):
        super(SkyMask, self).__init__()
        self.loader = DataLoader(root_dataset, batch_size=32, shuffle=True,
                                  num_workers=getattr(args, 'num_workers', 0))
        self.model = net_ps
        self.args = args
        self.device = device
        self.mask_lr = mask_lr
        self.mask_epochs = mask_epochs
        self.rounds = 0
        self.detection_stats = {}
        self._server_momentum = None
    
    def _compute_server_gradient(self):
        """Compute a reference gradient using the server's clean data."""
        model = deepcopy(self.model).to(self.device)
        model.train()
        
        # Do one training step on clean data
        for data in self.loader:
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            break
        
        # Extract flattened gradient
        grad_parts = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                grad_parts.append(p.grad.detach().flatten())
        
        server_grad = torch.cat(grad_parts)
        del model
        return server_grad
    
    def _gmm_cluster(self, mask_values, server_idx):
        """
        Use GMM to cluster clients into benign vs malicious.
        
        Args:
            mask_values: np.array of shape (n_total,) with optimized mask values
            server_idx: Index of the server reference in mask_values
            
        Returns:
            List of benign client indices (excluding server)
        """
        import warnings
        n = len(mask_values)
        
        # Use PCA if mask_values is multi-dimensional, else reshape
        if mask_values.ndim == 1:
            features = mask_values.reshape(-1, 1)
        else:
            n_comp = min(2, mask_values.shape[1], n)
            pca = PCA(n_components=n_comp)
            features = pca.fit_transform(mask_values)
        
        # Check if there's enough variance to cluster meaningfully
        # If all mask values are nearly identical, skip clustering
        if features.std() < 1e-6:
            # No meaningful separation — treat all as benign
            return [i for i in range(n) if i != server_idx]
        
        # Try GMM with 2 components (suppress convergence warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(n_components=2, random_state=42)
                labels = gmm.fit_predict(features)
                
                # Check if GMM actually found 2 distinct clusters
                if len(set(labels)) < 2:
                    # Only 1 cluster found — treat all as benign
                    return [i for i in range(n) if i != server_idx]
        except Exception:
            # Fallback to KMeans
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
            except Exception:
                return [i for i in range(n) if i != server_idx]
        
        # The benign cluster is the one containing the server reference
        server_label = labels[server_idx]
        
        # Collect benign indices (exclude server reference itself)
        benign_indices = [i for i in range(n) if labels[i] == server_label and i != server_idx]
        
        return benign_indices
    
    def __call__(self, inputs):
        """
        Aggregate using SkyMask learnable masks.
        
        Args:
            inputs: List of flattened gradient tensors from clients
            
        Returns:
            Aggregated gradient tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        n_clients = len(inputs)
        device = inputs[0].device
        inputs = [g.to(device) for g in inputs]
        d = inputs[0].shape[0]
        
        # Step 1: Compute server reference gradient
        server_grad = self._compute_server_gradient().to(device)
        
        # All gradients: clients + server reference (server is last)
        all_grads = inputs + [server_grad]
        n_total = len(all_grads)
        server_idx = n_total - 1
        
        # Stack all gradients
        grad_matrix = torch.stack(all_grads, dim=0).detach()  # (n_total, d)
        
        # Step 2: Initialize learnable mask parameters
        mask_params = torch.zeros(n_total, requires_grad=True, device=device)
        optimizer = torch.optim.SGD([mask_params], lr=self.mask_lr)
        
        # Step 3: Optimize masks on clean data
        model_copy = deepcopy(self.model).to(device)
        saved_state = {k: v.clone() for k, v in model_copy.state_dict().items()}
        
        for epoch in range(self.mask_epochs):
            # Get a batch of clean data
            for data in self.loader:
                x, y = data
                x, y = x.to(device), y.to(device)
                break
            
            optimizer.zero_grad()
            
            # Compute sigmoid-weighted aggregate
            weights = torch.sigmoid(mask_params)  # (n_total,)
            weight_sum = weights.sum() + 1e-10
            weighted_grad = (weights.unsqueeze(1) * grad_matrix).sum(dim=0) / weight_sum  # (d,)
            
            # Apply the weighted gradient to a copy of the model and evaluate
            model_copy.load_state_dict(saved_state)
            idx = 0
            for p in model_copy.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.data.sub_(self.args.lr * weighted_grad[idx:idx + numel].view(p.shape))
                    idx += numel
            
            # Compute loss on clean data
            model_copy.eval()
            logits = model_copy(x)
            loss = F.cross_entropy(logits, y)
            
            # Backprop to mask parameters
            loss.backward()
            optimizer.step()
        
        del model_copy, saved_state
        
        # Step 4: Collect optimized mask values
        mask_values = torch.sigmoid(mask_params).detach().cpu().numpy()
        
        # Step 5-6: GMM clustering to separate benign from malicious
        benign_indices = self._gmm_cluster(mask_values, server_idx)
        
        # Fallback if no benign clients detected
        if len(benign_indices) == 0:
            benign_indices = list(range(n_clients))
        
        # Filter to only actual client indices (not server)
        benign_indices = [i for i in benign_indices if i < n_clients]
        if len(benign_indices) == 0:
            benign_indices = list(range(n_clients))
        
        # Step 7: Average benign client gradients
        benign_grads = torch.stack([inputs[i] for i in benign_indices])
        aggregated = benign_grads.mean(dim=0)
        
        # Stats
        self.detection_stats = {
            'total_clients': n_clients,
            'benign_detected': len(benign_indices),
            'filtered_out': n_clients - len(benign_indices),
            'benign_ratio': len(benign_indices) / n_clients,
            'mask_mean': float(mask_values[:n_clients].mean()),
            'mask_std': float(mask_values[:n_clients].std()),
        }
        self.rounds += 1
        
        return aggregated
    
    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'SkyMask-benign-ratio': self.detection_stats.get('benign_ratio', 1.0),
            'SkyMask-mask-mean': self.detection_stats.get('mask_mean', 0.5),
        }
