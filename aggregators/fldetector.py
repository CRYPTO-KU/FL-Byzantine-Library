"""
FLDetector: Defending Federated Learning Against Model Poisoning Attacks via
Detecting Malicious Clients

Based on: SkyMask repo (https://github.com/KoalaYan/SkyMask)

Core idea: The server predicts what benign gradient updates should look like using
L-BFGS approximation of the Hessian. Clients whose actual updates deviate
significantly from the predicted update accumulate a "malicious score". KMeans
clustering (k=2) on accumulated scores separates benign from malicious clients.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from .base import _BaseAggregator


class FLDetector(_BaseAggregator):
    """
    FLDetector aggregation using L-BFGS prediction and KMeans detection.
    
    Steps per round:
        1. Compute median of client gradients as a baseline
        2. If enough history (>20 rounds), use L-BFGS to predict expected gradient  
        3. Score each client: distance between actual update and predicted update
        4. Accumulate malicious scores across rounds
        5. After warmup (>10 rounds), use KMeans(k=2) on accumulated scores
        6. Filter malicious clients and re-aggregate using median
    
    Args:
        n_clients: Total number of clients
        warmup_rounds: Number of rounds before detection starts (default: 10)
        lbfgs_history: Number of rounds of history for L-BFGS (default: 5)
    """
    
    def __init__(self, n_clients, warmup_rounds=10, lbfgs_history=5):
        super(FLDetector, self).__init__()
        self.n_clients = n_clients
        self.warmup_rounds = warmup_rounds
        self.lbfgs_history = lbfgs_history
        
        # History for L-BFGS approximation
        self.weight_record = []  # S_k: difference in aggregated weights
        self.grad_record = []    # Y_k: difference in aggregated gradients
        self.last_weight = None  # Last round's aggregated weight
        self.last_grad = None    # Last round's aggregated gradient
        
        # Malicious scoring
        self.malicious_scores = None  # (rounds_seen, n_clients)
        self.rounds = 0
        self.detection_stats = {}
    
    def _lbfgs(self, S_list, Y_list, v):
        """
        L-BFGS two-loop recursion to approximate H*v (Hessian-vector product).
        
        Args:
            S_list: List of weight difference vectors
            Y_list: List of gradient difference vectors
            v: Vector to multiply with approximate Hessian
            
        Returns:
            Approximate Hessian-vector product
        """
        curr_S = S_list.copy()
        curr_Y = Y_list.copy()
        k = len(curr_S)
        
        if k == 0:
            return v.clone()
        
        rho = []
        for i in range(k):
            sy = torch.dot(curr_S[i].flatten(), curr_Y[i].flatten())
            rho.append(1.0 / (sy + 1e-10))
        
        alpha = [0.0] * k
        q = v.clone()
        
        # First loop (backward)
        for i in range(k - 1, -1, -1):
            alpha[i] = rho[i] * torch.dot(curr_S[i].flatten(), q.flatten())
            q = q - alpha[i] * curr_Y[i]
        
        # Scaling
        if k > 0:
            sy = torch.dot(curr_Y[-1].flatten(), curr_S[-1].flatten())
            yy = torch.dot(curr_Y[-1].flatten(), curr_Y[-1].flatten())
            gamma = sy / (yy + 1e-10)
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for i in range(k):
            beta = rho[i] * torch.dot(curr_Y[i].flatten(), r.flatten())
            r = r + (alpha[i] - beta) * curr_S[i]
        
        return r
    
    def _detect(self, scores):
        """
        Use KMeans (k=2) to separate malicious from benign based on scores.
        
        Args:
            scores: np.array of shape (n,) with accumulated malicious scores
            
        Returns:
            np.array of shape (n,) with 1 for benign, 0 for malicious
        """
        n = len(scores)
        if n <= 2:
            return np.ones(n)
        
        estimator = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = estimator.fit_predict(scores.reshape(-1, 1))
        
        # The benign cluster is the one with lower average score 
        # (lower score = closer to predicted, i.e. more benign)
        cluster_means = [scores[labels == c].mean() for c in range(2)]
        benign_cluster = np.argmin(cluster_means)
        
        result = np.zeros(n)
        result[labels == benign_cluster] = 1
        return result
    
    def _gap_statistic(self, scores, n_refs=10):
        """
        Gap statistic to check if more than one cluster exists.
        
        Returns:
            Number of clusters detected (1 or 2)
        """
        if len(scores) <= 3:
            return 1
        
        # Compute within-cluster dispersion for k=1
        scores_reshaped = scores.reshape(-1, 1)
        km1 = KMeans(n_clusters=1, random_state=42, n_init=10)
        km1.fit(scores_reshaped)
        wk1 = km1.inertia_
        
        # Compute within-cluster dispersion for k=2
        try:
            km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
            km2.fit(scores_reshaped)
            wk2 = km2.inertia_
        except Exception:
            return 1
        
        # Simple gap check: if k=2 dispersion is significantly less
        if wk1 > 0 and (wk1 - wk2) / wk1 > 0.3:
            return 2
        return 1
    
    def __call__(self, inputs):
        """
        Aggregate using FLDetector prediction-based filtering.
        
        Args:
            inputs: List of flattened gradient tensors from clients
            
        Returns:
            Aggregated gradient tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        n = len(inputs)
        device = inputs[0].device
        inputs = [g.to(device) for g in inputs]
        
        # Stack gradients
        grads = torch.stack(inputs, dim=0)  # (n, d)
        
        # Check for NaN/Inf and filter them out
        grads_np = grads.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)
        
        if n_invalid > 0:
            print(f"FLDetector: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out")
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # All gradients are invalid - fallback to simple mean
                print("FLDetector: All gradients invalid, using simple mean")
                self.rounds += 1
                return grads.mean(dim=0)
            
            # Use only valid gradients
            grads = grads[valid_indices]
            inputs_valid = [inputs[i] for i in valid_indices]
            n = len(inputs_valid)
        else:
            inputs_valid = inputs
            valid_indices = list(range(n))
        
        # Compute median as base aggregation
        median_grad = grads.median(dim=0).values
        
        # Compute per-client distances to median (base malicious score)
        distances = torch.norm(grads - median_grad.unsqueeze(0), dim=1).detach().cpu().numpy()
        
        # Use L-BFGS prediction if enough history
        if self.rounds > 20 and self.last_weight is not None and self.last_grad is not None:
            current_weight = median_grad.detach()
            weight_diff = current_weight - self.last_weight
            
            # Predict expected gradient using L-BFGS
            hvp = self._lbfgs(self.weight_record, self.grad_record, weight_diff)
            predicted_grad = self.last_grad + hvp
            
            # Use prediction-based distances instead
            pred_distances = torch.norm(
                grads - predicted_grad.unsqueeze(0), dim=1
            ).detach().cpu().numpy()
            distances = pred_distances
        
        # Update L-BFGS history
        current_weight = median_grad.detach().clone()
        current_grad = median_grad.detach().clone()
        
        if self.last_weight is not None and self.last_grad is not None:
            s_k = current_weight - self.last_weight
            y_k = current_grad - self.last_grad
            
            self.weight_record.append(s_k)
            self.grad_record.append(y_k)
            
            # Keep only recent history
            if len(self.weight_record) > self.lbfgs_history:
                self.weight_record.pop(0)
                self.grad_record.pop(0)
        
        self.last_weight = current_weight
        self.last_grad = current_grad
        
        # Accumulate malicious scores
        if self.malicious_scores is None:
            self.malicious_scores = distances.reshape(1, -1)
        else:
            # Handle varying number of clients
            if distances.shape[0] == self.malicious_scores.shape[1]:
                self.malicious_scores = np.vstack([self.malicious_scores, distances.reshape(1, -1)])
            else:
                self.malicious_scores = distances.reshape(1, -1)
        
        # Detection after warmup
        result_grad = median_grad
        benign_indices = list(range(n))
        
        if self.malicious_scores.shape[0] >= self.warmup_rounds + 1:
            # Use last 10 rounds of accumulated scores
            recent_scores = np.sum(self.malicious_scores[-10:], axis=0)
            
            # Check if attack is occurring (gap statistic)
            n_clusters = self._gap_statistic(recent_scores)
            
            if n_clusters > 1:
                # Detect malicious clients
                res = self._detect(recent_scores)
                benign_indices = [i for i in range(n) if res[i] == 1]
                
                if len(benign_indices) > 0:
                    # Re-aggregate with only benign clients
                    benign_grads = grads[benign_indices]
                    result_grad = benign_grads.median(dim=0).values
                else:
                    benign_indices = list(range(n))
        
        # Stats
        self.detection_stats = {
            'total_clients': n,
            'benign_detected': len(benign_indices),
            'filtered_out': n - len(benign_indices),
            'benign_ratio': len(benign_indices) / n,
            'mean_distance': float(np.mean(distances)),
        }
        self.rounds += 1
        
        return result_grad
    
    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'FLDetector-benign-ratio': self.detection_stats.get('benign_ratio', 1.0),
            'FLDetector-mean-dist': self.detection_stats.get('mean_distance', 0.0),
        }
