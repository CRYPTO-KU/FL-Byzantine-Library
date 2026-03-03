"""
FL-Defender: Combating Targeted Attacks in Federated Learning

Based on: https://github.com/najeebjebreel/FL-Defender

Core idea: Uses PCA on cosine similarity matrix + accumulated reputation scoring
across rounds to identify and penalize malicious clients. Trust values determine
the weighted contribution of each client to the aggregated update.
"""

import torch
import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base import _BaseAggregator


class FLDefender(_BaseAggregator):
    """
    FL-Defender aggregation with PCA-based reputation scoring.
    
    Steps per round:
        1. Compute pairwise cosine similarity matrix among client gradients
        2. Standardize + PCA(n_components=2) on the similarity matrix
        3. Compute centroid using median of PCA vectors
        4. Score each client = cosine similarity of PCA vector to centroid
        5. Accumulate scores in score_history across rounds
        6. Compute trust: trust = score_history - Q1, normalize, clip negatives
        7. Weighted average of gradients using trust values

    Args:
        n_clients: Total number of clients
    """

    def __init__(self, n_clients):
        super(FLDefender, self).__init__()
        self.n_clients = n_clients
        self.score_history = np.zeros(n_clients)
        self.rounds = 0
        self.detection_stats = {}

    def _get_pca(self, similarity_matrix):
        """Standardize and apply PCA(n_components=2) to the similarity matrix."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(similarity_matrix)
        n_components = min(2, scaled.shape[0], scaled.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled)

    def __call__(self, inputs):
        """
        Aggregate using FL-Defender reputation-based weighting.

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
        grads_np = torch.stack(inputs, dim=0).detach().cpu().numpy()
        
        # Check for NaN/Inf and filter them out
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)
        
        if n_invalid > 0:
            print(f"FL-Defender: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out")
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # All gradients are invalid - fallback to simple mean of inputs
                print("FL-Defender: All gradients invalid, using simple mean")
                return torch.stack(inputs).mean(dim=0)
            
            # Use only valid gradients
            grads_np = grads_np[valid_indices]
            inputs_valid = [inputs[i] for i in valid_indices]
            n = len(inputs_valid)
        else:
            inputs_valid = inputs

        # Step 1: Compute pairwise cosine similarity
        cs = smp.cosine_similarity(grads_np) - np.eye(n)

        # Step 2: Standardize + PCA
        cs_pca = self._get_pca(cs)

        # Step 3: Compute centroid using median
        centroid = np.median(cs_pca, axis=0)

        # Step 4: Score each client
        scores = smp.cosine_similarity([centroid], cs_pca)[0]

        # Step 5: Accumulate scores
        # Handle case where number of participating clients != n_clients
        # (could happen with client sampling, but in cross-silo all participate)
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            # If subset of clients or some filtered, just use current-round scores
            self.score_history = np.zeros(self.n_clients)
            if n_invalid == 0:
                self.score_history[:n] = scores
            else:
                # Only update scores for valid clients
                self.score_history[valid_indices] = scores

        # Step 6: Compute trust values
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1
        max_trust = trust.max()
        if max_trust > 0:
            trust = trust / max_trust
        trust = np.clip(trust, 0, None)

        # Use trust values for current valid participants
        if n_invalid == 0:
            trust_weights = trust[:n]
        else:
            trust_weights = trust[valid_indices]

        # Step 7: Weighted aggregation
        total_weight = trust_weights.sum()
        if total_weight > 0:
            weighted_sum = torch.zeros_like(inputs_valid[0])
            for i, (grad, w) in enumerate(zip(inputs_valid, trust_weights)):
                weighted_sum += w * grad
            aggregated = weighted_sum / total_weight
        else:
            # Fallback to simple average
            aggregated = torch.stack(inputs_valid).mean(dim=0)

        # Store stats
        self.detection_stats = {
            'total_clients': len(inputs),
            'valid_clients': n,
            'invalid_clients': n_invalid,
            'active_clients': int(np.sum(trust_weights > 0)),
            'filtered_out': int(np.sum(trust_weights == 0)),
            'active_ratio': float(np.sum(trust_weights > 0) / n) if n > 0 else 0.0,
            'mean_trust': float(np.mean(trust_weights)),
        }
        self.rounds += 1

        return aggregated

    def get_attack_stats(self):
        if not self.detection_stats:
            return {}
        return {
            'FLDef-active-ratio': self.detection_stats.get('active_ratio', 1.0),
            'FLDef-mean-trust': self.detection_stats.get('mean_trust', 0.0),
        }
