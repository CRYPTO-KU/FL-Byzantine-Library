import torch
import numpy as np
from .base import _BaseAggregator


def cosine_similarity_torch(X):
    """
    Compute cosine similarity matrix using PyTorch
    """
    X = torch.tensor(X, dtype=torch.float32)
    # Normalize each row to unit length
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    # Compute cosine similarity matrix
    similarity = torch.mm(X_norm, X_norm.t())
    return similarity.numpy()


class FoolsGold(_BaseAggregator):
    """
    Implementation of FoolsGold: Mitigating Sybils in Federated Learning Poisoning
    
    FoolsGold identifies and penalizes Sybil (coordinated malicious) clients based on 
    the similarity of their gradient updates. It uses cosine similarity to detect
    clients that submit similar gradients and reduces their influence in aggregation.
    
    Paper: "Mitigating Sybils in Federated Learning Poisoning"
    Authors: Clement Fung, Chris J.M. Yoon, Ivan Beschastnikh
    ArXiv: https://arxiv.org/abs/1808.04866
    
    Args:
        n (int): Total number of clients
        use_memory (bool): Whether to use gradient history (default: True)
        memory_size (int): Number of previous rounds to remember (default: 10)
        epsilon (float): Small constant to avoid division by zero (default: 1e-5)
    """
    
    def __init__(self, n, use_memory=True, memory_size=10, epsilon=1e-5):
        super(FoolsGold, self).__init__()
        self.n = n
        self.use_memory = use_memory
        self.memory_size = memory_size
        self.epsilon = epsilon
        
        # Initialize gradient memory
        self.gradient_memory = None
        self.round_count = 0
        
        # Track attack statistics
        self.client_weights = None
        
    def _initialize_memory(self, grad_shape):
        """Initialize gradient memory with appropriate dimensions"""
        if self.use_memory:
            self.gradient_memory = np.zeros((self.n, grad_shape, self.memory_size))
        else:
            self.gradient_memory = np.zeros((self.n, grad_shape))
    
    def _flatten_gradients(self, inputs):
        """Flatten and stack gradients from all clients"""
        flattened_grads = []
        for grad in inputs:
            if isinstance(grad, torch.Tensor):
                flat_grad = grad.flatten().detach().cpu().numpy()
            else:
                flat_grad = np.array(grad).flatten()
            flattened_grads.append(flat_grad)
        
        return np.array(flattened_grads)
    
    def _compute_foolsgold_weights(self, gradients):
        """
        Core FoolsGold algorithm to compute client weights based on gradient similarity
        
        Args:
            gradients (np.array): Gradients from all clients, shape (n_clients, grad_dim)
            
        Returns:
            np.array: Weight vector for each client
        """
        n_clients = gradients.shape[0]
        
        # Compute pairwise cosine similarity matrix
        cs = cosine_similarity_torch(gradients) - np.eye(n_clients)
        
        # Compute maximum cosine similarity for each client
        maxcs = np.max(cs, axis=1) + self.epsilon
        
        # Pardoning: reduce similarity scores based on maximum similarity
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        
        # Compute weights: 1 - max similarity with other clients
        wv = 1 - np.max(cs, axis=1)
        wv[wv > 1] = 1
        wv[wv < 0] = 0
        
        # Rescale so that max weight is close to 1
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        wv[wv == 1] = 0.99
        
        # Apply logit transformation for smoother weighting
        wv = np.log(wv / (1 - wv + self.epsilon)) + 0.5
        wv[np.isinf(wv) | (wv > 1)] = 1
        wv[wv < 0] = 0
        
        return wv
    
    def __call__(self, inputs):
        """
        Aggregate gradients using FoolsGold weighting scheme
        
        Args:
            inputs (list): List of gradient tensors from clients
            
        Returns:
            torch.Tensor: Weighted aggregate of input gradients
        """
        if len(inputs) == 0:
            raise ValueError("No inputs provided for aggregation")
        
        # Flatten gradients for similarity computation
        flat_gradients = self._flatten_gradients(inputs)
        grad_dim = flat_gradients.shape[1]
        
        # Initialize memory if first round
        if self.gradient_memory is None:
            self._initialize_memory(grad_dim)
        
        # Update gradient memory
        if self.use_memory and self.gradient_memory is not None:
            # Add current gradients to memory
            memory_idx = self.round_count % self.memory_size
            self.gradient_memory[:, :, memory_idx] = flat_gradients
            
            # Use cumulative gradients for similarity computation
            if self.round_count < self.memory_size:
                # Not enough history yet, use current gradients
                similarity_gradients = flat_gradients
            else:
                # Use sum of gradients over memory window
                similarity_gradients = np.sum(self.gradient_memory, axis=2)
        else:
            # Use only current gradients
            similarity_gradients = flat_gradients
        
        # Compute FoolsGold weights
        weights = self._compute_foolsgold_weights(similarity_gradients)
        self.client_weights = weights.copy()  # Store for statistics
        
        # Apply weights to aggregate gradients
        weighted_sum = torch.zeros_like(inputs[0])
        total_weight = 0.0
        
        for i, (grad, weight) in enumerate(zip(inputs, weights)):
            weighted_sum += weight * grad
            total_weight += weight
        
        # Normalize by total weight to maintain gradient scale
        if total_weight > 0:
            aggregated_grad = weighted_sum / total_weight
        else:
            # Fallback to simple average if all weights are zero
            aggregated_grad = torch.stack(inputs).mean(dim=0)
        
        self.round_count += 1
        
        return aggregated_grad
    
    
    def reset_memory(self):
        """Reset gradient memory and round counter"""
        self.gradient_memory = None
        self.round_count = 0
        self.client_weights = None
