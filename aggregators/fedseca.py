"""
FedSECA: Sign Election and Coordinate-wise Aggregation of Gradients for Byzantine Tolerant FL
From: "FedSECA: Sign Election and Coordinate-wise Aggregation of Gradients for Byzantine Tolerant Federated Learning"
CVPR 2025 - https://github.com/JosephGeoBenjamin/FedSECA-ByzantineTolerance

FedSECA consists of two main components:
1. CRISE (Concordance Ratio Induced Sign Election): Determines consensus direction for each parameter
2. RoCA (Robust Coordinate-wise Aggregation): Aggregates variance-reduced sparse gradients aligned with elected sign

The algorithm uses variance reduction via:
- Clipping: L2-norm clipping using median norm as threshold
- Clamping: Coordinate-wise magnitude clamping using median values
- Sparsification: Keeping only top (1-γ) fraction of high-magnitude gradients
"""

import torch
from .base import _BaseAggregator


class FedSECA(_BaseAggregator):
    """
    FedSECA: Sign Election and Coordinate-wise Aggregation
    
    This robust aggregator combines:
    1. Concordance ratio-based client weighting
    2. Weighted sign election for each parameter
    3. Variance reduction via clipping, clamping, and sparsification
    4. Coordinate-wise aggregation of sign-aligned gradients only
    
    Args:
        sparsity_gamma: Sparsification factor - fraction of gradients to zero out (default: 0.9)
    """
    
    def __init__(self, sparsity_gamma=0.9):
        super(FedSECA, self).__init__()
        
        self.sparsity_gamma = sparsity_gamma
        
        # Statistics tracking
        self.rounds = 0
        self.detection_stats = {}
        
    def compute_sign_concordance(self, g1, g2):
        """
        Compute sign concordance between two gradient vectors (Eq. 2)
        
        ω(g1, g2) = (1/D) * Σ sgn(g1_j) * sgn(g2_j)
        
        Args:
            g1: First gradient vector
            g2: Second gradient vector
            
        Returns:
            Sign concordance value in [-1, 1]
        """
        sign_product = torch.sign(g1) * torch.sign(g2)
        return sign_product.mean()
    
    def compute_concordance_ratios(self, grads):
        """
        Compute concordance ratio for each client (Eq. 3)
        
        ρ_k = max[0, (1/K) * Σ_ℓ sgn(ω(g_k, g_ℓ))]
        
        Args:
            grads: Tensor of shape (K, D) - K clients, D parameters
            
        Returns:
            Tensor of shape (K,) - concordance ratios for each client
        """
        K = grads.shape[0]
        concordance_ratios = torch.zeros(K, device=grads.device)
        
        # Compute sign of pairwise concordances
        for k in range(K):
            sign_sum = 0.0
            for ell in range(K):
                omega = self.compute_sign_concordance(grads[k], grads[ell])
                sign_sum += torch.sign(omega)
            
            # ρ_k = max[0, (1/K) * Σ sgn(ω)]
            concordance_ratios[k] = max(0.0, sign_sum.item() / K)
        
        return concordance_ratios
    
    def crise_sign_election(self, grads, concordance_ratios):
        """
        CRISE: Concordance Ratio Induced Sign Election (Eq. 4)
        
        s_j = sgn(Σ_k ρ_k * sgn(g_k^j))
        
        Args:
            grads: Tensor of shape (K, D) - K clients, D parameters
            concordance_ratios: Tensor of shape (K,) - client weights
            
        Returns:
            Elected signs tensor of shape (D,) with values in {-1, 0, 1}
        """
        # Weighted sum of signs: Σ_k ρ_k * sgn(g_k^j) for each coordinate j
        # Shape: (K, D) * (K, 1) -> sum -> (D,)
        weighted_signs = (torch.sign(grads) * concordance_ratios.unsqueeze(1)).sum(dim=0)
        
        # Take sign of the weighted sum
        elected_signs = torch.sign(weighted_signs)
        
        return elected_signs
    
    def clip_gradients(self, grads):
        """
        Clip gradient vectors using median L2-norm as threshold (Eq. 5)
        
        ĝ_k = g_k * min(1, τ / ||g_k||_2)
        where τ = median({||g_k||_2 : k ∈ K})
        
        Args:
            grads: Tensor of shape (K, D)
            
        Returns:
            Clipped gradients tensor of shape (K, D)
        """
        # Compute L2 norms for each client
        norms = torch.norm(grads, dim=1, keepdim=True)  # (K, 1)
        
        # Compute median norm as clipping threshold
        tau = torch.median(norms)
        
        # Compute scaling factors: min(1, τ / ||g_k||_2)
        scale_factors = torch.clamp(tau / (norms + 1e-10), max=1.0)
        
        # Apply clipping
        clipped_grads = grads * scale_factors
        
        return clipped_grads
    
    def clamp_gradients(self, grads):
        """
        Clamp gradient magnitudes coordinate-wise using median (Eq. 6)
        
        g̅_k^j = sgn(ĝ_k^j) * min(μ^j, |ĝ_k^j|)
        where μ^j = median({|ĝ_k^j| : k ∈ K})
        
        Args:
            grads: Tensor of shape (K, D) - already clipped
            
        Returns:
            Clamped gradients tensor of shape (K, D)
        """
        # Compute median absolute value for each coordinate
        # μ^j = median of |g_k^j| across all clients k
        mu = torch.median(grads.abs(), dim=0).values  # (D,)
        
        # Get signs
        signs = torch.sign(grads)
        
        # Clamp magnitudes: min(μ^j, |g_k^j|)
        clamped_magnitudes = torch.minimum(grads.abs(), mu.unsqueeze(0))
        
        # Apply signs back
        clamped_grads = signs * clamped_magnitudes
        
        return clamped_grads
    
    def sparsify_gradients(self, raw_grads, clamped_grads):
        """
        Sparsify gradients by keeping only top (1-γ) fraction (Eq. 7)
        
        g̈_k^j = g̅_k^j * I(|g_k^j| > λ_k)
        where λ_k = γ-quantile of {|g_k^j| : j = 1,...,D}
        
        Note: λ_k is computed from raw gradients, but sparsification is applied to clamped gradients
        
        Args:
            raw_grads: Original gradients (K, D) - for computing thresholds
            clamped_grads: Clamped gradients (K, D) - to be sparsified
            
        Returns:
            Sparse gradients tensor of shape (K, D)
        """
        K, D = raw_grads.shape
        sparse_grads = torch.zeros_like(clamped_grads)
        
        for k in range(K):
            # Compute λ_k = γ-quantile of absolute values in raw gradient
            lambda_k = torch.quantile(raw_grads[k].abs(), self.sparsity_gamma)
            
            # Create mask: keep only where |raw_g_k^j| > λ_k
            mask = raw_grads[k].abs() > lambda_k
            
            # Apply mask to clamped gradients
            sparse_grads[k] = clamped_grads[k] * mask.float()
        
        return sparse_grads
    
    def variance_reduced_sparse_gradients(self, grads):
        """
        Apply full VRS pipeline: clipping -> clamping -> sparsification
        
        Args:
            grads: Raw gradients tensor of shape (K, D)
            
        Returns:
            VRS gradients tensor of shape (K, D)
        """
        # Step 1: Clip by L2 norm
        clipped = self.clip_gradients(grads)
        
        # Step 2: Clamp coordinate-wise
        clamped = self.clamp_gradients(clipped)
        
        # Step 3: Sparsify (using raw grads for threshold computation)
        sparse = self.sparsify_gradients(grads, clamped)
        
        return sparse
    
    def roca_aggregation(self, sparse_grads, elected_signs):
        """
        RoCA: Robust Coordinate-wise Aggregation (Eq. 4)
        
        Aggregate only gradients aligned with elected sign:
        δ_k^j = I(s^j * g_k^j > 0)
        g̃^j = Σ_k (δ_k^j * g_k^j) / Σ_k δ_k^j
        
        Args:
            sparse_grads: VRS gradients tensor of shape (K, D)
            elected_signs: Elected signs tensor of shape (D,)
            
        Returns:
            Aggregated gradient tensor of shape (D,)
        """
        K, D = sparse_grads.shape
        
        # Compute alignment indicator: δ_k^j = I(s^j * g_k^j > 0)
        # This is 1 when gradient and elected sign have same sign (or gradient is 0)
        alignment = (elected_signs.unsqueeze(0) * sparse_grads > 0).float()  # (K, D)
        
        # For coordinates where elected sign is 0, include all gradients
        zero_sign_mask = (elected_signs == 0).unsqueeze(0)  # (1, D)
        alignment = torch.where(zero_sign_mask, torch.ones_like(alignment), alignment)
        
        # Numerator: Σ_k (δ_k^j * g_k^j)
        numerator = (alignment * sparse_grads).sum(dim=0)  # (D,)
        
        # Denominator: Σ_k δ_k^j
        denominator = alignment.sum(dim=0)  # (D,)
        
        # Avoid division by zero - if no aligned gradients, result is 0
        aggregated = numerator / (denominator + 1e-10)
        aggregated = torch.where(denominator == 0, torch.zeros_like(aggregated), aggregated)
        
        return aggregated
    
    def __call__(self, inputs):
        """
        Apply FedSECA aggregation
        
        Args:
            inputs: List of flattened parameter tensors (client gradients)
            
        Returns:
            Aggregated gradient tensor
        """
        if len(inputs) == 0:
            raise ValueError("No client updates provided")
        
        # Stack inputs into tensor
        grads = torch.stack(inputs, dim=0)  # (K, D)
        K, D = grads.shape
        
        # Step 1: Compute concordance ratios for each client (using raw gradients)
        concordance_ratios = self.compute_concordance_ratios(grads)
        
        # Step 2: CRISE sign election (using raw gradients)
        elected_signs = self.crise_sign_election(grads, concordance_ratios)
        
        # Step 3: Variance-reduced sparse gradients
        vrs_grads = self.variance_reduced_sparse_gradients(grads)
        
        # Step 4: RoCA aggregation
        aggregated = self.roca_aggregation(vrs_grads, elected_signs)
        
        # Store statistics
        self.detection_stats[f'round_{self.rounds}'] = {
            'num_clients': K,
            'num_params': D,
            'concordance_ratios': concordance_ratios.cpu().tolist(),
            'sparsity_gamma': self.sparsity_gamma,
            'nonzero_ratio': (aggregated != 0).float().mean().item()
        }
        
        self.rounds += 1
        
        return aggregated
    
    def get_attack_stats(self):
        """Get attack statistics and detection info"""
        if not self.detection_stats:
            return {}
        
        # Get stats from most recent round
        latest_round = f'round_{self.rounds - 1}'
        if latest_round in self.detection_stats:
            stats = self.detection_stats[latest_round]
            return {
                'FedSECA-concordance-min': min(stats['concordance_ratios']),
                'FedSECA-concordance-max': max(stats['concordance_ratios']),
                'FedSECA-nonzero-ratio': stats['nonzero_ratio']
            }
        return {}
