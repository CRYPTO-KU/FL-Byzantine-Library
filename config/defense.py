"""Defense / aggregator-specific hyperparameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DefenseConfig:
    """Hyperparameters for robust aggregation strategies.

    Parameters are grouped by the aggregator that uses them. Generic
    parameters (used by multiple aggregators) come first.
    """

    # --- Centered Clipping (CC, SCC, ECC) ---
    tau: float = 1.0
    """Clipping radius for Centered Clipping."""

    n_iter: int = 1
    """Number of clipping iterations."""

    buck_rand: bool = False
    """Random bucket selection for sequential CC."""

    buck_len: int = 3
    """Bucket length for sequential CC."""

    buck_len_ecc: int = 3
    """Bucket length for sequential CC in ECC variant."""

    buck_avg: bool = False
    """Average within buckets for sequential CC."""

    multi_clip: bool = False
    """Apply multi-clipping."""

    bucket_op: Optional[str] = None
    """Bucket remainder operation: None, 'merge', or 'split'."""

    ref_fixed: bool = False
    """Use static reference point for sequential ECC CC."""

    shuffle_bucket_order: bool = False
    """Shuffle bucket processing order for sequential CC L2."""

    combine_bucket: bool = False
    """Combine all buckets for sequential CC L2."""

    # --- RFA ---
    T: int = 3
    """RFA inner iteration count."""

    nu: float = 0.1
    """RFA norm budget."""

    # --- GAS ---
    gas_p: int = 1000
    """Number of gradient chunks for GAS aggregation."""

    # --- FoolsGold ---
    fg_use_memory: bool = True
    """Use gradient history for Sybil detection."""

    fg_memory_size: int = 10
    """Number of previous rounds to remember."""

    fg_epsilon: float = 1e-5
    """Numerical stability constant."""

    # --- Chebyshev TM ---
    cheby_k_sigma: float = 1.0
    """Standard deviation scale for outlier detection."""

    # --- FoundationFL ---
    foundation_num_synthetic: int = 2
    """Number of synthetic updates to generate."""

    # --- LASA ---
    lalambda_n: float = 1.0
    """Magnitude detection threshold for Byzantine clients."""

    lalambda_s: float = 1.0
    """Sign detection threshold for Byzantine clients."""

    lasa_sparsity_ratio: float = 0.7
    """Sparsity ratio for client updates."""

    # --- FedSECA ---
    fedseca_sparsity_gamma: float = 0.9
    """Sparsification factor (fraction of gradients to zero out)."""

    # --- FedREDefense ---
    fedredefense_n_components: int = 3
    """Number of PCA components for reconstruction."""

    fedredefense_threshold: float = 0.6
    """Reconstruction error threshold for filtering."""

    # --- SkyMask ---
    skymask_lr: float = 0.01
    """Learning rate for mask optimization."""

    skymask_epochs: int = 20
    """Number of epochs for mask optimization."""

    # --- FLDetector ---
    fldetector_warmup: int = 10
    """Warmup rounds before detection starts."""

    fldetector_lbfgs_history: int = 5
    """Number of rounds for L-BFGS history."""

    # --- FLAME ---
    flame_epsilon: float = 3000.0
    """Differential privacy epsilon."""

    flame_delta: float = 0.01
    """Differential privacy delta."""

    flame_add_noise: bool = True
    """Whether to add DP noise after aggregation."""

    # --- DnC ---
    dnc_sub_dim: int = 10000
    """Dimensionality of random subsamples."""

    dnc_num_iters: int = 1
    """Number of random subsampling iterations."""

    dnc_filter_frac: float = 1.0
    """Filtering fraction (c): removes c*m clients per iteration."""

    # --- Experimental (may be removed) ---
    num_clustering: int = 3
    """Number of clusters for cluster-based aggregators."""

    bucket_shift: str = 'sequential'
    """Bucket shift type: 'sequential' or 'random'."""

    shift_amount: int = 1
    """Bucket shift amount for sequential shifting."""

    buck_len_l2: int = 3
    """Bucket length for L2-distance bucketing."""

    apply_TM: bool = False
    """Apply Trimmed Mean within buckets."""

    seq_update: bool = False
    """Use sequential model updates."""
