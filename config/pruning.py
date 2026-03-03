"""Pruning and sparsity configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PruningConfig:
    """Settings for network pruning and sparse attacks."""

    # --- Mask loading ---
    load_mask: Optional[str] = None
    """Path to a pre-computed pruning mask (indices of non-pruned params)."""

    # --- Pruning algorithm ---
    prune_method: str = 'force'
    """Pruning algorithm: 'iter_snip', 'grasp_it', 'force', 'synflow',
    'grasp', 'snip', 'lamp', 'erk', 'uniform', 'uniform+', 'random',
    'random+', 'random_layerwise+', 'force_std'."""

    pruning_factor: float = 0.005
    """Fraction of connections remaining after pruning."""

    prune_dataset_split: float = 1.0
    """Fraction of dataset used for pruning saliency computation."""

    omniscient_pruning: bool = True
    """Use benign client data for pruning (omniscient attacker)."""

    prune_bias: bool = False
    """Include bias parameters in pruning."""

    prune_bn: bool = False
    """Include BatchNorm parameters in pruning."""

    keep_orig_weights: bool = True
    """Keep original weight values after pruning (vs. re-initialize)."""

    first_layer_constraint: int = -1
    """Limit kept parameters in first layer. -1 = no constraint."""

    last_layer_constraint: int = -1
    """Limit kept parameters in last layer. -1 = no constraint."""

    min_threshold: float = -1.0
    """Minimum pruning threshold (overrides method heuristics)."""

    inout_layers: bool = False
    """Include input and output layers in the pruning mask."""

    # --- Iterative pruning ---
    num_steps: int = 100
    """Number of steps for iterative pruning."""

    mode: str = 'exp'
    """Step schedule: 'linear' or 'exp'."""

    num_batches: int = 3
    """Batches for gradient computation. -1 averages over entire dataset."""

    prune_bs: int = 32
    """Batch size for pruning saliency computation."""

    # --- FORCE saliency ---
    force_w: float = 1.0
    """Saliency scaling factor for weights."""

    force_g: float = 1.0
    """Saliency scaling factor for gradients."""

    force_v: float = 1.0
    """Saliency scaling factor for variance."""

    # --- Initialization ---
    init: str = '-'
    """Weight initialization before pruning: 'normal_kaiming' or '-' (None).
    Model and pruning should share the same initialization."""

    mask_scope: str = 'local'
    """Pruning scope: 'global' or 'local' (layer-wise)."""

    # --- Sparse attack ---
    sparse_cfg: int = 50
    """Configuration identifier for sparse attacks."""

    sparse_scale: float = 1.5
    """Attack scale for remaining (non-pruned) parameters."""

    sparse_sign: str = 'inv_std'
    """Sign strategy for sparse attack: 'inv_std' or 'inv_mean'."""

    sparse_th: Optional[str] = None
    """Thresholding method: 'iqr', 'z_score', 'gradient', or None."""
