"""Federated learning protocol configuration."""

from dataclasses import dataclass


@dataclass
class FederationConfig:
    """Federated learning protocol settings."""

    global_epoch: int = 100
    """Total number of communication rounds."""

    local_iter: int = 1
    """Number of local training epochs per round."""

    num_clients: int = 25
    """Total number of clients in the federation."""

    participation_ratio: float = 1.0
    """Fraction of clients participating per round (1.0 = cross-silo)."""

    traitor_ratio: float = 0.2
    """Fraction (or count) of Byzantine clients. Values <1 are treated as
    a ratio; values >=1 are treated as an absolute count."""

    attack: str = 'sparse'
    """Attack strategy name. See Attacks/ for available options."""

    aggregator: str = 'tm'
    """Robust aggregation rule. See Aggregators/ for available options."""

    hybrid_aggregator_list: str = 'cc+tm'
    """'+'-separated list of aggregators for hybrid aggregation."""

    embedded_momentum: bool = False
    """Enable FedADC embedded momentum."""

    early_stop: bool = False
    """Stop training early if convergence is detected."""

    mitm: bool = True
    """Adversary has full knowledge of benign gradients (man-in-the-middle)."""

    # --- Bucketing ---
    bucketing: bool = False
    """Apply bucketing before aggregation."""

    bucket_type: str = 'Random'
    """Bucketing strategy: 'Random', 'Cosine Distance', or 'L2 distance'."""

    bucket_size: int = 3
    """Number of clients per bucket."""
