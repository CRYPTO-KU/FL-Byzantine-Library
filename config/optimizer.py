"""Local optimizer configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class OptimizerConfig:
    """Settings for the local (client-side) optimizer."""

    name: str = 'sgd'
    """Optimizer type: 'sgd', 'adam', or 'adamw'."""

    lr: float = 0.1
    """Initial learning rate."""

    lr_decay_epochs: List[int] = field(default_factory=lambda: [75])
    """Epochs at which learning rate is multiplied by 0.1."""

    weight_decay: float = 0.0
    """L2 regularization strength."""

    momentum: float = 0.9
    """SGD momentum coefficient."""

    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam/AdamW beta coefficients."""

    max_grad_norm: float = -1.0
    """Maximum gradient norm for clipping. -1 disables clipping."""

    worker_momentum: bool = True
    """Scale gradient by (1 - momentum) before accumulating (Adam-like)."""

    nesterov: bool = False
    """Use Nesterov momentum for SGD."""
