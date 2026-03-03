"""Attack-specific hyperparameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AttackConfig:
    """Hyperparameters for Byzantine attack strategies."""

    # --- General ---
    z_max: Optional[float] = None
    """Attack scale. None for automatic generation."""

    alie_z_max: Optional[float] = None
    """Attack scale specifically for the ALIE attack."""

    nestrov_attack: bool = False
    """Perform a clean step first (for non-omniscient attacks)."""

    # --- IPM ---
    epsilon: float = 0.2
    """Inner Product Manipulation attack scale."""

    # --- Min-Max / Min-Sum ---
    pert_vec: str = 'std'
    """Perturbation vector type: 'unit_vec', 'sign', or 'std'."""

    delta_coeff: float = 0.9
    """Scaling coefficient for perturbation."""

    # --- LASA Attack ---
    lasa_attack_k1: float = 0.01
    """LASA attack parameter k1."""

    lasa_attack_k2: float = 1
    """LASA attack parameter k2."""

    # --- Modular Attack (ROP) ---
    pi: float = 1.0
    """Location of the attack. 1 = full relocation to aggregator reference."""

    angle: float = 270.0
    """Angle of the perturbation (degrees): 180, 90, or None."""

    lamb: float = 0.9
    """Reference point for attack when angle is not None."""
