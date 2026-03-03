"""Experiment-level configuration."""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Top-level experiment settings (hardware, I/O, repetitions)."""

    trials: int = 1
    """Number of independent trials to run."""

    gpu_id: int = 0
    """GPU device id. Use -1 for CPU."""

    save_loc: str = ''
    """Directory to save results. Empty string defaults to 'Results/'."""
