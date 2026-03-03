"""
FL-Byzantine-Library Configuration System.

Organized into domain-specific dataclasses for clean, typed configuration.
All configs compose into a single FLConfig for the full experiment setup.
"""

from .base import ExperimentConfig
from .federation import FederationConfig
from .optimizer import OptimizerConfig
from .model import ModelConfig
from .defense import DefenseConfig
from .attack import AttackConfig
from .pruning import PruningConfig
from .parser import FLConfig, parse_args, args_parser

__all__ = [
    'ExperimentConfig',
    'FederationConfig',
    'OptimizerConfig',
    'ModelConfig',
    'DefenseConfig',
    'AttackConfig',
    'PruningConfig',
    'FLConfig',
    'parse_args',
    'args_parser',
]
