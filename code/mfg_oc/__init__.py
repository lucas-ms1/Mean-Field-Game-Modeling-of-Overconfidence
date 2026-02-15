"""MFG overconfidence simulation scaffold."""

from .config import SimulationConfig, load_config
from .simulate import simulate

__all__ = ["SimulationConfig", "load_config", "simulate"]
