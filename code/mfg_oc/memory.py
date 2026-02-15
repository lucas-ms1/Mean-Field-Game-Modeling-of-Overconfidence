"""Salience and fading-memory utilities."""

from __future__ import annotations

import math
from typing import Iterable, List


def compute_salience(g: float, ema_g: float, eps: float) -> float:
    """Compute salience from a magnitude signal and its EMA.

    The ratio g / (ema_g + eps) measures surprise relative to recent scale.

    >>> round(compute_salience(2.0, 1.0, 0.1), 3)
    1.818
    """
    return max(0.0, g / (ema_g + eps))


def update_ema(prev_ema: float, g: float, alpha: float) -> float:
    """Update exponential moving average of g."""
    return (1.0 - alpha) * prev_ema + alpha * g


def memory_weight(age: int, h0: float, h1: float, salience: float) -> float:
    """Compute fading memory weight for a given age (in steps).

    w = exp( - age / (h0 + h1 * salience) )

    >>> round(memory_weight(10, 5.0, 5.0, 0.0), 3)
    0.135
    >>> memory_weight(10, 5.0, 5.0, 2.0) > memory_weight(10, 5.0, 5.0, 0.0)
    True
    """
    scale = max(h0 + h1 * salience, 1e-6)
    return math.exp(-float(age) / scale)


def decay_factor(h0: float, h1: float, salience: float) -> float:
    """Single-step decay factor implied by the fading-memory scale (in steps)."""
    scale = max(h0 + h1 * salience, 1e-6)
    return math.exp(-1.0 / scale)


def fading_kernel(length: int, rho: float) -> List[float]:
    """Exponential fading-memory kernel that sums to 1."""
    weights = [rho ** i for i in range(length)]
    total = sum(weights) if weights else 1.0
    return [w / total for w in weights]


def apply_fading(values: Iterable[float], rho: float) -> float:
    """Compute a fading-memory average for a sequence."""
    vals = list(values)
    weights = fading_kernel(len(vals), rho)
    return sum(v * w for v, w in zip(vals, weights))
