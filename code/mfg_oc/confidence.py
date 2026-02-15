"""Dynamic overconfidence updates."""

from __future__ import annotations

import math
from typing import Dict

from .memory import decay_factor


def update_k(
    prev_k: float,
    prev_u: float,
    correctness: float,
    salience: float,
    k_mean: float,
    params: Dict[str, float],
) -> tuple[float, float]:
    """Update overconfidence parameter k_t.

    correctness: signal of predictive accuracy in [-1, 1]
    salience: nonnegative salience weight
    """
    k_min = params.get("k_min", 0.5)
    k_max = params.get("k_max", 3.0)
    k_bar = params.get("k_bar", 1.0)
    psi = params.get("psi", 1.0)
    alpha_u = params.get("alpha_u", 0.1)
    lambda_herd = params.get("lambda_herd", 0.0)
    h0 = params.get("h0", 1.0)
    h1 = params.get("h1", 0.0)

    phi = decay_factor(h0=h0, h1=h1, salience=salience)
    u_next = phi * prev_u + alpha_u * correctness * salience
    k_next = k_bar * math.exp(psi * u_next)
    k_next = max(k_min, min(k_max, k_next))
    k_next = k_next + lambda_herd * (k_mean - k_next)
    k_next = max(k_min, min(k_max, k_next))
    return k_next, u_next
