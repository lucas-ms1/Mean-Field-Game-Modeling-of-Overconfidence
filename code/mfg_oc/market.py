"""Market aggregation and price impact rules."""

from __future__ import annotations

import math
import random
from typing import Iterable, Optional


def aggregate_demand(demands: Iterable[float]) -> float:
    return sum(demands)


def price_impact(total_demand: float, price: float, lambda_price: float, dt: float = 1.0) -> float:
    """Simple price impact rule."""
    dt = float(dt)
    dt = max(dt, 0.0)
    return price + lambda_price * total_demand * dt


def anchored_impact(
    price: float,
    anchor: float,
    total_demand: float,
    num_agents: int,
    kappa: float,
    impact: float,
    noise_sigma: float,
    rng: random.Random,
    dt: float = 1.0,
    *,
    # Amplification extensions
    state_dependent_kappa: bool = False,
    kappa_decay_rate: float = 0.1,
    overconfidence_dependent_impact: bool = False,
    impact_sensitivity: float = 0.2,
    k_mean: Optional[float] = None,
) -> float:
    """
    Anchored price rule with impact and optional noise.
    
    Supports amplification extensions:
    - State-dependent anchoring: κ weakens with mispricing (limits to arbitrage)
    - Overconfidence-dependent impact: λ increases with aggregate overconfidence
    """
    dt = float(dt)
    dt = max(dt, 0.0)
    noise = rng.gauss(0.0, noise_sigma * (dt**0.5)) if noise_sigma > 0.0 and dt > 0.0 else 0.0
    
    # State-dependent anchoring: κ(t) = κ_base * exp(-α|p_t - v_t|)
    kappa_effective = float(kappa)
    if state_dependent_kappa:
        mispricing = abs(price - anchor)
        kappa_effective = kappa * math.exp(-kappa_decay_rate * mispricing)
        kappa_effective = max(kappa_effective, kappa * 0.01)  # Floor at 1% of base
    
    # Overconfidence-dependent impact: λ_eff = λ_base * (1 + β_λ * (k_mean - 1))
    impact_effective = float(impact)
    if overconfidence_dependent_impact and k_mean is not None:
        impact_effective = impact * (1.0 + impact_sensitivity * max(k_mean - 1.0, 0.0))
    
    drift = kappa_effective * (anchor - price) * dt + impact_effective * (total_demand / max(num_agents, 1)) * dt
    return price + drift + noise
