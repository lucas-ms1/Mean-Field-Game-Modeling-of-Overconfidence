"""Typed containers for simulation state and transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class AgentBeliefState:
    """Belief over fundamental value."""

    v_hat: float
    Sigma: float


@dataclass(frozen=True)
class AgentInternalState:
    belief: AgentBeliefState
    k: float
    u: float
    x: float
    wealth: float


@dataclass(frozen=True)
class MarketState:
    p: float
    p_prev: float
    v_true: float
    aggregate_demand: float
    regime_post: bool = False


@dataclass(frozen=True)
class SimState:
    agent: AgentInternalState
    market: MarketState


@dataclass(frozen=True)
class Transition:
    t: int
    agent_id: int
    state: SimState
    action: float
    reward: float
    next_state: SimState
    info: Dict[str, Union[float, List[float]]]


@dataclass(frozen=True)
class SimulationOutput:
    prices: List[float]
    fundamentals: List[float]
    mean_beliefs: List[float]
    demands: List[float]
    k_series: List[List[float]]
    metrics: List[Dict[str, Union[float, int]]]


Vector = Tuple[float, float, float, float]


def state_vector(agent: AgentInternalState, market: MarketState) -> np.ndarray:
    """Convert agent + market state into a minimal vector for function approximation."""
    return np.array(
        [
            1.0,
            market.p,
            agent.belief.v_hat,
            agent.belief.Sigma ** 0.5,
            agent.k,
            market.p_prev,
            float(market.regime_post),  # Add regime_post feature
        ],
        dtype=float,
    )
