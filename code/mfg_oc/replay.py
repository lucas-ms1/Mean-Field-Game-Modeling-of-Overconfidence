"""Experience replay buffer with salience-weighted sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

from .memory import memory_weight
from .types import Transition


@dataclass
class ReplayBuffer:
    capacity: int
    h0: float
    h1: float
    rho_pos: float
    rho_neg: float
    seed: int = 0
    _buffer: List[Transition] = field(default_factory=list)
    _weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, transition: Transition, weight: float) -> None:
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
            self._weights.pop(0)
        self._buffer.append(transition)
        self._weights.append(max(weight, 1e-6))

    def normalized_weights(self, t: int) -> List[float]:
        """Compute normalized sampling weights for the current buffer."""
        if not self._buffer:
            return []
        raw: List[float] = []
        for transition in self._buffer:
            age = max(t - transition.t, 0)
            sal = transition.info.get("salience", 0.0)
            base = memory_weight(age=age, h0=self.h0, h1=self.h1, salience=sal)
            rho = self.rho_pos if transition.reward >= 0.0 else self.rho_neg
            bias = math.exp(rho * (1.0 if transition.reward >= 0.0 else -1.0) * sal)
            raw.append(base * bias)
        total = sum(raw) if raw else 1.0
        return [w / total for w in raw]

    def sample(self, batch_size: int, t: int, rng: random.Random | None = None, min_t: int | None = None) -> List[Transition]:
        """Sample transitions with salience-biased fading-memory weights.
        
        Args:
            batch_size: Number of transitions to sample
            t: Current timestep (for age computation)
            rng: Random number generator
            min_t: Minimum timestep to include (recency gating). If None, no filtering.
        """
        if not self._buffer:
            return []
        
        # Filter by min_t if provided
        eligible_indices = list(range(len(self._buffer)))
        if min_t is not None:
            eligible_indices = [i for i in eligible_indices if self._buffer[i].t >= min_t]
        
        # If no eligible transitions, fall back to most recent
        if not eligible_indices:
            eligible_indices = [i for i in range(len(self._buffer)) if self._buffer[i].t >= max(0, t - 50)]
        
        if not eligible_indices:
            return []
        
        # Compute weights only over eligible set
        eligible_buffer = [self._buffer[i] for i in eligible_indices]
        raw: List[float] = []
        for transition in eligible_buffer:
            age = max(t - transition.t, 0)
            sal = transition.info.get("salience", 0.0)
            base = memory_weight(age=age, h0=self.h0, h1=self.h1, salience=sal)
            rho = self.rho_pos if transition.reward >= 0.0 else self.rho_neg
            bias = math.exp(rho * (1.0 if transition.reward >= 0.0 else -1.0) * sal)
            raw.append(base * bias)
        
        total = sum(raw) if raw else 1.0
        probs = [w / total for w in raw]
        picker = rng or self._rng
        sampled_local = picker.choices(range(len(eligible_indices)), probs, k=batch_size)
        return [eligible_buffer[i] for i in sampled_local]
