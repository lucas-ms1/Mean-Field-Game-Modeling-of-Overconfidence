"""Lightweight in-memory logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Logger:
    prices: List[float] = field(default_factory=list)
    fundamentals: List[float] = field(default_factory=list)
    mean_beliefs: List[float] = field(default_factory=list)
    total_demands: List[float] = field(default_factory=list)
    metrics: List[dict] = field(default_factory=list)

    def log_step(
        self,
        price: float,
        fundamental: float,
        mean_belief: float,
        total_demand: float,
    ) -> None:
        self.prices.append(price)
        self.fundamentals.append(fundamental)
        self.mean_beliefs.append(mean_belief)
        self.total_demands.append(total_demand)

    def log_metrics(self, data: dict) -> None:
        self.metrics.append(data)
