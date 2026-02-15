"""Online price transition model (RLS with fading memory)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import PriceModelConfig


class PriceTransitionModel:
    """Online linear transition model for prices.

    Model: p_{t+1} = theta0 + theta1 * p_t + theta2 * m_t + theta3 * D_t + eps

    Uses recursive least squares (RLS) with forgetting factor lambda.
    - lambda < 1 implies fading memory in estimation.
    - weight > 1 increases the impact of that observation (salience-weighted learning).
    """

    def __init__(self, config: PriceModelConfig, theta: Optional[np.ndarray] = None) -> None:
        self.config = config
        self.theta = theta if theta is not None else np.zeros(5, dtype=float)  # 5 features including regime_post
        self.P = np.eye(5, dtype=float) * float(config.delta)  # 5x5 covariance
        self.delta = float(config.delta)  # Store delta for reset

    @staticmethod
    def features(p_t: float, m_t: float, D_t: float, regime_post: bool = False) -> np.ndarray:
        """Features: [1, p_t, m_t, D_t, regime_post]"""
        return np.array([1.0, p_t, m_t, D_t, float(regime_post)], dtype=float)

    def predict(self, p_t: float, m_t: float, D_t: float, regime_post: bool = False) -> float:
        phi = self.features(p_t, m_t, D_t, regime_post)
        return float(phi @ self.theta)

    def update(
        self,
        p_t: float,
        m_t: float,
        D_t: float,
        p_next: float,
        weight: float = 1.0,
        regime_post: bool = False,
    ) -> None:
        phi = self.features(p_t, m_t, D_t, regime_post)
        lam = max(self.config.forget_factor, 1e-6)
        ridge = max(self.config.ridge, 0.0)
        weight = max(weight, 1e-6)

        denom = lam + phi.T @ self.P @ phi + ridge
        gain = (self.P @ phi) / denom
        innovation = (p_next - float(phi @ self.theta)) * weight
        self.theta = self.theta + gain * innovation

        outer = np.outer(gain, phi.T @ self.P)
        self.P = (self.P - outer) / lam
        if ridge > 0.0:
            self.P = self.P + ridge * np.eye(5, dtype=float)  # 5x5 for 5 features
    
    def reset_covariance(self) -> None:
        """Reset RLS covariance matrix to initial state (for regime break adaptation)."""
        self.P = np.eye(5, dtype=float) * self.delta