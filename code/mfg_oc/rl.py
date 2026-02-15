"""Online actor-critic with linear value and Gaussian policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import RLConfig
from .types import Transition, state_vector


@dataclass
class ActorCriticLinearGaussian:
    dim: int
    cfg: RLConfig
    rng: np.random.Generator
    theta: np.ndarray
    w: np.ndarray
    n_real_updates: int = 0
    n_planning_updates: int = 0
    avg_abs_delta: float = 0.0
    last: Dict[str, float] | None = None

    def __init__(self, dim: int, cfg: RLConfig, rng: np.random.Generator) -> None:
        self.dim = dim
        self.cfg = cfg
        self.rng = rng
        self.theta = np.zeros(dim, dtype=float)
        self.w = np.zeros(dim, dtype=float)

    def act(self, s_vec: np.ndarray) -> float:
        mu = float(self.theta @ s_vec)
        if self.cfg.action_clip > 0.0:
            mu = float(np.clip(mu, -self.cfg.action_clip, self.cfg.action_clip))
        a = float(mu + self.rng.normal(0.0, self.cfg.policy_sigma))
        if self.cfg.action_clip > 0.0:
            a = float(np.clip(a, -self.cfg.action_clip, self.cfg.action_clip))
        self.last = {"mu": mu, "a": a}
        return a

    def value(self, s_vec: np.ndarray) -> float:
        return float(self.w @ s_vec)

    def reset_critic(self) -> None:
        """Reset critic weight vector to zeros without changing shape."""
        self.w = np.zeros_like(self.w)

    def update_from_transition(self, tr: Transition, synthetic: bool, update_actor: bool = True, lr_scale: float = 1.0) -> Dict[str, float]:
        s = state_vector(tr.state.agent, tr.state.market)
        s_next = state_vector(tr.next_state.agent, tr.next_state.market)
        mu = float(self.theta @ s)
        a = float(tr.action)
        if self.cfg.action_clip > 0.0:
            a = float(np.clip(a, -self.cfg.action_clip, self.cfg.action_clip))

        v = float(self.w @ s)
        v_next = float(self.w @ s_next)
        delta = tr.reward + self.cfg.beta * v_next - v
        max_delta = 10.0 * max(1.0, self.cfg.action_clip)
        delta = float(np.clip(delta, -max_delta, max_delta))
        
        # Always update critic with lr_scale applied
        lr_scale_eff = max(0.0, float(lr_scale))
        alpha_w_eff = self.cfg.eta_v * lr_scale_eff
        self.w = self.w + alpha_w_eff * delta * s

        # Only update actor if update_actor is True
        if update_actor:
            grad_logp = ((a - mu) / (self.cfg.policy_sigma ** 2)) * s
            grad_logp = np.clip(grad_logp, -max_delta, max_delta)
            self.theta = self.theta + self.cfg.eta_pi * delta * grad_logp

        self.avg_abs_delta = 0.9 * self.avg_abs_delta + 0.1 * abs(delta)
        if synthetic:
            self.n_planning_updates += 1
        else:
            self.n_real_updates += 1

        out = {"delta": float(delta), "r": float(tr.reward), "mu": mu, "a": a, "synthetic": float(synthetic)}
        self.last = out
        return out
