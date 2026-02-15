"""Configuration loading for the scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class PriceModelConfig:
    enabled: bool
    method: str
    delta: float
    forget_factor: float
    ridge: float
    salience_weighting: bool
    w_scale: float
    reset_on_break: bool


@dataclass(frozen=True)
class PlanningConfig:
    enabled: bool
    K: int
    use_model: bool
    horizon: int
    recent_window: int
    gate_after_break: int
    use_salience_weights: bool
    update_actor: bool
    warmup_steps: int
    synthetic_lr_scale: float
    only_post_break: bool = False


@dataclass(frozen=True)
class MemoryConfig:
    salience_mode: str
    h0: float
    h1: float
    ema_alpha: float
    eps: float


@dataclass(frozen=True)
class ConfidenceConfig:
    memory_enabled: bool
    alpha_u: float
    psi: float
    k_bar: float
    lambda_herd: float


@dataclass(frozen=True)
class RLConfig:
    enabled: bool
    beta: float
    gamma: float
    sigma_p2: float
    eta_v: float
    eta_pi: float
    policy_sigma: float
    action_clip: float
    seed: int
    reward_mode: str
    use_belief_var: bool
    reset_critic_on_break: bool = False


@dataclass(frozen=True)
class MarketConfig:
    price_rule: str
    kappa: float
    impact: float
    noise_sigma: float
    noise_vol_scale: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    regime_break_enabled: bool
    t_break: int
    post_kappa: float | None
    post_impact: float | None
    post_sigma_v: float | None
    post_obs_noise: float | None


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int
    num_agents: int
    seed: int
    alpha_0: float
    gamma: float
    chi: float
    sigma_p2: float
    lambda_price: float
    k_init: float
    k_min: float
    k_max: float
    fundamental_mu: float
    fundamental_sigma: float
    observation_sigma: float
    observation_common_sigma: float
    rho_pos: float
    rho_neg: float
    price_model: PriceModelConfig
    planning: PlanningConfig
    memory: MemoryConfig
    confidence: ConfidenceConfig
    rl: RLConfig
    market: MarketConfig
    experiment: ExperimentConfig | None
    use_price_in_filter: bool = False
    price_obs_sigma: float = 1.0
    dt: float = 1.0
    # Optional heterogeneity in k at initialization.
    # - "fixed": all agents start at k_init
    # - "uniform": k ~ Uniform[k_min, k_max]
    # - "normal": k ~ TruncNormal(mean=k_init, std=k_std) clipped to [k_min, k_max]
    k_dist: str = "fixed"
    k_std: float = 0.0
    # Policy type: "myopic" (default) or "intertemporal" (approximate using constant-gain)
    use_intertemporal_policy: bool = False
    # Risk-denominator variance used in the one-step myopic policy:
    # - "belief" (default): use the current posterior variance Sigma_{i,t}
    # - "fixed_initial": freeze Sigma in the risk term at each agent's initial Sigma_{i,0}
    # - "fixed_steady_state": freeze Sigma in the risk term at each agent's perceived steady-state posterior variance
    # - "fixed_value": freeze Sigma in the risk term at risk_variance_sigma_value
    risk_variance_mode: str = "belief"
    risk_variance_sigma_value: float = 0.0
    # Amplification extensions (state-dependent κ, k-dependent λ). Default {}; missing keys no-ops.
    amplification: Optional[Dict[str, Any]] = None


def load_config(path: str | Path) -> SimulationConfig:
    """Load a YAML config into a SimulationConfig."""
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping of keys to values.")
    price_model_raw = raw.pop("price_model", None)
    planning_raw = raw.pop("planning", None)
    memory_raw = raw.pop("memory", None)
    confidence_raw = raw.pop("confidence", None)
    rl_raw = raw.pop("rl", None)
    market_raw = raw.pop("market", None)
    experiment_raw = raw.pop("experiment", None)
    amplification_raw = raw.pop("amplification", None)
    if not isinstance(price_model_raw, dict):
        raise ValueError("price_model section is required.")
    if not isinstance(planning_raw, dict):
        raise ValueError("planning section is required.")
    if not isinstance(rl_raw, dict):
        raise ValueError("rl section is required.")
    if not isinstance(market_raw, dict):
        raise ValueError("market section is required.")
    price_model = PriceModelConfig(**price_model_raw)
    # Set default for synthetic_lr_scale if not present
    if "synthetic_lr_scale" not in planning_raw:
        planning_raw["synthetic_lr_scale"] = 1.0
    planning = PlanningConfig(**planning_raw)
    if not isinstance(memory_raw, dict):
        raise ValueError("memory section is required.")
    if not isinstance(confidence_raw, dict):
        raise ValueError("confidence section is required.")
    memory = MemoryConfig(**memory_raw)
    confidence = ConfidenceConfig(**confidence_raw)
    rl = RLConfig(**rl_raw)
    market = MarketConfig(**market_raw)
    experiment = None
    if experiment_raw is not None:
        experiment = ExperimentConfig(**experiment_raw)
    amplification = amplification_raw if isinstance(amplification_raw, dict) else {}
    return SimulationConfig(
        price_model=price_model,
        planning=planning,
        memory=memory,
        confidence=confidence,
        rl=rl,
        market=market,
        experiment=experiment,
        amplification=amplification,
        **raw,
    )
