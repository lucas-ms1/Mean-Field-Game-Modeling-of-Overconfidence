"""Run the reference scaffold simulation."""

from __future__ import annotations

import argparse

from mfg_oc.config import load_config
from mfg_oc.simulate import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MFG overconfidence scaffold.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = simulate(
        horizon=cfg.horizon,
        num_agents=cfg.num_agents,
        params={
            "seed": cfg.seed,
            "alpha_0": cfg.alpha_0,
            "gamma": cfg.gamma,
            "chi": cfg.chi,
            "sigma_p2": cfg.sigma_p2,
            "lambda_price": cfg.lambda_price,
            "k_init": cfg.k_init,
            "k_min": cfg.k_min,
            "k_max": cfg.k_max,
            "k_dist": getattr(cfg, "k_dist", "fixed"),
            "k_std": float(getattr(cfg, "k_std", 0.0)),
            "fundamental_mu": cfg.fundamental_mu,
            "fundamental_sigma": cfg.fundamental_sigma,
            "observation_sigma": cfg.observation_sigma,
            "observation_common_sigma": cfg.observation_common_sigma,
            "rho_pos": cfg.rho_pos,
            "rho_neg": cfg.rho_neg,
            "price_model": cfg.price_model,
            "planning": cfg.planning,
            "memory": cfg.memory,
            "confidence": cfg.confidence,
            "rl": cfg.rl,
            "market": cfg.market,
            "experiment": cfg.experiment,
            "use_price_in_filter": cfg.use_price_in_filter,
            "price_obs_sigma": cfg.price_obs_sigma,
            "dt": cfg.dt,
            "use_intertemporal_policy": getattr(cfg, "use_intertemporal_policy", False),
            "risk_variance_mode": getattr(cfg, "risk_variance_mode", "belief"),
            "risk_variance_sigma_value": float(getattr(cfg, "risk_variance_sigma_value", 0.0)),
            "amplification": getattr(cfg, "amplification", None) or {},
        },
    )

    print(f"steps={len(output.prices)} price_final={output.prices[-1]:.4f}")


if __name__ == "__main__":
    main()
