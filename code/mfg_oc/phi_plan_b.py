"""
Plan B: variance-consistent calibration of phi (price-noise volatility).

Implements a simulation-based moments() function and a robust bisection calibrator
that matches a target return volatility while respecting the model-implied variance
decomposition of price increments under anchored_impact.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .config import MarketConfig, SimulationConfig
from .simulate import simulate


def _config_to_params(cfg: SimulationConfig, seed: int, *, log_metrics: bool) -> Dict[str, Any]:
    # Keep this in sync with mfg_oc.experiments._config_to_params, but avoid importing
    # a private helper into library code.
    rl_cfg = replace(cfg.rl, seed=seed)
    return {
        "seed": seed,
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
        "rl": rl_cfg,
        "market": cfg.market,
        "experiment": cfg.experiment,
        "use_price_in_filter": cfg.use_price_in_filter,
        "price_obs_sigma": cfg.price_obs_sigma,
        "dt": cfg.dt,
        "use_intertemporal_policy": getattr(cfg, "use_intertemporal_policy", False),
        "risk_variance_mode": getattr(cfg, "risk_variance_mode", "belief"),
        "risk_variance_sigma_value": float(getattr(cfg, "risk_variance_sigma_value", 0.0)),
        "amplification": getattr(cfg, "amplification", None) or {},
        "log_metrics": bool(log_metrics),
    }


def _validate_window(num_increments: int, burn: int, sample: int) -> Tuple[int, int]:
    burn_i = int(max(burn, 0))
    sample_i = int(max(sample, 0))
    if burn_i >= num_increments:
        raise ValueError(f"burn={burn_i} must be < number of increments ({num_increments}).")
    stop = burn_i + sample_i
    if sample_i <= 0:
        raise ValueError("sample must be > 0.")
    if stop > num_increments:
        raise ValueError(
            f"burn+sample={stop} must be <= number of increments ({num_increments})."
        )
    return burn_i, stop


def component_increments_from_paths(
    *,
    prices: Sequence[float],
    fundamentals: Sequence[float],
    total_demands: Sequence[float],
    num_agents: int,
    dt: float,
    kappa: float,
    impact: float,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct (dp, A, B, C) from the minimal logged paths.

    Logging convention (mfg_oc.simulate):
    - At simulation iteration t, the code updates v and then updates price to p_{t+1},
      and logs price=p_{t+1}, v_true=v_{t+1}, total_demand=D_t.
    - Therefore, the price increment dp_i := p_i - p_{i-1} aligns with v_i and D_i
      (for i >= 1). We use this alignment in the decomposition.
    """
    p = np.asarray(prices, dtype=float)
    v = np.asarray(fundamentals, dtype=float)
    d = np.asarray(total_demands, dtype=float)
    if not (len(p) == len(v) == len(d)):
        raise ValueError("prices, fundamentals, and total_demands must have the same length.")
    if len(p) < 2:
        raise ValueError("Need at least 2 logged prices to compute increments.")

    dt = float(dt)
    dt = max(dt, 0.0)
    n = max(int(num_agents), 1)

    dp = p[1:] - p[:-1]
    mispricing = v[1:] - p[:-1]  # anchor (v_i) minus previous price (p_{i-1})
    bar_u = d[1:] / n

    A = float(kappa) * mispricing * dt
    B = float(impact) * bar_u * dt
    C = dp - A - B  # realized residual (noise + any approximation error)

    return {
        "dp": dp,
        "A": A,
        "B": B,
        "C": C,
        "mispricing": mispricing,
        "bar_u": bar_u,
    }


def moments_from_paths(
    *,
    prices: Sequence[float],
    fundamentals: Sequence[float],
    total_demands: Sequence[float],
    num_agents: int,
    dt: float,
    kappa: float,
    impact: float,
    phi: float,
    burn: int,
    sample: int,
    scale_s: float = 1.0,
) -> Dict[str, float]:
    """
    Compute stationary moments and a variance reconstruction check on a window.

    Returns moments in price-increment units (dp) plus return-volatility units (scale_s * dp).
    """
    comps = component_increments_from_paths(
        prices=prices,
        fundamentals=fundamentals,
        total_demands=total_demands,
        num_agents=num_agents,
        dt=dt,
        kappa=kappa,
        impact=impact,
    )
    dp = comps["dp"]
    A = comps["A"]
    B = comps["B"]
    C = comps["C"]
    mispricing = comps["mispricing"]
    bar_u = comps["bar_u"]

    burn_i, stop = _validate_window(len(dp), burn=burn, sample=sample)
    sl = slice(burn_i, stop)

    dp_w = dp[sl]
    A_w = A[sl]
    B_w = B[sl]
    C_w = C[sl]
    m_w = mispricing[sl]
    u_w = bar_u[sl]

    def _var(x: np.ndarray) -> float:
        return float(np.var(x, ddof=0))

    def _cov(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))

    var_dp = _var(dp_w)
    var_A = _var(A_w)
    var_B = _var(B_w)
    var_C_resid = _var(C_w)
    cov_AB = _cov(A_w, B_w)
    cov_AC = _cov(A_w, C_w)
    cov_BC = _cov(B_w, C_w)

    var_C_model = float(phi) ** 2 * float(dt)

    decomp_rhs_resid = var_A + var_B + var_C_resid + 2.0 * cov_AB + 2.0 * cov_AC + 2.0 * cov_BC
    decomp_rhs_model = var_A + var_B + var_C_model + 2.0 * cov_AB  # Plan-B identity under independence

    sigma_r_sim = float(np.std(float(scale_s) * dp_w, ddof=0))

    return {
        # Return target object
        "sigma_r_sim": sigma_r_sim,
        # Total dp variance and components
        "var_dp": var_dp,
        "var_A": var_A,
        "var_B": var_B,
        "var_C_resid": var_C_resid,
        "var_C_model": var_C_model,
        "cov_AB": cov_AB,
        "cov_AC": cov_AC,
        "cov_BC": cov_BC,
        # State/control moments
        "var_m": _var(m_w),
        "var_ubar": _var(u_w),
        "cov_m_ubar": _cov(m_w, u_w),
        # Decomposition diagnostics
        "decomp_err_full": float(var_dp - decomp_rhs_resid),
        "decomp_err_plan_b": float(var_dp - decomp_rhs_model),
    }


def moments(
    *,
    base_cfg: SimulationConfig,
    phi: float,
    seeds: Sequence[int],
    burn: int,
    sample: int,
    kappa: Optional[float] = None,
    impact: Optional[float] = None,
    scale_s: float = 1.0,
    log_metrics: bool = False,
) -> Dict[str, float]:
    """
    Simulate moments under candidate phi, averaged across seeds (common random numbers).

    Note: also sets sigma_p2 := phi^2 in the myopic risk denominator for internal consistency.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    dt = float(base_cfg.dt)
    kappa_val = float(kappa) if kappa is not None else float(base_cfg.market.kappa)
    impact_val = float(impact) if impact is not None else float(base_cfg.market.impact)

    per_seed: List[Dict[str, float]] = []
    for seed in seeds:
        market = replace(base_cfg.market, kappa=kappa_val, impact=impact_val, noise_sigma=float(phi))
        cfg = replace(
            base_cfg,
            seed=int(seed),
            sigma_p2=float(phi) ** 2,
            rl=replace(base_cfg.rl, sigma_p2=float(phi) ** 2, seed=int(seed)),
            market=market,
        )
        params = _config_to_params(cfg, seed=int(seed), log_metrics=log_metrics)
        out = simulate(horizon=cfg.horizon, num_agents=cfg.num_agents, params=params)
        per_seed.append(
            moments_from_paths(
                prices=out.prices,
                fundamentals=out.fundamentals,
                total_demands=out.demands,
                num_agents=cfg.num_agents,
                dt=dt,
                kappa=kappa_val,
                impact=impact_val,
                phi=float(phi),
                burn=burn,
                sample=sample,
                scale_s=scale_s,
            )
        )

    # Average moments across seeds; match volatility on the variance scale for stability.
    keys = list(per_seed[0].keys())
    out: Dict[str, float] = {"phi": float(phi), "kappa": kappa_val, "impact": impact_val, "dt": dt}
    # Volatility aggregation: sigma = sqrt(mean(var))
    var_r = np.array([m["sigma_r_sim"] ** 2 for m in per_seed], dtype=float)
    out["sigma_r_sim"] = float(math.sqrt(float(np.mean(var_r))))
    out["sigma_r_sim_se"] = float(np.std(np.sqrt(var_r), ddof=0) / math.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0

    for k in keys:
        if k == "sigma_r_sim":
            continue
        vals = np.array([m[k] for m in per_seed], dtype=float)
        out[k] = float(np.mean(vals))
        out[f"{k}_se"] = float(np.std(vals, ddof=0) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0

    out["n_seeds"] = float(len(per_seed))
    out["burn"] = float(burn)
    out["sample"] = float(sample)
    return out


def calibrate_phi_bisection(
    *,
    base_cfg: SimulationConfig,
    sigma_r_target: float,
    scale_s: float,
    seeds: Sequence[int],
    burn: int,
    sample: int,
    kappa: Optional[float] = None,
    impact: Optional[float] = None,
    phi_min: float = 0.0,
    phi_max: Optional[float] = None,
    tol_abs: float = 1e-4,
    tol_rel: float = 5e-3,
    max_iter: int = 30,
    phi_cap: float = 1e3,
    log_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Solve for phi so that std(scale_s * dp_sim(phi)) matches sigma_r_target.

    Uses bisection with common random numbers (fixed seeds).
    Returns a dict containing phi_star and moments at the solution.
    """
    sigma_r_target = float(sigma_r_target)
    scale_s = float(scale_s)
    if not (sigma_r_target >= 0.0):
        raise ValueError("sigma_r_target must be non-negative.")
    if not (scale_s > 0.0):
        raise ValueError("scale_s must be > 0.")

    cache: Dict[float, Dict[str, float]] = {}

    def eval_g(phi: float) -> Tuple[float, Dict[str, float]]:
        phi = float(phi)
        if phi in cache:
            m = cache[phi]
        else:
            m = moments(
                base_cfg=base_cfg,
                phi=phi,
                seeds=seeds,
                burn=burn,
                sample=sample,
                kappa=kappa,
                impact=impact,
                scale_s=scale_s,
                log_metrics=log_metrics,
            )
            cache[phi] = m
        g = float(m["sigma_r_sim"] - sigma_r_target)
        return g, m

    phi_lo = max(float(phi_min), 0.0)
    if phi_max is None:
        # Initial heuristic: old "sigma_r ≈ s*phi" mapping, capped away from 0.
        phi_hi = max(sigma_r_target / scale_s, 1e-6)
    else:
        phi_hi = float(phi_max)

    g_lo, m_lo = eval_g(phi_lo)
    if g_lo >= 0.0:
        # Endogenous volatility already meets/exceeds target at phi=0.
        return {
            "phi_star": 0.0,
            "sigma_r_target": sigma_r_target,
            "scale_s": scale_s,
            "g(phi_star)": g_lo,
            "moments": m_lo,
            "bracket": (phi_lo, phi_lo),
            "n_eval": len(cache),
            "status": "phi_star=0 (target met without exogenous noise)",
        }

    g_hi, m_hi = eval_g(phi_hi)
    while g_hi < 0.0 and phi_hi < float(phi_cap):
        phi_hi = min(2.0 * phi_hi, float(phi_cap))
        g_hi, m_hi = eval_g(phi_hi)

    if g_hi < 0.0:
        return {
            "phi_star": None,
            "sigma_r_target": sigma_r_target,
            "scale_s": scale_s,
            "bracket": (phi_lo, phi_hi),
            "g_lo": g_lo,
            "g_hi": g_hi,
            "moments_lo": m_lo,
            "moments_hi": m_hi,
            "n_eval": len(cache),
            "status": "bracketing_failed (even phi_cap too small)",
        }

    phi_a, phi_b = phi_lo, phi_hi
    g_a, g_b = g_lo, g_hi
    m_mid: Dict[str, float] = m_hi
    for _ in range(int(max_iter)):
        phi_mid = 0.5 * (phi_a + phi_b)
        g_mid, m_mid = eval_g(phi_mid)
        sigma_mid = float(m_mid["sigma_r_sim"])
        err_abs = abs(sigma_mid - sigma_r_target)
        err_rel = err_abs / max(sigma_r_target, 1e-12)
        if err_abs <= float(tol_abs) or err_rel <= float(tol_rel):
            return {
                "phi_star": float(phi_mid),
                "sigma_r_target": sigma_r_target,
                "scale_s": scale_s,
                "g(phi_star)": float(g_mid),
                "moments": m_mid,
                "bracket": (float(phi_a), float(phi_b)),
                "n_eval": len(cache),
                "status": "converged",
            }
        if g_mid > 0.0:
            phi_b, g_b = phi_mid, g_mid
        else:
            phi_a, g_a = phi_mid, g_mid

    return {
        "phi_star": float(0.5 * (phi_a + phi_b)),
        "sigma_r_target": sigma_r_target,
        "scale_s": scale_s,
        "g(phi_star)": float(m_mid["sigma_r_sim"] - sigma_r_target),
        "moments": m_mid,
        "bracket": (float(phi_a), float(phi_b)),
        "n_eval": len(cache),
        "status": "max_iter_reached",
    }


def calibrate_phi_fixed_point(
    *,
    base_cfg: SimulationConfig,
    sigma_r_target: float,
    scale_s: float,
    seeds: Sequence[int],
    burn: int,
    sample: int,
    kappa: Optional[float] = None,
    impact: Optional[float] = None,
    phi_init: Optional[float] = None,
    tol_abs: float = 1e-4,
    tol_rel: float = 5e-3,
    max_iter: int = 8,
    damping: float = 0.5,
    log_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Faster alternative: fixed-point iteration using the Plan-B variance identity.

    Iterates:
      phi_{new} = sqrt(max(target_var_dp - Var(A+B), 0) / dt)
    where Var(A+B) is recomputed under each candidate phi (so it is not the one-shot residual).

    This typically converges in a handful of simulations and can be used to seed a short bisection.
    """
    sigma_r_target = float(sigma_r_target)
    scale_s = float(scale_s)
    if not (sigma_r_target >= 0.0):
        raise ValueError("sigma_r_target must be non-negative.")
    if not (scale_s > 0.0):
        raise ValueError("scale_s must be > 0.")

    dt = float(base_cfg.dt)
    target_var_dp = (sigma_r_target / scale_s) ** 2

    if phi_init is None:
        phi = max(sigma_r_target / scale_s, 0.0)
    else:
        phi = max(float(phi_init), 0.0)

    damping = float(damping)
    damping = min(max(damping, 0.0), 1.0)

    last_m: Dict[str, float] | None = None
    for it in range(int(max_iter)):
        m = moments(
            base_cfg=base_cfg,
            phi=phi,
            seeds=seeds,
            burn=burn,
            sample=sample,
            kappa=kappa,
            impact=impact,
            scale_s=scale_s,
            log_metrics=log_metrics,
        )
        last_m = m
        var_ab = float(m["var_A"] + m["var_B"] + 2.0 * m["cov_AB"])
        resid = target_var_dp - var_ab
        phi_new = math.sqrt(max(resid, 0.0) / max(dt, 1e-18))
        phi_next = (1.0 - damping) * float(phi) + damping * float(phi_new)

        # Convergence check in volatility space (matches the calibration target)
        sigma_sim = float(m["sigma_r_sim"])
        err_abs = abs(sigma_sim - sigma_r_target)
        err_rel = err_abs / max(sigma_r_target, 1e-12)
        if err_abs <= float(tol_abs) or err_rel <= float(tol_rel):
            return {
                "phi_star": float(phi),
                "sigma_r_target": sigma_r_target,
                "scale_s": scale_s,
                "g(phi_star)": float(sigma_sim - sigma_r_target),
                "moments": m,
                "n_iter": it + 1,
                "status": "converged",
            }

        # If the update collapses to 0 and we're below target, stop early.
        if phi_next <= 0.0 and sigma_sim < sigma_r_target:
            phi_next = 0.0

        phi = phi_next

    return {
        "phi_star": float(phi),
        "sigma_r_target": sigma_r_target,
        "scale_s": scale_s,
        "g(phi_star)": float((last_m["sigma_r_sim"] - sigma_r_target) if last_m else float("nan")),
        "moments": last_m,
        "n_iter": int(max_iter),
        "status": "max_iter_reached",
    }


def variance_share_rows_from_moments(m: Mapping[str, float]) -> List[Dict[str, float]]:
    """Compute variance contributions and shares from a moments() output dict."""
    V = float(m["var_dp"])
    VA = float(m["var_A"])
    VB = float(m["var_B"])
    VC = float(m["var_C_resid"])
    VAB = 2.0 * float(m["cov_AB"])
    VAC = 2.0 * float(m["cov_AC"])
    VBC = 2.0 * float(m["cov_BC"])
    denom = V if abs(V) > 1e-18 else 1.0
    return [
        {"label": r"Fundamental pull $\Var(A)$", "value": VA, "share": 100.0 * VA / denom},
        {"label": r"Endogenous flow $\Var(B)$", "value": VB, "share": 100.0 * VB / denom},
        {"label": r"Exogenous noise $\Var(C)$ (realized)", "value": VC, "share": 100.0 * VC / denom},
        {"label": r"Cross term $2\Cov(A,B)$", "value": VAB, "share": 100.0 * VAB / denom},
        {"label": r"Cross term $2\Cov(A,C)$", "value": VAC, "share": 100.0 * VAC / denom},
        {"label": r"Cross term $2\Cov(B,C)$", "value": VBC, "share": 100.0 * VBC / denom},
        {"label": r"Total $\Var(\Delta p)$", "value": V, "share": 100.0},
    ]


def render_variance_share_table_tex(
    m: Mapping[str, float],
    *,
    value_fmt: str = ".4g",
    share_fmt: str = ".1f",
) -> str:
    """Render a small LaTeX tabular body for variance shares under phi*."""
    rows = variance_share_rows_from_moments(m)
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Component & Variance & Share (\%) \\", r"\midrule"]
    for row in rows:
        label = str(row["label"])
        value = float(row["value"])
        share = float(row["share"])
        lines.append(f"{label} & {value:{value_fmt}} & {share:{share_fmt}} \\\\")
    V = float(m["var_dp"])
    denom = V if abs(V) > 1e-18 else 1.0

    # Plan-B identity gap (assumes Cov(A,C)=Cov(B,C)=0 and Var(C)=phi^2 dt).
    rhs_plan_b = float(m["var_A"] + m["var_B"] + m["var_C_model"] + 2.0 * m["cov_AB"])
    err_plan_b = V - rhs_plan_b
    err_plan_b_pct = 100.0 * err_plan_b / denom

    # Accounting identity gap using realized residual C := dp - A - B.
    rhs_full = float(
        m["var_A"]
        + m["var_B"]
        + m["var_C_resid"]
        + 2.0 * m["cov_AB"]
        + 2.0 * m["cov_AC"]
        + 2.0 * m["cov_BC"]
    )
    err_full = V - rhs_full
    err_full_pct = 100.0 * err_full / denom

    lines += [
        r"\midrule",
        f"Plan-B gap & {err_plan_b:{value_fmt}} & {err_plan_b_pct:{share_fmt}} \\\\",
        f"Accounting gap & {err_full:{value_fmt}} & {err_full_pct:{share_fmt}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"
