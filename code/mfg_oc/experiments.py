"""Experiment harness and metrics for paper artifacts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import SimulationConfig
from .simulate import simulate


def _config_to_params(cfg: SimulationConfig, seed: int) -> Dict[str, Any]:
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
        "rl": replace(cfg.rl, seed=seed),
        "market": cfg.market,
        "experiment": cfg.experiment,
        "use_price_in_filter": cfg.use_price_in_filter,
        "price_obs_sigma": cfg.price_obs_sigma,
        "dt": cfg.dt,
        "use_intertemporal_policy": getattr(cfg, "use_intertemporal_policy", False),
        "risk_variance_mode": getattr(cfg, "risk_variance_mode", "belief"),
        "risk_variance_sigma_value": float(getattr(cfg, "risk_variance_sigma_value", 0.0)),
        "amplification": getattr(cfg, "amplification", None) or {},
    }


def run_once(cfg: SimulationConfig, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a single simulation and return summary + series."""
    output = simulate(horizon=cfg.horizon, num_agents=cfg.num_agents, params=_config_to_params(cfg, seed))
    series = _postprocess_series(output.metrics)
    # Add experiment config to series for adaptation metrics
    if cfg.experiment:
        from dataclasses import asdict
        exp_dict = asdict(cfg.experiment)
        for row in series:
            row["experiment_cfg"] = exp_dict
    summary = compute_metrics(series)
    summary["seed"] = seed
    return summary, series


def _postprocess_series(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute returns and mispricing derived fields from per-step metrics."""
    series: List[Dict[str, Any]] = []
    prev_price = metrics[0]["price"] if metrics else 0.0
    for row in metrics:
        price = float(row["price"])
        v_true = float(row["v_true"])
        mean_belief = float(row["mean_belief"])
        ret = price - prev_price
        mispricing_true = price - v_true
        mispricing_belief = price - mean_belief
        vol_proxy = abs(ret)
        enriched = dict(row)
        enriched.update(
            {
                "ret": ret,
                "mispricing_true": mispricing_true,
                "mispricing_belief": mispricing_belief,
                "vol_proxy": vol_proxy,
            }
        )
        series.append(enriched)
        prev_price = price
    return series


def compute_metrics(series: List[Dict[str, Any]], H: int = 50) -> Dict[str, Any]:
    """Compute summary metrics from a single run."""
    if not series:
        return {}
    prices = np.array([row["price"] for row in series], dtype=float)
    rets = np.array([row["ret"] for row in series], dtype=float)
    mispricing = np.array([row["mispricing_true"] for row in series], dtype=float)
    mean_k = np.array([row["mean_k"] for row in series], dtype=float)
    xs_k_std = np.array([row.get("xs_k_std", 0.0) for row in series], dtype=float)
    demand = np.array([row["total_demand"] for row in series], dtype=float)
    xs_action_std = np.array([row.get("xs_action_std", 0.0) for row in series], dtype=float)
    xs_action_mean_abs = np.array([row.get("xs_action_mean_abs", 0.0) for row in series], dtype=float)
    xs_belief_std = np.array([row.get("xs_belief_std", 0.0) for row in series], dtype=float)
    xs_sigma_mean = np.array([row.get("xs_sigma_mean", 0.0) for row in series], dtype=float)
    xs_rho_mean = np.array([row.get("xs_rho_mean", 0.0) for row in series], dtype=float)
    xs_rho_max = np.array([row.get("xs_rho_max", 0.0) for row in series], dtype=float)

    abs_ret = np.abs(rets)
    abs_mis = np.abs(mispricing)
    ret_std = float(np.std(rets))
    abs_mis_mean = float(np.mean(abs_mis))
    abs_mis_median = float(np.median(abs_mis))
    abs_mis_p95 = float(np.percentile(abs_mis, 95))
    abs_mis_p99 = float(np.percentile(abs_mis, 99))
    abs_mis_max = float(np.max(abs_mis))
    k_mean = float(np.mean(mean_k))
    # Cross-sectional std of k_i across agents, time-averaged (0 when k is homogeneous)
    k_std = float(np.mean(xs_k_std)) if len(xs_k_std) > 0 else float(np.std(mean_k))
    demand_std = float(np.std(demand))
    xs_action_std_mean = float(np.mean(xs_action_std))
    xs_action_mean_abs_mean = float(np.mean(xs_action_mean_abs))
    xs_belief_std_mean = float(np.mean(xs_belief_std))
    xs_sigma_mean_mean = float(np.mean(xs_sigma_mean))
    xs_rho_mean_mean = float(np.mean(xs_rho_mean))
    xs_rho_max_mean = float(np.mean(xs_rho_max))
    if len(abs_ret) > 1:
        ac1 = np.corrcoef(abs_ret[1:], abs_ret[:-1])[0, 1]
        abs_ret_ac1 = float(ac1) if not np.isnan(ac1) else 0.0
    else:
        abs_ret_ac1 = 0.0

    if len(mispricing) > 1:
        mis_ac1 = np.corrcoef(mispricing[1:], mispricing[:-1])[0, 1]
        mispricing_ac1 = float(mis_ac1) if not np.isnan(mis_ac1) else 0.0
        abs_mis_ac1 = np.corrcoef(abs_mis[1:], abs_mis[:-1])[0, 1]
        abs_mispricing_ac1 = float(abs_mis_ac1) if not np.isnan(abs_mis_ac1) else 0.0
    else:
        mispricing_ac1 = 0.0
        abs_mispricing_ac1 = 0.0
    
    # Amplification metrics: mispricing persistence (half-life)
    mispricing_halflife = None
    if len(abs_mis) > 10:
        # Estimate half-life using autocorrelation decay
        # For AR(1) process: half-life = -ln(2) / ln(ρ) where ρ is autocorrelation
        if abs_mispricing_ac1 > 0 and abs_mispricing_ac1 < 1:
            mispricing_halflife = float(-np.log(2.0) / np.log(abs_mispricing_ac1))
        else:
            # Fallback: find time for mispricing to decay to half its initial value
            # Use exponential decay fit on autocorrelation function
            mispricing_halflife = float(abs_mispricing_ac1 * len(abs_mis)) if abs_mispricing_ac1 > 0 else 0.0
    
    # Volatility clustering: GARCH-like measures
    # Autocorrelation of squared returns (volatility clustering)
    ret_squared = rets ** 2
    volatility_clustering_ac1 = 0.0
    if len(ret_squared) > 1:
        vol_ac1 = np.corrcoef(ret_squared[1:], ret_squared[:-1])[0, 1]
        volatility_clustering_ac1 = float(vol_ac1) if not np.isnan(vol_ac1) else 0.0
    
    # Higher-order autocorrelations for volatility clustering
    volatility_clustering_ac5 = 0.0
    if len(ret_squared) > 5:
        vol_ac5 = np.corrcoef(ret_squared[5:], ret_squared[:-5])[0, 1]
        volatility_clustering_ac5 = float(vol_ac5) if not np.isnan(vol_ac5) else 0.0

    # Event study metrics - use salience-based events (aligned with memory mechanism)
    event_decay = event_study_k_decay(series, H=H)
    event_study_k_mean = event_decay.tolist()
    
    # Get arrays for event study
    mean_k = np.array([row["mean_k"] for row in series], dtype=float)
    mean_salience = np.array([row.get("mean_salience", 0.0) for row in series], dtype=float)
    mean_correctness = np.array([row.get("mean_correctness", 0.0) for row in series], dtype=float)
    T = len(mean_k)
    
    event_study_k_abs = None
    event_study_k_norm = None
    event_study_k_halflife = None
    event_study_k_by_correctness = {"correct": None, "incorrect": None}
    
    if T > 0 and len(mean_salience) > 0:
        # Define events as top 1% of salience (at least 5 events)
        n_events = max(5, int(0.01 * T))
        if n_events < len(mean_salience):
            event_indices = np.argsort(mean_salience)[-n_events:]
            events = np.sort(event_indices)
        else:
            events = np.arange(T)
        
        if len(events) > 0:
            # Compute absolute deviation persistence
            abs_responses = []
            norm_responses = []
            halflives = []
            correct_responses = []
            incorrect_responses = []
            
            for idx in events:
                if idx + H >= len(mean_k):
                    continue
                # Pre-shock baseline: mean over [t0-5, t0)
                pre_start = max(idx - 5, 0)
                pre_end = max(idx - 1, 0)
                if pre_end < pre_start:
                    k_pre = mean_k[idx] if idx < len(mean_k) else 0.0
                else:
                    k_pre = float(np.mean(mean_k[pre_start:pre_end + 1]))
                
                # Deviation curve
                response = mean_k[idx : idx + H + 1] - k_pre
                d0 = response[0] if len(response) > 0 else 0.0
                
                # Normalized curve (scale-free)
                abs_d0 = max(abs(d0), 1e-6)
                norm_curve = np.abs(response) / abs_d0
                
                abs_responses.append(np.abs(response))
                norm_responses.append(norm_curve)
                
                # Half-life: smallest h such that norm_curve[h] <= 0.5
                halflife = H
                for h in range(len(norm_curve)):
                    if norm_curve[h] <= 0.5:
                        halflife = h
                        break
                halflives.append(halflife)
                
                # Split by correctness at shock time
                correctness_at_t0 = mean_correctness[idx] if idx < len(mean_correctness) else 0.0
                if correctness_at_t0 >= 0:
                    correct_responses.append(response)
                else:
                    incorrect_responses.append(response)
            
            if abs_responses:
                event_study_k_abs = np.mean(np.stack(abs_responses, axis=0), axis=0).tolist()
            if norm_responses:
                event_study_k_norm = np.mean(np.stack(norm_responses, axis=0), axis=0).tolist()
            if halflives:
                event_study_k_halflife = float(np.mean(halflives))
            if correct_responses:
                event_study_k_by_correctness["correct"] = np.mean(np.stack(correct_responses, axis=0), axis=0).tolist()
            if incorrect_responses:
                event_study_k_by_correctness["incorrect"] = np.mean(np.stack(incorrect_responses, axis=0), axis=0).tolist()
    
    # Default to zeros if no events found
    if event_study_k_abs is None:
        event_study_k_abs = [0.0] * (H + 1)
    if event_study_k_norm is None:
        event_study_k_norm = [0.0] * (H + 1)
    if event_study_k_halflife is None:
        event_study_k_halflife = float(H)
    if event_study_k_by_correctness["correct"] is None:
        event_study_k_by_correctness["correct"] = [0.0] * (H + 1)
    if event_study_k_by_correctness["incorrect"] is None:
        event_study_k_by_correctness["incorrect"] = [0.0] * (H + 1)

    result = {
        "ret_std": ret_std,
        "abs_mispricing_mean": abs_mis_mean,
        "abs_mispricing_median": abs_mis_median,
        "abs_mispricing_p95": abs_mis_p95,
        "abs_mispricing_p99": abs_mis_p99,
        "abs_mispricing_max": abs_mis_max,
        "mispricing_ac1": mispricing_ac1,
        "abs_mispricing_ac1": abs_mispricing_ac1,
        "k_mean": k_mean,
        "k_std": k_std,
        "demand_std": demand_std,
        "xs_action_std_mean": xs_action_std_mean,
        "xs_action_mean_abs_mean": xs_action_mean_abs_mean,
        "xs_belief_std_mean": xs_belief_std_mean,
        "xs_sigma_mean_mean": xs_sigma_mean_mean,
        "xs_rho_mean_mean": xs_rho_mean_mean,
        "xs_rho_max_mean": xs_rho_max_mean,
        "abs_ret_ac1": abs_ret_ac1,
        "event_study_k_decay": event_study_k_mean,  # Keep old name for backward compatibility
        "event_study_k_mean": event_study_k_mean,
        "event_study_k_abs": event_study_k_abs,
        "event_study_k_norm": event_study_k_norm,
        "event_study_k_halflife": event_study_k_halflife,
        "event_study_k_by_correctness": event_study_k_by_correctness,
        # Amplification metrics
        "mispricing_halflife": mispricing_halflife,
        "volatility_clustering_ac1": volatility_clustering_ac1,
        "volatility_clustering_ac5": volatility_clustering_ac5,
    }
    
    # Regime-break adaptation metrics
    experiment_cfg = series[0].get("experiment_cfg", None) if series else None
    regime_break_enabled = experiment_cfg is not None and experiment_cfg.get("regime_break_enabled", False) if experiment_cfg else False
    t_break = experiment_cfg.get("t_break", 0) if experiment_cfg else 0
    
    mispricing_overshoot = None
    recovery_time_mispricing = None
    vol_jump = None
    cluster_jump = None
    
    if regime_break_enabled and t_break > 0 and len(series) > t_break:
        # Find t_break index
        t_break_idx = None
        for i, row in enumerate(series):
            if int(row.get("t", 0)) == t_break:
                t_break_idx = i
                break
        
        if t_break_idx is not None and t_break_idx >= 100:
            pre_window = max(0, t_break_idx - 100)
            pre_end = t_break_idx - 1
            post_start = t_break_idx
            post_end = min(len(series) - 1, t_break_idx + 199)
            
            # Pre-window metrics
            pre_mispricing = np.array([abs(row.get("mispricing_true", 0.0)) for row in series[pre_window:pre_end + 1]], dtype=float)
            pre_rets = np.array([row.get("ret", 0.0) for row in series[pre_window:pre_end + 1]], dtype=float)
            pre_abs_rets = np.abs(pre_rets)
            pre_mean_mispricing = float(np.mean(pre_mispricing))
            pre_ret_std = float(np.std(pre_rets))
            if len(pre_abs_rets) > 1:
                pre_abs_ret_ac1 = float(np.corrcoef(pre_abs_rets[1:], pre_abs_rets[:-1])[0, 1]) if not np.isnan(np.corrcoef(pre_abs_rets[1:], pre_abs_rets[:-1])[0, 1]) else 0.0
            else:
                pre_abs_ret_ac1 = 0.0
            
            # Post-window metrics
            post_mispricing = np.array([abs(row.get("mispricing_true", 0.0)) for row in series[post_start:post_end + 1]], dtype=float)
            post_rets = np.array([row.get("ret", 0.0) for row in series[post_start:post_end + 1]], dtype=float)
            post_abs_rets = np.abs(post_rets)
            post_ret_std = float(np.std(post_rets))
            if len(post_abs_rets) > 1:
                post_abs_ret_ac1 = float(np.corrcoef(post_abs_rets[1:], post_abs_rets[:-1])[0, 1]) if not np.isnan(np.corrcoef(post_abs_rets[1:], post_abs_rets[:-1])[0, 1]) else 0.0
            else:
                post_abs_ret_ac1 = 0.0
            
            # 1) Mispricing overshoot: max in first 50 steps after break minus pre mean
            overshoot_window = min(50, len(post_mispricing))
            if overshoot_window > 0:
                max_post_mispricing = float(np.max(post_mispricing[:overshoot_window]))
                mispricing_overshoot = max_post_mispricing - pre_mean_mispricing
            
            # 2) Recovery time: smallest h where rolling 20-step mean <= 1.1 * pre_mean
            recovery_time_mispricing = 200  # default if never recovers
            for h in range(len(post_mispricing) - 20):
                rolling_mean = float(np.mean(post_mispricing[h:h+20]))
                if rolling_mean <= 1.1 * pre_mean_mispricing:
                    recovery_time_mispricing = h
                    break
            
            # 3) Volatility jump
            vol_jump = post_ret_std - pre_ret_std
            
            # 4) Clustering jump
            cluster_jump = post_abs_ret_ac1 - pre_abs_ret_ac1
    
    # Add adaptation metrics to return dict
    if mispricing_overshoot is not None:
        result["mispricing_overshoot"] = mispricing_overshoot
    if recovery_time_mispricing is not None:
        result["recovery_time_mispricing"] = recovery_time_mispricing
    if vol_jump is not None:
        result["vol_jump"] = vol_jump
    if cluster_jump is not None:
        result["cluster_jump"] = cluster_jump
    
    return result


def event_study_k_decay(series: List[Dict[str, Any]], H: int = 50) -> np.ndarray:
    """Event-study response of mean_k after top 1% |ret| shocks.
    
    Returns average deviation curve d_h = mean_k[t0+h] - k_pre for h=0..H.
    """
    rets = np.array([row["ret"] for row in series], dtype=float)
    mean_k = np.array([row["mean_k"] for row in series], dtype=float)
    if len(rets) == 0:
        return np.zeros(H + 1, dtype=float)
    threshold = np.percentile(np.abs(rets), 99)
    events = np.where(np.abs(rets) >= threshold)[0]
    if len(events) == 0:
        return np.zeros(H + 1, dtype=float)
    responses = []
    for idx in events:
        if idx + H >= len(mean_k):
            continue
        pre_start = max(idx - 5, 0)
        pre_end = max(idx - 1, 0)
        if pre_end < pre_start:
            pre_mean = mean_k[idx] if idx < len(mean_k) else 0.0
        else:
            pre_mean = float(np.mean(mean_k[pre_start:pre_end + 1]))
        response = mean_k[idx : idx + H + 1] - pre_mean
        responses.append(response)
    if not responses:
        return np.zeros(H + 1, dtype=float)
    return np.mean(np.stack(responses, axis=0), axis=0)


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics over multiple runs."""
    if not runs:
        return {}
    agg: Dict[str, Any] = {}
    keys = [k for k in runs[0].keys() if k != "seed"]
    for key in keys:
        if key in ("event_study_k_decay", "event_study_k_mean", "event_study_k_abs", "event_study_k_norm"):
            arrays = [np.array(r[key], dtype=float) for r in runs]
            agg[key] = np.mean(np.stack(arrays, axis=0), axis=0).tolist()
        elif key == "event_study_k_halflife":
            # Average halflife across runs
            agg[key] = float(np.mean([r.get(key, 50.0) for r in runs]))
        elif key == "event_study_k_by_correctness":
            # Aggregate the nested dict structure
            agg[key] = {}
            for correctness_key in ("correct", "incorrect"):
                arrays = []
                for r in runs:
                    nested = r.get(key, {})
                    arr = nested.get(correctness_key, [0.0])
                    arrays.append(np.array(arr, dtype=float))
                if arrays:
                    agg[key][correctness_key] = np.mean(np.stack(arrays, axis=0), axis=0).tolist()
                else:
                    agg[key][correctness_key] = [0.0]
        else:
            agg[key] = float(np.mean([r[key] for r in runs]))
    return agg
