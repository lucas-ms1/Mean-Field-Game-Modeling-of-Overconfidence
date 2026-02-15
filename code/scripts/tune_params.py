"""Grid search parameter tuning to find a publishable regime under anchored_impact."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfg_oc.config import SimulationConfig, load_config
from mfg_oc.experiments import aggregate_runs, run_once


def _override_cfg_baseline(cfg: SimulationConfig) -> SimulationConfig:
    """Create baseline_no_learning variant."""
    return replace(
        cfg,
        rl=replace(cfg.rl, enabled=False),
        price_model=replace(cfg.price_model, enabled=False),
        planning=replace(cfg.planning, enabled=False),
        rho_pos=0.0,
        rho_neg=0.0,
        memory=replace(cfg.memory, h1=0.0),
        confidence=replace(cfg.confidence, memory_enabled=False),
    )


def _override_cfg_memory_only(cfg: SimulationConfig) -> SimulationConfig:
    """Create memory_only variant."""
    return replace(
        cfg,
        rl=replace(cfg.rl, enabled=False),
        price_model=replace(cfg.price_model, enabled=False),
        planning=replace(cfg.planning, enabled=False),
        rho_pos=0.0,
        rho_neg=0.0,
        confidence=replace(cfg.confidence, memory_enabled=True),
    )


def _compute_persistence(agg_metrics: Dict[str, Any], H: int = 50) -> float:
    """Compute persistence as mean over horizons h=10..H of event_study_k_abs."""
    abs_curve = agg_metrics.get("event_study_k_abs", agg_metrics.get("event_study_k_decay", [0.0]))
    abs_arr = np.array(abs_curve, dtype=float)
    if len(abs_arr) < H + 1:
        return 0.0
    return float(np.mean(abs_arr[10 : H + 1]))


def _evaluate_candidate(
    cfg: SimulationConfig,
    seeds: int,
    base_h1: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Evaluate a candidate config by running baseline and memory_only variants.
    
    Returns:
        (baseline_metrics, memory_only_metrics, persistence_ratio)
    """
    # Run baseline_no_learning
    cfg_baseline = _override_cfg_baseline(cfg)
    runs_baseline = [run_once(cfg_baseline, seed=seed)[0] for seed in range(seeds)]
    agg_baseline = aggregate_runs(runs_baseline)
    
    # Run memory_only (low h1)
    cfg_memory_low = _override_cfg_memory_only(replace(cfg, memory=replace(cfg.memory, h1=0.0)))
    runs_memory_low = [run_once(cfg_memory_low, seed=1000 + seed)[0] for seed in range(seeds)]
    agg_memory_low = aggregate_runs(runs_memory_low)
    persistence_low = _compute_persistence(agg_memory_low)
    
    # Run memory_only (high h1 = 2x base)
    cfg_memory_high = _override_cfg_memory_only(replace(cfg, memory=replace(cfg.memory, h1=base_h1 * 2.0)))
    runs_memory_high = [run_once(cfg_memory_high, seed=2000 + seed)[0] for seed in range(seeds)]
    agg_memory_high = aggregate_runs(runs_memory_high)
    persistence_high = _compute_persistence(agg_memory_high)
    
    persistence_ratio = persistence_high / max(persistence_low, 1e-6)
    
    return agg_baseline, aggregate_runs(runs_memory_low), persistence_ratio


def _check_constraints(agg_baseline: Dict[str, Any]) -> bool:
    """Check hard constraints to avoid degenerate regimes."""
    ret_std = agg_baseline.get("ret_std", 0.0)
    demand_std = agg_baseline.get("demand_std", 0.0)
    abs_mispricing_mean = agg_baseline.get("abs_mispricing_mean", 0.0)
    
    if not (0.005 <= ret_std <= 0.2):
        return False
    if not (0.1 <= demand_std <= 50.0):
        return False
    if not (0.01 <= abs_mispricing_mean <= 5.0):
        return False
    return True


def _compute_score(
    agg_baseline: Dict[str, Any],
    agg_memory_only: Dict[str, Any],
    persistence_ratio: float,
) -> float:
    """Compute total score for a candidate.
    
    Score = 2.0*A + 0.5*B + 1.0*log(persistence_ratio)
    where:
      A = abs_ret_ac1_memory_only - abs_ret_ac1_baseline (want > 0)
      B = ret_std_memory_only - ret_std_baseline (optional, want modest > 0)
    """
    A = agg_memory_only.get("abs_ret_ac1", 0.0) - agg_baseline.get("abs_ret_ac1", 0.0)
    B = agg_memory_only.get("ret_std", 0.0) - agg_baseline.get("ret_std", 0.0)
    log_persistence = math.log(max(persistence_ratio, 1e-6))
    
    score = 2.0 * A + 0.5 * B + 1.0 * log_persistence
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search parameter tuning.")
    parser.add_argument("--config", default="code/configs/baseline.yaml")
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="code/configs/calibrated.yaml")
    args = parser.parse_args()
    
    # Load baseline config
    cfg = load_config(args.config)
    cfg = replace(cfg, horizon=args.T, num_agents=args.N)
    base_h1 = cfg.memory.h1
    
    # Define search grids
    kappa_grid = [0.005, 0.01, 0.02, 0.05]
    impact_grid = [0.01, 0.02, 0.05, 0.1]
    gamma_grid = [0.2, 0.5, 1.0]
    alpha_u_grid = [0.5, 1.0, 2.0]
    
    candidates: List[Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any], float]] = []
    
    total_combinations = len(kappa_grid) * len(impact_grid) * len(gamma_grid) * len(alpha_u_grid)
    print(f"Searching over {total_combinations} parameter combinations...")
    print(f"Grid: kappa={kappa_grid}, impact={impact_grid}, gamma={gamma_grid}, alpha_u={alpha_u_grid}\n")
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc="Testing candidates", unit="candidate", ncols=100)
    
    idx = 0
    for kappa in kappa_grid:
        for impact in impact_grid:
            for gamma in gamma_grid:
                for alpha_u in alpha_u_grid:
                    idx += 1
                    pbar.set_description(f"Testing kappa={kappa}, impact={impact}, gamma={gamma}, alpha_u={alpha_u}")
                    
                    # Create candidate config
                    candidate_cfg = replace(
                        cfg,
                        market=replace(cfg.market, kappa=kappa, impact=impact),
                        rl=replace(cfg.rl, gamma=gamma),
                        confidence=replace(cfg.confidence, alpha_u=alpha_u),
                    )
                    
                    try:
                        agg_baseline, agg_memory_only, persistence_ratio = _evaluate_candidate(
                            candidate_cfg, args.seeds, base_h1
                        )
                        
                        # Check constraints
                        if not _check_constraints(agg_baseline):
                            pbar.set_postfix({"status": "SKIP (constraints)", "valid": len(candidates)})
                            pbar.update(1)
                            continue
                        
                        # Compute score
                        score = _compute_score(agg_baseline, agg_memory_only, persistence_ratio)
                        
                        candidates.append((
                            {
                                "kappa": kappa,
                                "impact": impact,
                                "gamma": gamma,
                                "alpha_u": alpha_u,
                            },
                            score,
                            agg_baseline,
                            agg_memory_only,
                            persistence_ratio,
                        ))
                        
                        pbar.set_postfix({"status": "OK", "valid": len(candidates), "best_score": f"{score:.2f}"})
                    except Exception as e:
                        pbar.set_postfix({"status": f"ERROR: {str(e)[:20]}", "valid": len(candidates)})
                    finally:
                        pbar.update(1)
    
    pbar.close()
    print()
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10
    print("\n" + "=" * 80)
    print("Top 10 candidates:")
    print("=" * 80)
    for i, (params, score, agg_baseline, agg_memory_only, persistence_ratio) in enumerate(candidates[:10], 1):
        print(f"\n{i}. Score={score:.4f}")
        print(f"   Params: kappa={params['kappa']}, impact={params['impact']}, "
              f"gamma={params['gamma']}, alpha_u={params['alpha_u']}")
        print(f"   Baseline: ret_std={agg_baseline.get('ret_std', 0):.4f}, "
              f"abs_ret_ac1={agg_baseline.get('abs_ret_ac1', 0):.4f}, "
              f"abs_mispricing_mean={agg_baseline.get('abs_mispricing_mean', 0):.4f}, "
              f"demand_std={agg_baseline.get('demand_std', 0):.4f}")
        print(f"   Memory_only: ret_std={agg_memory_only.get('ret_std', 0):.4f}, "
              f"abs_ret_ac1={agg_memory_only.get('abs_ret_ac1', 0):.4f}, "
              f"abs_mispricing_mean={agg_memory_only.get('abs_mispricing_mean', 0):.4f}")
        print(f"   Persistence_ratio={persistence_ratio:.4f}")
    
        # Write best candidate
        if candidates:
            best_params, best_score, best_baseline, best_memory_only, best_persistence = candidates[0]
            
            # Create config with best parameters
            best_cfg = replace(
                cfg,
                market=replace(cfg.market, kappa=best_params["kappa"], impact=best_params["impact"]),
                rl=replace(cfg.rl, gamma=best_params["gamma"]),
                confidence=replace(cfg.confidence, alpha_u=best_params["alpha_u"]),
            )
            
            # Convert to dict and write YAML
            from dataclasses import asdict
            cfg_dict = asdict(best_cfg)
            
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg_dict, handle, default_flow_style=False)
            
            # Compute score components for summary
            A = best_memory_only.get("abs_ret_ac1", 0.0) - best_baseline.get("abs_ret_ac1", 0.0)
            B = best_memory_only.get("ret_std", 0.0) - best_baseline.get("ret_std", 0.0)
            
            # Write summary JSON
            summary_path = out_path.parent / "calibrated_summary.json"
            summary = {
                "chosen_params": {
                    "kappa": best_params["kappa"],
                    "impact": best_params["impact"],
                    "gamma": best_params["gamma"],
                    "alpha_u": best_params["alpha_u"],
                    "h0": best_cfg.memory.h0,
                    "h1": best_cfg.memory.h1,
                },
                "score": best_score,
                "score_components": {
                    "A": A,
                    "B": B,
                    "persistence_ratio": best_persistence,
                },
                "constraints": {
                    "ret_std_baseline": best_baseline.get("ret_std", 0.0),
                    "demand_std_baseline": best_baseline.get("demand_std", 0.0),
                    "abs_mispricing_mean_baseline": best_baseline.get("abs_mispricing_mean", 0.0),
                },
                "baseline_no_learning": {
                    "ret_std": best_baseline.get("ret_std", 0.0),
                    "abs_mispricing_mean": best_baseline.get("abs_mispricing_mean", 0.0),
                    "abs_ret_ac1": best_baseline.get("abs_ret_ac1", 0.0),
                    "k_mean": best_baseline.get("k_mean", 0.0),
                    "k_std": best_baseline.get("k_std", 0.0),
                    "demand_std": best_baseline.get("demand_std", 0.0),
                },
                "memory_only": {
                    "ret_std": best_memory_only.get("ret_std", 0.0),
                    "abs_mispricing_mean": best_memory_only.get("abs_mispricing_mean", 0.0),
                    "abs_ret_ac1": best_memory_only.get("abs_ret_ac1", 0.0),
                    "k_mean": best_memory_only.get("k_mean", 0.0),
                    "k_std": best_memory_only.get("k_std", 0.0),
                    "demand_std": best_memory_only.get("demand_std", 0.0),
                },
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            
            print(f"\n{'=' * 80}")
            print(f"Best candidate written to: {out_path}")
            print(f"Summary written to: {summary_path}")
            print(f"Score: {best_score:.4f}")
            print(f"Parameters: kappa={best_params['kappa']}, impact={best_params['impact']}, "
                  f"gamma={best_params['gamma']}, alpha_u={best_params['alpha_u']}")
        else:
            print("\nNo valid candidates found!")


if __name__ == "__main__":
    main()
