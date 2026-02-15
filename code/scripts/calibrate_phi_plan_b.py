"""
Calibrate phi via Plan B (variance-consistent, simulation-based) given an existing
moment-match targets JSON (e.g. paper_artifacts/moment_match/estimates_SPY.json).

Writes LaTeX fragments into the output directory:
- table_moment_match_phi.tex              (phi*, used by the paper)
- table_moment_match_phi_naive.tex        (phi from the targets JSON, for comparison)
- table_moment_match_phi_variance_shares.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../code

from mfg_oc.config import load_config
from mfg_oc.phi_plan_b import (
    calibrate_phi_bisection,
    calibrate_phi_fixed_point,
    render_variance_share_table_tex,
)


def _tex(v: float | None, fmt: str = ".4f") -> str:
    if v is None:
        return "---"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "---"
    return f"{v:{fmt}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan B calibration of phi from a targets JSON")
    parser.add_argument("--targets-json", type=str, required=True, help="Path to estimates_*.json containing sigma_r, scale_s, kappa.")
    parser.add_argument("--sim-config", type=str, default=None, help="Simulation config YAML (default: code/configs/paper_baseline.yaml).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for LaTeX fragments (default: targets-json directory).")
    parser.add_argument("--burn", type=int, default=2000, help="Burn-in increments.")
    parser.add_argument("--sample", type=int, default=20000, help="Sample increments.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds (common random numbers).")
    parser.add_argument("--impact", type=float, default=None, help="Override lambda/impact used in simulation (default: sim-config market.impact).")
    parser.add_argument("--method", type=str, default="fixed_point", choices=["fixed_point", "bisection"], help="Calibration method.")
    parser.add_argument("--fp-max-iter", type=int, default=6, help="Fixed-point iterations (when --method=fixed_point).")
    parser.add_argument("--fp-damping", type=float, default=0.5, help="Damping in (0,1] for fixed-point updates.")
    parser.add_argument("--max-iter", type=int, default=18, help="Max bisection iterations.")
    parser.add_argument("--tol-abs", type=float, default=1e-4, help="Absolute tolerance on daily sigma_r.")
    parser.add_argument("--tol-rel", type=float, default=5e-3, help="Relative tolerance on daily sigma_r.")
    args = parser.parse_args()

    targets_path = Path(args.targets_json)
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    sigma_r_target = float(targets.get("sigma_r"))
    scale_s = float(targets.get("scale_s"))
    kappa = float(targets.get("kappa"))
    phi_naive = float(targets.get("phi")) if targets.get("phi") is not None else float("nan")

    if not (np.isfinite(sigma_r_target) and np.isfinite(scale_s) and scale_s > 0.0):
        raise SystemExit("targets-json must contain finite sigma_r and scale_s>0.")

    default_sim_cfg = Path(__file__).resolve().parents[2] / "code" / "configs" / "paper_baseline.yaml"
    sim_cfg_path = Path(args.sim_config) if args.sim_config else default_sim_cfg
    cfg = load_config(sim_cfg_path)

    burn = int(args.burn)
    sample = int(args.sample)
    horizon_needed = burn + sample + 2
    impact = float(args.impact) if args.impact is not None else float(cfg.market.impact)
    cfg = replace(cfg, horizon=horizon_needed, market=replace(cfg.market, kappa=float(kappa), impact=float(impact)))

    seeds = []
    for part in str(args.seeds).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        seeds = [0, 1, 2]

    if args.method == "bisection":
        payload = calibrate_phi_bisection(
            base_cfg=cfg,
            sigma_r_target=sigma_r_target,
            scale_s=scale_s,
            seeds=seeds,
            burn=burn,
            sample=sample,
            phi_min=0.0,
            phi_max=phi_naive if np.isfinite(phi_naive) else None,
            tol_abs=float(args.tol_abs),
            tol_rel=float(args.tol_rel),
            max_iter=int(args.max_iter),
            log_metrics=False,
        )
    else:
        payload = calibrate_phi_fixed_point(
            base_cfg=cfg,
            sigma_r_target=sigma_r_target,
            scale_s=scale_s,
            seeds=seeds,
            burn=burn,
            sample=sample,
            phi_init=phi_naive if np.isfinite(phi_naive) else None,
            tol_abs=float(args.tol_abs),
            tol_rel=float(args.tol_rel),
            max_iter=int(args.fp_max_iter),
            damping=float(args.fp_damping),
            log_metrics=False,
        )
    phi_star = payload.get("phi_star")
    if phi_star is None:
        raise SystemExit(f"Plan B calibration failed: {payload.get('status')}")
    phi_star = float(phi_star)

    out_dir = Path(args.out_dir) if args.out_dir else targets_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "table_moment_match_phi.tex").write_text(_tex(phi_star, ".3f"), encoding="utf-8")
    (out_dir / "table_moment_match_phi_naive.tex").write_text(_tex(phi_naive, ".3f"), encoding="utf-8")
    sigma_eta_implied = phi_star / impact if impact > 0 else float("nan")
    (out_dir / "table_moment_match_sigma_eta_implied.tex").write_text(_tex(sigma_eta_implied, ".2f"), encoding="utf-8")
    (out_dir / "table_moment_match_phi_variance_shares.tex").write_text(
        render_variance_share_table_tex(payload.get("moments", {})),
        encoding="utf-8",
    )

    print(f"phi* = {phi_star:.6g} ({payload.get('status')}); wrote LaTeX fragments to {out_dir}")


if __name__ == "__main__":
    main()
