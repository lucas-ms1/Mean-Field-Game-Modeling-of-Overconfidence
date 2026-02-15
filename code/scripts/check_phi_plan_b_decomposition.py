"""
Guardrail: check that the anchored_impact decomposition is indexed consistently.

Verifies on a short stationary window that:
- C := dp - A - B has variance close to phi^2 dt,
- Cov(A,C) and Cov(B,C) are small relative to Var(dp),
- The full accounting identity Var(dp) == Var(A)+Var(B)+Var(C)+2Cov(.,.) holds up to FP error.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../code

from mfg_oc.config import load_config
from mfg_oc.phi_plan_b import moments


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Plan B variance decomposition consistency")
    parser.add_argument("--config", type=str, default=None, help="Simulation config YAML (default: code/configs/paper_baseline.yaml).")
    parser.add_argument("--phi", type=float, default=1.00, help="phi (noise_sigma) to test.")
    parser.add_argument("--kappa", type=float, default=None, help="Override kappa used for decomposition.")
    parser.add_argument("--impact", type=float, default=None, help="Override impact (lambda) used for decomposition.")
    parser.add_argument("--burn", type=int, default=500, help="Burn-in increments.")
    parser.add_argument("--sample", type=int, default=5000, help="Sample increments.")
    parser.add_argument("--seeds", type=str, default="0,1", help="Comma-separated seeds.")
    parser.add_argument("--scale-s", type=float, default=1.0, help="Scale s used for sigma_r_sim (not used in checks).")
    parser.add_argument(
        "--force-stationary-mu",
        action="store_true",
        help="Override fundamental_mu=0 to avoid drift (recommended for stationary variance checks).",
    )
    parser.add_argument("--tol-varC-rel", type=float, default=0.10, help="Tolerance for |Var(C)-phi^2 dt| / (phi^2 dt).")
    parser.add_argument("--tol-cov-share", type=float, default=0.01, help="Tolerance for |2Cov(A,C)|/Var(dp) and |2Cov(B,C)|/Var(dp).")
    args = parser.parse_args()

    default_cfg = Path(__file__).resolve().parents[2] / "code" / "configs" / "paper_baseline.yaml"
    cfg_path = Path(args.config) if args.config else default_cfg
    cfg = load_config(cfg_path)
    if args.force_stationary_mu:
        cfg = replace(cfg, fundamental_mu=0.0)

    burn = int(args.burn)
    sample = int(args.sample)
    horizon_needed = burn + sample + 2
    cfg = replace(cfg, horizon=horizon_needed)

    seeds = []
    for part in str(args.seeds).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise SystemExit("Need at least one seed.")

    m = moments(
        base_cfg=cfg,
        phi=float(args.phi),
        seeds=seeds,
        burn=burn,
        sample=sample,
        kappa=float(args.kappa) if args.kappa is not None else None,
        impact=float(args.impact) if args.impact is not None else None,
        scale_s=float(args.scale_s),
        log_metrics=False,
    )

    var_dp = float(m["var_dp"])
    var_c_real = float(m["var_C_resid"])
    var_c_model = float(m["var_C_model"])
    cov_ac = float(m["cov_AC"])
    cov_bc = float(m["cov_BC"])
    err_full = float(m["decomp_err_full"])

    critical = np.array([var_dp, var_c_real, var_c_model, cov_ac, cov_bc, err_full], dtype=float)
    if not np.all(np.isfinite(critical)):
        print("FAIL: non-finite moments encountered (divergent simulation or window contains inf/nan).")
        print(f"  Var(dp)={var_dp}")
        print(f"  Var(C)={var_c_real} vs phi^2 dt={var_c_model}")
        print(f"  Cov(A,C)={cov_ac}, Cov(B,C)={cov_bc}, decomp_err_full={err_full}")
        raise SystemExit(1)

    denom_dp = var_dp if abs(var_dp) > 1e-18 else 1.0
    denom_c = var_c_model if abs(var_c_model) > 1e-18 else 1.0

    rel_varc = abs(var_c_real - var_c_model) / denom_c
    share_ac = abs(2.0 * cov_ac) / denom_dp
    share_bc = abs(2.0 * cov_bc) / denom_dp
    rel_err_full = abs(err_full) / denom_dp

    ok = True
    if rel_varc > float(args.tol_varC_rel):
        ok = False
        print(f"FAIL: Var(C) mismatch rel={rel_varc:.4g} (Var(C)={var_c_real:.6g}, phi^2 dt={var_c_model:.6g})")
    if share_ac > float(args.tol_cov_share):
        ok = False
        print(f"FAIL: |2Cov(A,C)|/Var(dp)={share_ac:.4g} (2Cov={2*cov_ac:.6g}, Var(dp)={var_dp:.6g})")
    if share_bc > float(args.tol_cov_share):
        ok = False
        print(f"FAIL: |2Cov(B,C)|/Var(dp)={share_bc:.4g} (2Cov={2*cov_bc:.6g}, Var(dp)={var_dp:.6g})")
    if rel_err_full > 1e-9:
        ok = False
        print(f"FAIL: accounting identity gap rel={rel_err_full:.4g} (err={err_full:.6g}, Var(dp)={var_dp:.6g})")

    if ok:
        print("OK")
        print(f"  Var(dp)={var_dp:.6g}")
        print(f"  Var(C)={var_c_real:.6g} vs phi^2 dt={var_c_model:.6g} (rel={rel_varc:.4g})")
        print(f"  2Cov(A,C)/Var(dp)={2*cov_ac/denom_dp:.4g}")
        print(f"  2Cov(B,C)/Var(dp)={2*cov_bc/denom_dp:.4g}")
        return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
