# Current Code Map (MFG_Overconfidence)

This repo contains both the LaTeX paper sources and the Python simulation pipeline used to generate `paper_artifacts/`.

## Where the simulation lives

- Package: `code/mfg_oc/`
  - Main simulation loop: `code/mfg_oc/simulate.py`
  - Belief update (discrete-time KF): `code/mfg_oc/belief.py`
  - Price rule (anchored impact): `code/mfg_oc/market.py`
  - Experiment harness + metrics: `code/mfg_oc/experiments.py`
  - Config loader (YAML → dataclasses): `code/mfg_oc/config.py`
- Entrypoints:
  - Run a single simulation: `code/scripts/run_simulate.py`
  - Generate paper figures/tables: `code/scripts/build_paper_artifacts.py`
- Configs:
  - `code/configs/*.yaml` (baseline, calibrated, amplification sweeps, etc.)

## How to run (local)

- One run: `python -m scripts.run_simulate --config configs/baseline.yaml` (run from `code/`)
- Paper artifacts: `python -m scripts.build_paper_artifacts --config configs/amplification_baseline.yaml ...` (see script help for modes/flags)

## Step timing (what the code does per step)

In `code/mfg_oc/simulate.py`, for each time step `t`:
1) Update fundamental `v_t` (Euler; optional regime break and extensions).
2) Generate observations `y_{i,t}` and update beliefs `(v_hat_{i,t}, Sigma_{i,t})` via `code/mfg_oc/belief.py`.
3) Compute myopic demands and solve the per-step mean-field fixed point (linear `A_i + B_i * bar_x` structure).
4) Update price via `code/mfg_oc/market.py::anchored_impact` (Euler discretization with optional noise).
5) Log per-step metrics (used by `code/mfg_oc/experiments.py` to build summary tables).

## Phase 3 alignment hooks (theory ↔ code channels)

- Discrete-time effective variance in the myopic denominator (the “Σ-channel in actions”): `code/mfg_oc/simulate.py` computes
  `eff_var = sigma_p2 + kappa^2 * Sigma_for_risk * dt` and uses it in the demand coefficients.
- Diagnostic ratio (paper notation): `rho_{i,t} = (kappa^2 * Sigma_for_risk * dt) / sigma_p2`.
  - Logged per step as `xs_rho_mean` / `xs_rho_max` and summarized as `xs_rho_mean_mean` in the LaTeX metrics table.

### Ablation toggle for Fix (7)

To “turn off” the Σ-channel in actions *without changing belief dynamics*, set in the YAML config:

- `risk_variance_mode: fixed_steady_state` (or `fixed_initial` / `fixed_value`)
- `risk_variance_sigma_value: <float>` (only used when `risk_variance_mode: fixed_value`)

This freezes `Sigma_for_risk` only in the risk denominator; the Kalman recursion still updates `Sigma_{i,t}` normally.
