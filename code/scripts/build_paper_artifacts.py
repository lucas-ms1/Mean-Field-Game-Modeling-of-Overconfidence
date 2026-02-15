"""Build paper artifacts (CSV, PNG, LaTeX table) from experiment variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfg_oc.config import SimulationConfig, load_config
from mfg_oc.experiments import aggregate_runs, compute_metrics, run_once

matplotlib.use("Agg")

VARIANT_LABELS = {
    "bull_k1": "Bull $k=1$",
    "bull_k2": "Bull $k=3$",
    "bear_k1": "Bear $k=1$",
    "bear_k2": "Bear $k=3$",
    "bull_k1_myopic": "Bull $k=1$ (myopic)",
    "bull_k2_myopic": "Bull $k=3$ (myopic)",
    "bull_k1_intertemporal": "Bull $k=1$ (intertemporal)",
    "bull_k2_intertemporal": "Bull $k=3$ (intertemporal)",
}


def _extract_numeric_scalars(obj, prefix=""):
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf")):
            out[key] = float(v)
        elif isinstance(v, dict):
            # Flatten one (or more) levels of nested dicts using underscore separators
            out.update(_extract_numeric_scalars(v, prefix=key + "_"))
    return out


def _override_cfg(cfg: SimulationConfig, variant: str) -> SimulationConfig:
    if variant in {"bull_k1", "bull_k2", "bear_k1", "bear_k2", "bull_k1_myopic", "bull_k2_myopic", "bull_k1_intertemporal", "bull_k2_intertemporal"}:
        mu = 0.02 if variant.startswith("bull_") else -0.02
        if variant.endswith("_k1") or variant.endswith("_k1_myopic") or variant.endswith("_k1_intertemporal"):
            k0 = 1.0
        else:
            k0 = 3.0
        # Treat market.noise_sigma as the price-noise volatility φ used in the paper.
        phi = float(cfg.market.noise_sigma)
        use_intertemporal = variant.endswith("_intertemporal")
        return replace(
            cfg,
            fundamental_mu=mu,
            sigma_p2=phi**2,
            k_init=k0,
            k_min=k0,
            k_max=k0,
            use_intertemporal_policy=use_intertemporal,
            rl=replace(cfg.rl, enabled=False),
            price_model=replace(cfg.price_model, enabled=False),
            planning=replace(cfg.planning, enabled=False),
            rho_pos=0.0,
            rho_neg=0.0,
            memory=replace(cfg.memory, h1=0.0),
            confidence=replace(cfg.confidence, memory_enabled=False, k_bar=k0, psi=0.0, alpha_u=0.0, lambda_herd=0.0),
            experiment=None,
        )
    if variant == "baseline_no_learning":
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
    if variant == "memory_only":
        return replace(
            cfg,
            rl=replace(cfg.rl, enabled=False),
            price_model=replace(cfg.price_model, enabled=False),
            planning=replace(cfg.planning, enabled=False),
            rho_pos=0.0,
            rho_neg=0.0,
            confidence=replace(cfg.confidence, memory_enabled=True),
        )
    if variant == "rl_only":
        return replace(
            cfg,
            rl=replace(cfg.rl, enabled=True, reward_mode="expected_utility"),
            price_model=replace(cfg.price_model, enabled=False),
            planning=replace(cfg.planning, enabled=False),
        )
    if variant == "full_learning":
        return replace(
            cfg,
            rl=replace(cfg.rl, enabled=True, reward_mode="expected_utility"),
            price_model=replace(cfg.price_model, enabled=True),
            planning=replace(cfg.planning, enabled=True, K=1),
        )
    raise ValueError(f"Unknown variant: {variant}")


def _write_csv(path: Path, series: List[Dict[str, Any]]) -> None:
    if not series:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series[0].keys()))
        writer.writeheader()
        writer.writerows(series)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.std(x[start : i + 1])
    return out


def _aggregate_series(series_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not series_list:
        return []
    length = min(len(s) for s in series_list)
    keys = series_list[0][0].keys()
    agg = []
    for t in range(length):
        row = {}
        for key in keys:
            vals = [s[t][key] for s in series_list]
            if isinstance(vals[0], (int, float)):
                row[key] = float(np.mean(vals))
            else:
                row[key] = vals[0]
        agg.append(row)
    return agg


def _aggregate_series_with_ci(
    series_list: List[List[Dict[str, Any]]], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, mean, ci) arrays for plotting. CI = 1.96 * se."""
    if not series_list:
        return np.array([]), np.array([]), np.array([])
    length = min(len(s) for s in series_list)
    mean_arr = np.zeros(length)
    ci_arr = np.zeros(length)
    for t in range(length):
        vals = [float(s[t].get(key, 0.0)) for s in series_list]
        mean_arr[t] = float(np.mean(vals))
        if len(vals) >= 2:
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            ci_arr[t] = 1.96 * se
    times = np.arange(length, dtype=float)
    return times, mean_arr, ci_arr


def _plot_2x2_grid(
    fig_path: Path,
    series_by_variant: Dict[str, List[List[Dict[str, Any]]]],
    key: str,
    title: str,
    ylabel: str,
    transform: str = "abs",
) -> None:
    """2x2 grid: rows=policy (myopic/intertemporal), cols=k (1/3). Each panel: path with CI band."""
    comp_order = [
        ("bull_k1_myopic", "Myopic, $k=1$"),
        ("bull_k2_myopic", "Myopic, $k=3$"),
        ("bull_k1_intertemporal", "Intertemporal, $k=1$"),
        ("bull_k2_intertemporal", "Intertemporal, $k=3$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    for idx, (variant, panel_title) in enumerate(comp_order):
        ax = axes[idx // 2, idx % 2]
        series_list = series_by_variant.get(variant, [])
        if not series_list:
            ax.set_title(panel_title)
            continue
        length = min(len(s) for s in series_list)
        mean_arr = np.zeros(length)
        ci_arr = np.zeros(length)
        for t in range(length):
            raw_vals = [float(s[t].get(key, 0.0)) for s in series_list]
            vals = [abs(v) for v in raw_vals] if transform == "abs" else raw_vals
            mean_arr[t] = float(np.mean(vals))
            if len(vals) >= 2:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ci_arr[t] = 1.96 * se
        times = np.arange(length, dtype=float)
        ax.fill_between(times, mean_arr - ci_arr, mean_arr + ci_arr, alpha=0.3)
        ax.plot(times, mean_arr, linewidth=1.2)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_policy_decomposition(
    fig_path: Path,
    series_by_variant: Dict[str, List[List[Dict[str, Any]]]],
) -> None:
    """Plot mean myopic term vs mean hedging term over time for intertemporal runs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, variant in enumerate(["bull_k1_intertemporal", "bull_k2_intertemporal"]):
        ax = axes[ax_idx]
        series_list = series_by_variant.get(variant, [])
        if not series_list:
            ax.set_title(f"Intertemporal $k={1 if 'k1' in variant else 3}$")
            continue
        length = min(len(s) for s in series_list)
        myopic_mean = np.zeros(length)
        hedging_mean = np.zeros(length)
        ratio_mean = np.zeros(length)
        for t in range(length):
            myopic_vals = [float(s[t].get("mean_myopic_term", 0.0)) for s in series_list]
            hedging_vals = [float(s[t].get("mean_hedging_term", 0.0)) for s in series_list]
            myopic_mean[t] = np.mean(myopic_vals)
            hedging_mean[t] = np.mean(hedging_vals)
            denom = np.abs(myopic_mean[t])
            ratio_mean[t] = hedging_mean[t] / denom if denom > 1e-12 else 0.0
        times = np.arange(length, dtype=float)
        ax.plot(times, myopic_mean, label="Myopic term", linewidth=1.2)
        ax.plot(times, hedging_mean, label="Hedging term", linewidth=1.2)
        ax.set_title(f"Intertemporal $k={1 if 'k1' in variant else 3}$")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean contribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Policy decomposition: myopic vs hedging term (intertemporal only)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_overlay(fig_path: Path, series_by_variant: Dict[str, List[Dict[str, Any]]], key: str, title: str) -> None:
    plt.figure()
    for variant, series in series_by_variant.items():
        y = np.array([row[key] for row in series], dtype=float)
        plt.plot(y, label=VARIANT_LABELS.get(variant, variant))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_volatility(fig_path: Path, series_by_variant: Dict[str, List[Dict[str, Any]]], window: int = 20) -> None:
    plt.figure()
    for variant, series in series_by_variant.items():
        rets = np.array([row["ret"] for row in series], dtype=float)
        vol = _rolling_std(rets, window=window)
        plt.plot(vol, label=VARIANT_LABELS.get(variant, variant))
    plt.title("Rolling std of returns")
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_event_study(fig_path: Path, summaries: Dict[str, Dict[str, Any]]) -> None:
    plt.figure()
    for variant, summary in summaries.items():
        decay = np.array(summary["event_study_k_decay"], dtype=float)
        plt.plot(decay, label=variant)
    plt.title("Event-study k decay (top 1% shocks)")
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_sweep(
    fig_path: Path,
    xs: List[float],
    ys: List[float],
    cis: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure()
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    ci = np.array(cis, dtype=float)
    plt.errorbar(x, y, yerr=ci, fmt="-o", capsize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _latex_table(metrics_by_variant: Dict[str, Dict[str, Any]]) -> str:
    cols = [
        "ret_std",
        "abs_mispricing_mean",
        "abs_mispricing_p95",
        "k_mean",
        "k_std",
        "demand_std",
        "abs_ret_ac1",
    ]
    header = "Variant & " + " & ".join(cols) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    for variant, metrics in metrics_by_variant.items():
        vals = [f"{metrics[c]:.4f}" for c in cols]
        lines.append(variant + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_metrics_ci(
    samples_by_variant: Dict[str, List[Dict[str, Any]]],
    variants_in_order: List[str],
) -> str:
    cols = [
        ("abs_mispricing_mean", "$\\E[|p-v|]$"),
        ("xs_belief_std_mean", "$\\E[\\sigma_i(\\hat v_i)]$"),
        ("xs_action_std_mean", "$\\E[\\sigma_i(x_i)]$"),
        ("xs_sigma_mean_mean", "$\\E[\\bar\\Sigma]$"),
        ("xs_rho_mean_mean", "$\\E[\\rho_t]$"),
    ]
    header = "Scenario & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    def _row(label: str, xs_by_key: Dict[str, List[float]]) -> None:
        vals: List[str] = []
        for key, _ in cols:
            xs = xs_by_key.get(key, [])
            if not xs:
                vals.append("---")
                continue
            mean = float(np.mean(xs))
            if len(xs) >= 2:
                se = float(np.std(xs, ddof=1) / np.sqrt(len(xs)))
                ci = 1.96 * se
            else:
                ci = 0.0
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(label + " & " + " & ".join(vals) + " \\\\")

    for variant in variants_in_order:
        runs = samples_by_variant.get(variant, [])
        if variant in VARIANT_LABELS:
            row_label = VARIANT_LABELS[variant]
        else:
            row_label = variant.replace("_", "\\_")
        xs_by_key = {}
        for key, _ in cols:
            xs_by_key[key] = [float(r[key]) for r in runs if key in r and isinstance(r[key], (int, float))]
        _row(row_label, xs_by_key)

    # Paired treatment effects if baseline regimes are present
    if "bull_k1" in samples_by_variant and "bull_k2" in samples_by_variant:
        runs1 = samples_by_variant["bull_k1"]
        runs2 = samples_by_variant["bull_k2"]
        xs_by_key = {}
        for key, _ in cols:
            diffs = []
            for j in range(min(len(runs1), len(runs2))):
                if key in runs1[j] and key in runs2[j]:
                    diffs.append(float(runs2[j][key]) - float(runs1[j][key]))
            xs_by_key[key] = diffs
        _row("$\\Delta$ Bull ($k=3-1$)", xs_by_key)

    if "bear_k1" in samples_by_variant and "bear_k2" in samples_by_variant:
        runs1 = samples_by_variant["bear_k1"]
        runs2 = samples_by_variant["bear_k2"]
        xs_by_key = {}
        for key, _ in cols:
            diffs = []
            for j in range(min(len(runs1), len(runs2))):
                if key in runs1[j] and key in runs2[j]:
                    diffs.append(float(runs2[j][key]) - float(runs1[j][key]))
            xs_by_key[key] = diffs
        _row("$\\Delta$ Bear ($k=3-1$)", xs_by_key)
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _mean_ci(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) >= 2:
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ci = 1.96 * se
    else:
        ci = 0.0
    return mean, ci


def _economic_flag(
    baseline_abs_mis: float,
    delta_abs_mis: float,
    baseline_ret_std: float,
    delta_ret_std: float,
    *,
    rel_mis: float = 0.10,
    abs_mis: float = 0.10,
    rel_ret: float = 0.20,
) -> str:
    """Return a compact LaTeX-ready flag for economic significance.

    Thresholds are stated, justified, and sensitivity-checked in the paper
    (Section VI, ``Economic significance (Flag)''): rel_mis 5--10%, abs_mis 0--0.10,
    rel_ret 20%; joint grid uses rel_mis=0.05, abs_mis=0; extension uses 0.05, 0.05.
    """
    mis_thresh = max(abs_mis, rel_mis * max(baseline_abs_mis, 0.0))
    ret_thresh = rel_ret * max(baseline_ret_std, 0.0)
    meaningful = (abs(delta_abs_mis) >= mis_thresh) or (abs(delta_ret_std) >= ret_thresh)
    return "$\\checkmark$" if meaningful else ""


def _latex_table_convergence(samples_by_N: Dict[int, List[Dict[str, Any]]]) -> str:
    # Report paired differences relative to the largest N (common random numbers across N).
    cols = [
        ("abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("xs_action_std_mean", "$\\Delta\\E[\\sigma_i(x_i)]$"),
        ("xs_sigma_mean_mean", "$\\Delta\\E[\\bar\\Sigma]$"),
    ]
    header = "$N$ & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{r" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    Ns_sorted = sorted(samples_by_N.keys())
    N_ref = max(Ns_sorted) if Ns_sorted else 0
    ref_runs = samples_by_N.get(N_ref, [])
    for N in Ns_sorted:
        runs = samples_by_N[N]
        vals = []
        for key, _ in cols:
            if N == N_ref:
                mean, ci = 0.0, 0.0
            else:
                diffs: List[float] = []
                for i in range(min(len(runs), len(ref_runs))):
                    if key in runs[i] and key in ref_runs[i] and isinstance(runs[i][key], (int, float)) and isinstance(ref_runs[i][key], (int, float)):
                        diffs.append(float(runs[i][key]) - float(ref_runs[i][key]))
                mean, ci = _mean_ci(diffs)
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(f"{N} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_lambda_sensitivity(samples_by_lambda: Dict[float, List[Dict[str, Any]]]) -> str:
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("d_xs_action_std_mean", "$\\Delta\\E[\\sigma_i(x_i)]$"),
        ("d_xs_sigma_mean_mean", "$\\Delta\\E[\\bar\\Sigma]$"),
    ]
    header = "$\\lambda$ & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{r" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    for lam in sorted(samples_by_lambda.keys()):
        runs = samples_by_lambda[lam]
        vals = []
        for key, _ in cols:
            xs = [float(r[key]) for r in runs if key in r and isinstance(r[key], (int, float))]
            mean, ci = _mean_ci(xs)
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(f"{lam:.2f} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_lambda_kappa_sweep(samples_by_lambda: Dict[float, List[Dict[str, Any]]], kappa: float) -> str:
    """Paired k=3-1 effects across impact values, reporting mispricing + volatility."""
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_abs_mispricing_p95", "$\\Delta q_{0.95}(|p-v|)$"),
        ("d_abs_mispricing_p99", "$\\Delta q_{0.99}(|p-v|)$"),
        ("d_ret_std", "$\\Delta\\,\\mathrm{std}(r)$"),
        ("d_abs_ret_ac1", "$\\Delta\\,\\rho_1(|r|)$"),
    ]
    header = "$\\lambda$ (impact) & $\\lambda/\\kappa$ & " + " & ".join(label for _, label in cols) + " & Flag \\\\"
    lines = [
        "\\begin{tabular}{r" + "r" + "c" * len(cols) + "c}",
        "\\hline",
        header,
        "\\hline",
    ]
    for lam in sorted(samples_by_lambda.keys()):
        runs = samples_by_lambda[lam]
        vals = []
        for key, _ in cols:
            xs = [float(r[key]) for r in runs if key in r and isinstance(r[key], (int, float))]
            mean, ci = _mean_ci(xs)
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        # Economic flag based on mean deltas relative to mean baseline (k=1) for this λ.
        base_mis, _ = _mean_ci([float(r.get("k1_abs_mispricing_mean", 0.0)) for r in runs])
        base_ret, _ = _mean_ci([float(r.get("k1_ret_std", 0.0)) for r in runs])
        d_mis, _ = _mean_ci([float(r.get("d_abs_mispricing_mean", 0.0)) for r in runs])
        d_ret, _ = _mean_ci([float(r.get("d_ret_std", 0.0)) for r in runs])
        flag = _economic_flag(base_mis, d_mis, base_ret, d_ret)
        ratio = float(lam) / max(float(kappa), 1e-12)
        lines.append(f"{lam:.2f} & {ratio:.1f} & " + " & ".join(vals) + f" & {flag} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_price_filter_robustness(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("d_xs_action_std_mean", "$\\Delta\\E[\\sigma_i(x_i)]$"),
        ("d_xs_sigma_mean_mean", "$\\Delta\\E[\\bar\\Sigma]$"),
    ]
    header = "Filter & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        label = row["label"]
        vals = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_signal_noise_sweep(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_abs_mispricing_p95", "$\\Delta q_{0.95}(|p-v|)$"),
        ("d_abs_mispricing_p99", "$\\Delta q_{0.99}(|p-v|)$"),
        ("d_ret_std", "$\\Delta\\,\\mathrm{std}(r)$"),
        ("d_abs_ret_ac1", "$\\Delta\\,\\rho_1(|r|)$"),
        ("d_xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("d_xs_action_std_mean", "$\\Delta\\E[\\sigma_i(x_i)]$"),
    ]
    header = "$\\sigma_\\epsilon$ & $\\sigma_v/\\sigma_\\epsilon$ & " + " & ".join(label for _, label in cols) + " & Flag \\\\"
    lines = [
        "\\begin{tabular}{r" + "r" + "c" * len(cols) + "c}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        sigma_eps = float(row["sigma_eps"])
        snr = float(row["snr"])
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        base_mis, _ = _mean_ci(row.get("k1_abs_mispricing_mean", []))
        base_ret, _ = _mean_ci(row.get("k1_ret_std", []))
        d_mis, _ = _mean_ci(row.get("d_abs_mispricing_mean", []))
        d_ret, _ = _mean_ci(row.get("d_ret_std", []))
        flag = _economic_flag(base_mis, d_mis, base_ret, d_ret)
        lines.append(f"{sigma_eps:.2f} & {snr:.3f} & " + " & ".join(vals) + f" & {flag} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_mispricing_persistence(samples_by_variant: Dict[str, List[Dict[str, Any]]], variants_in_order: List[str]) -> str:
    cols = [
        ("abs_mispricing_mean", "$\\E[|p-v|]$"),
        ("abs_mispricing_p95", "$q_{0.95}(|p-v|)$"),
        ("abs_mispricing_p99", "$q_{0.99}(|p-v|)$"),
        ("abs_mispricing_ac1", "$\\rho_1(|p-v|)$"),
        ("mispricing_ac1", "$\\rho_1(p-v)$"),
    ]
    header = "Scenario & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    def _row(label: str, xs_by_key: Dict[str, List[float]]) -> None:
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(xs_by_key.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(label + " & " + " & ".join(vals) + " \\\\")

    for variant in variants_in_order:
        runs = samples_by_variant.get(variant, [])
        label = VARIANT_LABELS.get(variant, variant.replace("_", "\\_"))
        xs_by_key = {k: [float(r[k]) for r in runs if k in r and isinstance(r[k], (int, float))] for k, _ in cols}
        _row(label, xs_by_key)

    # Paired deltas (k=3-1) within bull/bear
    for base in [("bull_k1", "bull_k2", "$\\Delta$ Bull ($k=3-1$)"), ("bear_k1", "bear_k2", "$\\Delta$ Bear ($k=3-1$)")]:
        v1, v2, label = base
        if v1 in samples_by_variant and v2 in samples_by_variant:
            runs1 = samples_by_variant[v1]
            runs2 = samples_by_variant[v2]
            xs_by_key: Dict[str, List[float]] = {k: [] for k, _ in cols}
            for i in range(min(len(runs1), len(runs2))):
                for key, _ in cols:
                    if key in runs1[i] and key in runs2[i]:
                        xs_by_key[key].append(float(runs2[i][key]) - float(runs1[i][key]))
            _row(label, xs_by_key)

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_mispricing_stress(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("abs_mispricing_mean", "$\\E[|p-v|]$"),
        ("abs_mispricing_p95", "$q_{0.95}(|p-v|)$"),
        ("ret_std", "$\\mathrm{std}(r)$"),
    ]
    lines = [
        "\\begin{tabular}{lrrcccccc}",
        "\\hline",
        "Scenario & $\\kappa$ & $\\phi$ & \\multicolumn{2}{c}{" + cols[0][1] + "} & \\multicolumn{2}{c}{" + cols[1][1] + "} & \\multicolumn{2}{c}{" + cols[2][1] + "} \\\\",
        " &  &  & $k=1$ & $k=3$ & $k=1$ & $k=3$ & $k=1$ & $k=3$ \\\\",
        "\\hline",
    ]
    for row in rows:
        label = row["label"]
        kappa = float(row["kappa"])
        sigma_eta = float(row["sigma_eta"])
        vals: List[str] = []
        for key, _ in cols:
            for side in ("k1", "k3"):
                xs = [float(v) for v in row.get(f"{side}_{key}", []) if isinstance(v, (int, float, np.floating))]
                mean, ci = _mean_ci(xs)
                vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(f"{label} & {kappa:.3f} & {sigma_eta:.2f} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_kappa_sweep(rows: List[Dict[str, Any]]) -> str:
    """Paired k=3-1 effects across anchoring strengths."""
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_abs_mispricing_p95", "$\\Delta q_{0.95}(|p-v|)$"),
        ("d_abs_mispricing_p99", "$\\Delta q_{0.99}(|p-v|)$"),
        ("d_ret_std", "$\\Delta\\,\\mathrm{std}(r)$"),
        ("d_abs_ret_ac1", "$\\Delta\\,\\rho_1(|r|)$"),
    ]
    header = "$\\kappa$ & $\\lambda/\\kappa$ & " + " & ".join(label for _, label in cols) + " & Flag \\\\"
    lines = [
        "\\begin{tabular}{r" + "r" + "c" * len(cols) + "c}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        kappa = float(row["kappa"])
        ratio = float(row["ratio"])
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        base_mis, _ = _mean_ci(row.get("k1_abs_mispricing_mean", []))
        base_ret, _ = _mean_ci(row.get("k1_ret_std", []))
        d_mis, _ = _mean_ci(row.get("d_abs_mispricing_mean", []))
        d_ret, _ = _mean_ci(row.get("d_ret_std", []))
        flag = _economic_flag(base_mis, d_mis, base_ret, d_ret)
        lines.append(f"{kappa:.3f} & {ratio:.1f} & " + " & ".join(vals) + f" & {flag} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_joint_move_kappa_lambda_sigma_eta(rows: List[Dict[str, Any]]) -> str:
    """Joint grid in (kappa, lambda, sigma_eta): paired k=3-1 deltas with CI + flags."""
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("d_xs_action_mean_abs_mean", "$\\Delta\\E[|x_i|]$"),
        ("d_demand_std", "$\\Delta\\,\\mathrm{std}(\\sum_i x_i)$"),
        ("d_ret_std", "$\\Delta\\,\\mathrm{std}(r)$"),
    ]
    header = "$\\phi$ & $\\kappa$ & $\\lambda$ & " + " & ".join(label for _, label in cols) + " & Flag \\\\"
    lines = [
        "\\begin{tabular}{rrr" + "c" * len(cols) + "c}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in sorted(rows, key=lambda r: (float(r["sigma_eta"]), float(r["kappa"]), float(r["lam"]))):
        sigma_eta = float(row["sigma_eta"])
        kappa = float(row["kappa"])
        lam = float(row["lam"])
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        base_mis, _ = _mean_ci(row.get("k1_abs_mispricing_mean", []))
        base_ret, _ = _mean_ci(row.get("k1_ret_std", []))
        d_mis, _ = _mean_ci(row.get("d_abs_mispricing_mean", []))
        d_ret, _ = _mean_ci(row.get("d_ret_std", []))
        flag = _economic_flag(base_mis, d_mis, base_ret, d_ret, rel_mis=0.05, abs_mis=0.0)
        lines.append(f"{sigma_eta:.2f} & {kappa:.3f} & {lam:.2f} & " + " & ".join(vals) + f" & {flag} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _plot_joint_grid_heatmaps(
    path: Path,
    *,
    sigma_eta_values: List[float],
    kappa_values: List[float],
    lam_values: List[float],
    mean_by_sigma: Dict[float, np.ndarray],
    ci_by_sigma: Dict[float, np.ndarray],
    title: str,
    cbar_label: str,
) -> None:
    """Plot 3 heatmap slices (one per sigma_eta), outlining statistically significant cells."""
    ncols = len(sigma_eta_values)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    vmax = max(float(np.max(np.abs(mean_by_sigma[s]))) for s in sigma_eta_values) if sigma_eta_values else 1.0
    vmax = max(vmax, 1e-9)

    for ax, sigma_eta in zip(axes, sigma_eta_values):
        Z = mean_by_sigma[sigma_eta]
        CI = ci_by_sigma[sigma_eta]
        im = ax.imshow(Z, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_title(f"$\\phi={sigma_eta:.2f}$")
        ax.set_xticks(range(len(lam_values)))
        ax.set_xticklabels([f"{v:.2f}" for v in lam_values])
        ax.set_yticks(range(len(kappa_values)))
        ax.set_yticklabels([f"{v:.3f}" for v in kappa_values])
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$\\kappa$")

        # Outline statistically significant cells: CI excludes 0.
        for i in range(len(kappa_values)):
            for j in range(len(lam_values)):
                mean = float(Z[i, j])
                ci = float(CI[i, j])
                if abs(mean) > ci and ci > 0.0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, linewidth=2.0))

    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(cbar_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _latex_table_regime_strength_sweep(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_abs_mispricing_p95", "$\\Delta q_{0.95}(|p-v|)$"),
        ("d_abs_mispricing_p99", "$\\Delta q_{0.99}(|p-v|)$"),
        ("d_ret_std", "$\\Delta\\,\\mathrm{std}(r)$"),
        ("d_abs_ret_ac1", "$\\Delta\\,\\rho_1(|r|)$"),
    ]
    header = "Regime & $\\mu_v$ & " + " & ".join(label for _, label in cols) + " & Flag \\\\"
    lines = [
        "\\begin{tabular}{l" + "r" + "c" * len(cols) + "c}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        regime = str(row["regime"])
        mu = float(row["mu"])
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        base_mis, _ = _mean_ci(row.get("k1_abs_mispricing_mean", []))
        base_ret, _ = _mean_ci(row.get("k1_ret_std", []))
        d_mis, _ = _mean_ci(row.get("d_abs_mispricing_mean", []))
        d_ret, _ = _mean_ci(row.get("d_ret_std", []))
        flag = _economic_flag(base_mis, d_mis, base_ret, d_ret)
        lines.append(f"{regime} & {mu:.3f} & " + " & ".join(vals) + f" & {flag} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_heterogeneous_k(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("abs_mispricing_mean", "$\\E[|p-v|]$"),
        ("abs_mispricing_p95", "$q_{0.95}(|p-v|)$"),
        ("abs_mispricing_p99", "$q_{0.99}(|p-v|)$"),
        ("ret_std", "$\\mathrm{std}(r)$"),
        ("k_std", "$\\mathrm{std}_{\\mathrm{xs}}(k_i)$"),
    ]
    header = "Case & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        label = str(row["label"]).replace("_", "\\_")
        vals: List[str] = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_table_dt_sensitivity(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("d_abs_mispricing_mean", "$\\Delta\\E[|p-v|]$"),
        ("d_xs_belief_std_mean", "$\\Delta\\E[\\sigma_i(\\hat v_i)]$"),
        ("d_xs_action_std_mean", "$\\Delta\\E[\\sigma_i(x_i)]$"),
        ("d_abs_mispricing_p99", "$\\Delta q_{0.99}(|p-v|)$"),
    ]
    header = "$\\Delta t$ & " + " & ".join(label for _, label in cols) + " \\\\"
    lines = [
        "\\begin{tabular}{r" + "c" * len(cols) + "}",
        "\\hline",
        header,
        "\\hline",
    ]
    for row in rows:
        dt = row["dt"]
        vals = []
        for key, _ in cols:
            mean, ci = _mean_ci(row.get(key, []))
            vals.append(f"${mean:.3f}\\pm{ci:.3f}$")
        lines.append(f"{dt:.2f} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_placeholder_table(path: Path, title: str) -> None:
    text = "\n".join(
        [
            "\\begin{tabular}{l}",
            "\\hline",
            title + " \\\\",
            "\\hline",
            "Not generated (run build script without --no-sensitivity). \\\\",
            "\\hline",
            "\\end{tabular}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_default_config() -> str:
    """Return default config path: calibrated.yaml if exists, else baseline.yaml."""
    calibrated = Path("code/configs/calibrated.yaml")
    baseline = Path("code/configs/baseline.yaml")
    if calibrated.exists():
        return str(calibrated)
    return str(baseline)


def _compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of file contents."""
    with file_path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_git_status() -> str:
    """Get git status if available, else return placeholder."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "clean"
        return "git not available"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "git not available"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper artifacts from experiments.")
    parser.add_argument("--config", default=None, help="Config file path (default: calibrated.yaml if exists, else baseline.yaml)")
    parser.add_argument(
        "--mode",
        choices=("baseline_regimes", "learning_variants"),
        default="baseline_regimes",
        help="Which experiment set to run.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        default=None,
        help="Seed budget for validation tables (convergence/sensitivity/robustness). Defaults to min(--seeds, 10).",
    )
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--outdir", default="paper_artifacts", dest="out", help="Output directory")
    parser.add_argument("--write-generated", action="store_true")
    parser.add_argument("--no-sensitivity", dest="sensitivity", action="store_false", default=True)
    args = parser.parse_args()

    # Determine config path
    if args.config is None:
        config_path = Path(_get_default_config())
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config and compute MD5
    cfg = load_config(str(config_path))
    cfg = replace(cfg, horizon=args.T, num_agents=args.N)
    config_md5 = _compute_md5(config_path)
    
    # Print config info
    print(f"Using config: {config_path}")
    print(f"Config MD5: {config_md5}")
    print()

    if args.mode == "learning_variants":
        variants = ["baseline_no_learning", "memory_only", "rl_only", "full_learning"]
    else:
        variants = ["bull_k1", "bull_k2", "bear_k1", "bear_k2"]
    out_root = Path(args.out)
    provenance_root = out_root / "provenance"
    runs_root = out_root / "runs"
    fig_root = out_root / "figures"
    table_root = out_root / "tables"
    
    # Create provenance directory and write metadata
    provenance_root.mkdir(parents=True, exist_ok=True)
    
    # Copy config to provenance
    with config_path.open("r", encoding="utf-8") as src:
        with (provenance_root / "config_used.yaml").open("w", encoding="utf-8") as dst:
            dst.write(src.read())
    
    # Write MD5
    (provenance_root / "config_md5.txt").write_text(config_md5, encoding="utf-8")
    
    # Write git status
    git_status = _get_git_status()
    (provenance_root / "git_status.txt").write_text(git_status, encoding="utf-8")
    
    # Write run args
    run_args = {
        "config": str(config_path),
        "config_md5": config_md5,
        "seeds": args.seeds,
        "T": args.T,
        "N": args.N,
        "outdir": args.out,
        "sensitivity": args.sensitivity,
    }
    (provenance_root / "run_args.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in run_args.items()),
        encoding="utf-8",
    )

    summaries: Dict[str, Dict[str, Any]] = {}
    agg_series_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    samples_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    created_paths: List[str] = []

    for variant in variants:
        vcfg = _override_cfg(cfg, variant)
        series_list: List[List[Dict[str, Any]]] = []
        run_summaries: List[Dict[str, Any]] = []
        for seed in range(args.seeds):
            summary, series = run_once(vcfg, seed=seed)
            # Flatten nested dicts in summary to ensure regime-break metrics are aggregated
            flattened = {}
            # Preserve special array/list keys that aggregate_runs handles specially
            special_keys = ("event_study_k_decay", "event_study_k_mean", "event_study_k_abs", "event_study_k_norm", "event_study_k_by_correctness")
            for k, v in summary.items():
                if k == "seed":
                    continue  # Skip seed key
                if k in special_keys:
                    flattened[k] = v
                elif isinstance(v, dict):
                    # Extract numeric scalars from nested dicts
                    numeric_scalars = _extract_numeric_scalars(v, prefix=k + "_")
                    flattened.update(numeric_scalars)
                elif isinstance(v, (int, float)):
                    # Keep top-level numeric scalars as-is
                    if v == v and v not in (float("inf"), float("-inf")):  # Check for NaN and inf
                        flattened[k] = float(v)
                elif isinstance(v, (list, tuple)):
                    # Keep lists/arrays as-is
                    flattened[k] = v
            run_summaries.append(flattened)
            series_list.append(series)

            series_path = runs_root / variant / f"seed_{seed}" / "series.csv"
            summary_path = runs_root / variant / f"seed_{seed}" / "summary.json"
            _write_csv(series_path, series)
            _write_json(summary_path, summary)
            created_paths.extend([str(series_path), str(summary_path)])

        summaries[variant] = aggregate_runs(run_summaries)
        samples_by_variant[variant] = run_summaries
        
        # aggregate_runs only processes keys from runs[0], so collect all keys from all runs
        # and manually aggregate any missing ones
        if run_summaries:
            all_keys = set()
            for run_summary in run_summaries:
                all_keys.update(run_summary.keys())
            # For any keys not in the aggregated summary, aggregate them manually
            for key in all_keys:
                if key not in summaries[variant] and key != "seed":
                    # Check if it's a numeric scalar that should be aggregated
                    values = [r.get(key) for r in run_summaries if key in r]
                    if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                        valid_values = [v for v in values if v is not None]
                        if valid_values:
                            summaries[variant][key] = float(np.mean(valid_values))
        agg_series_by_variant[variant] = _aggregate_series(series_list)
        agg_path = out_root / f"summary_{variant}.json"
        _write_json(agg_path, summaries[variant])
        created_paths.append(str(agg_path))

        if args.write_generated:
            gen_dir = Path("code/configs/generated")
            gen_dir.mkdir(parents=True, exist_ok=True)
            gen_path = gen_dir / f"{variant}.yaml"
            with gen_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(asdict(vcfg), handle, default_flow_style=False)
            created_paths.append(str(gen_path))

    _plot_overlay(fig_root / "fig_price_path.png", agg_series_by_variant, "price", "Mean price path")
    plt.figure()
    for variant, series in agg_series_by_variant.items():
        y = np.array([abs(row["mispricing_true"]) for row in series], dtype=float)
        plt.plot(y, label=VARIANT_LABELS.get(variant, variant))
    plt.title("Mean |mispricing|")
    plt.legend()
    plt.tight_layout()
    fig_root.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_root / "fig_mispricing.png", dpi=150)
    plt.close()
    _plot_volatility(fig_root / "fig_volatility.png", agg_series_by_variant)
    created_paths.extend(
        [
            str(fig_root / "fig_price_path.png"),
            str(fig_root / "fig_mispricing.png"),
            str(fig_root / "fig_volatility.png"),
        ]
    )
    if args.mode == "learning_variants":
        _plot_overlay(fig_root / "fig_k_dynamics.png", agg_series_by_variant, "mean_k", "Mean k over time")
        _plot_event_study(fig_root / "fig_event_study_k.png", summaries)
        created_paths.extend([str(fig_root / "fig_k_dynamics.png"), str(fig_root / "fig_event_study_k.png")])

    table_tex = _latex_table(summaries)
    table_path = table_root / "table_ablations.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table_tex, encoding="utf-8")
    created_paths.append(str(table_path))

    metrics_table_tex = _latex_table_metrics_ci(samples_by_variant, variants)
    metrics_table_path = table_root / "table_metrics.tex"
    metrics_table_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table_path.write_text(metrics_table_tex, encoding="utf-8")
    created_paths.append(str(metrics_table_path))

    if args.mode == "baseline_regimes":
        mis_tex = _latex_table_mispricing_persistence(samples_by_variant, variants)
        mis_path = table_root / "table_mispricing_persistence.tex"
        mis_path.write_text(mis_tex, encoding="utf-8")
        created_paths.append(str(mis_path))

    # Baseline-mode validation: convergence in N and sensitivity in impact strength.
    if args.sensitivity and args.mode == "baseline_regimes":
        # 1) Convergence check in N (run a single scenario with a small seed budget).
        seeds_val = min(int(args.seeds), 10) if args.validation_seeds is None else int(args.validation_seeds)
        Ns = [200, 500, 2000]
        conv_samples: Dict[int, List[Dict[str, Any]]] = {}
        base_variant = "bull_k1"
        base_cfg = _override_cfg(cfg, base_variant)
        for N in Ns:
            cfgN = replace(base_cfg, num_agents=int(N))
            runsN = [run_once(cfgN, seed=10_000 + i)[0] for i in range(seeds_val)]
            conv_samples[int(N)] = runsN
        conv_tex = _latex_table_convergence(conv_samples)
        conv_path = table_root / "table_convergence.tex"
        conv_path.write_text(conv_tex, encoding="utf-8")
        created_paths.append(str(conv_path))

        # 2) Sensitivity in impact strength (paired k=3 - k=1 differences per seed).
        lam_values = [0.10, 0.20, 0.24]
        sens_samples: Dict[float, List[Dict[str, Any]]] = {}
        for lam in lam_values:
            base_k1 = _override_cfg(cfg, "bull_k1")
            base_k2 = _override_cfg(cfg, "bull_k2")
            cfg_k1 = replace(base_k1, market=replace(base_k1.market, impact=float(lam)))
            cfg_k2 = replace(base_k2, market=replace(base_k2.market, impact=float(lam)))
            diffs: List[Dict[str, Any]] = []
            for i in range(seeds_val):
                s1 = run_once(cfg_k1, seed=20_000 + i)[0]
                s2 = run_once(cfg_k2, seed=20_000 + i)[0]
                diffs.append(
                    {
                        "d_abs_mispricing_mean": float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)),
                        "d_xs_belief_std_mean": float(s2.get("xs_belief_std_mean", 0.0) - s1.get("xs_belief_std_mean", 0.0)),
                        "d_xs_action_std_mean": float(s2.get("xs_action_std_mean", 0.0) - s1.get("xs_action_std_mean", 0.0)),
                        "d_xs_sigma_mean_mean": float(s2.get("xs_sigma_mean_mean", 0.0) - s1.get("xs_sigma_mean_mean", 0.0)),
                    }
                )
            sens_samples[float(lam)] = diffs
        sens_tex = _latex_table_lambda_sensitivity(sens_samples)
        sens_path = table_root / "table_lambda_sensitivity.tex"
        sens_path.write_text(sens_tex, encoding="utf-8")
        created_paths.append(str(sens_path))

        # 2b) Wider sweep in impact strength: report mispricing + volatility deltas, and λ/κ ratio.
        # Extended range for amplification analysis: λ/κ ratios from 20 to 500
        kappa_ref = 0.005
        lam_sweep = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
        sweep_samples: Dict[float, List[Dict[str, Any]]] = {}
        for lam in lam_sweep:
            base_k1 = _override_cfg(cfg, "bull_k1")
            base_k2 = _override_cfg(cfg, "bull_k2")
            cfg_k1 = replace(base_k1, market=replace(base_k1.market, kappa=float(kappa_ref), impact=float(lam)))
            cfg_k2 = replace(base_k2, market=replace(base_k2.market, kappa=float(kappa_ref), impact=float(lam)))
            diffs: List[Dict[str, Any]] = []
            for i in range(seeds_val):
                s1 = run_once(cfg_k1, seed=22_000 + i)[0]
                s2 = run_once(cfg_k2, seed=22_000 + i)[0]
                diffs.append(
                    {
                        "k1_abs_mispricing_mean": float(s1.get("abs_mispricing_mean", 0.0)),
                        "k1_ret_std": float(s1.get("ret_std", 0.0)),
                        "d_abs_mispricing_mean": float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)),
                        "d_abs_mispricing_p95": float(s2.get("abs_mispricing_p95", 0.0) - s1.get("abs_mispricing_p95", 0.0)),
                        "d_abs_mispricing_p99": float(s2.get("abs_mispricing_p99", 0.0) - s1.get("abs_mispricing_p99", 0.0)),
                        "d_ret_std": float(s2.get("ret_std", 0.0) - s1.get("ret_std", 0.0)),
                        "d_abs_ret_ac1": float(s2.get("abs_ret_ac1", 0.0) - s1.get("abs_ret_ac1", 0.0)),
                    }
                )
            sweep_samples[float(lam)] = diffs
        sweep_tex = _latex_table_lambda_kappa_sweep(sweep_samples, kappa=float(kappa_ref))
        sweep_path = table_root / "table_lambda_kappa_sweep.tex"
        sweep_path.write_text(sweep_tex, encoding="utf-8")
        created_paths.append(str(sweep_path))

        # Figures: effect of λ/κ on mispricing + volatility (bull regime, k=3-1).
        xs = sorted(sweep_samples.keys())
        y_mis = []
        ci_mis = []
        y_vol = []
        ci_vol = []
        for lam in xs:
            runs = sweep_samples[lam]
            m, ci = _mean_ci([float(r.get("d_abs_mispricing_mean", 0.0)) for r in runs])
            v, vci = _mean_ci([float(r.get("d_ret_std", 0.0)) for r in runs])
            y_mis.append(m)
            ci_mis.append(ci)
            y_vol.append(v)
            ci_vol.append(vci)
        _plot_sweep(
            fig_root / "fig_lambda_kappa_sweep_mispricing.png",
            xs,
            y_mis,
            ci_mis,
            title="Impact-to-anchoring sweep: ΔE[|p-v|] (k=3-1, bull)",
            xlabel="$\\lambda$ (impact)",
            ylabel="ΔE[|p-v|]",
        )
        _plot_sweep(
            fig_root / "fig_lambda_kappa_sweep_volatility.png",
            xs,
            y_vol,
            ci_vol,
            title="Impact-to-anchoring sweep: $\\Delta\\,\\mathrm{std}(r)$ (k=3-1, bull)",
            xlabel="$\\lambda$ (impact)",
            ylabel="$\\Delta\\,\\mathrm{std}(r)$",
        )
        created_paths.extend(
            [
                str(fig_root / "fig_lambda_kappa_sweep_mispricing.png"),
                str(fig_root / "fig_lambda_kappa_sweep_volatility.png"),
            ]
        )

        # 3) Robustness: augment private filtering with a noisy price observation (heuristic).
        rows: List[Dict[str, Any]] = []
        for use_price, label in [(False, "Private-only"), (True, "Private + price")]:
            cfg_k1 = replace(_override_cfg(cfg, "bull_k1"), use_price_in_filter=use_price, price_obs_sigma=1.0)
            cfg_k2 = replace(_override_cfg(cfg, "bull_k2"), use_price_in_filter=use_price, price_obs_sigma=1.0)
            d_abs_mispricing_mean: List[float] = []
            d_xs_belief_std_mean: List[float] = []
            d_xs_action_std_mean: List[float] = []
            d_xs_sigma_mean_mean: List[float] = []
            for i in range(seeds_val):
                s1 = run_once(cfg_k1, seed=30_000 + i)[0]
                s2 = run_once(cfg_k2, seed=30_000 + i)[0]
                d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                d_xs_belief_std_mean.append(float(s2.get("xs_belief_std_mean", 0.0) - s1.get("xs_belief_std_mean", 0.0)))
                d_xs_action_std_mean.append(float(s2.get("xs_action_std_mean", 0.0) - s1.get("xs_action_std_mean", 0.0)))
                d_xs_sigma_mean_mean.append(float(s2.get("xs_sigma_mean_mean", 0.0) - s1.get("xs_sigma_mean_mean", 0.0)))
            rows.append(
                {
                    "label": label,
                    "d_abs_mispricing_mean": d_abs_mispricing_mean,
                    "d_xs_belief_std_mean": d_xs_belief_std_mean,
                    "d_xs_action_std_mean": d_xs_action_std_mean,
                    "d_xs_sigma_mean_mean": d_xs_sigma_mean_mean,
                }
            )
        price_tex = _latex_table_price_filter_robustness(rows)
        price_path = table_root / "table_price_filter.tex"
        price_path.write_text(price_tex, encoding="utf-8")
        created_paths.append(str(price_path))

        # 3b) Signal-to-noise sweep: vary idiosyncratic signal noise sigma_epsilon (observation_sigma),
        # holding sigma_v fixed (via fundamental_sigma from override).
        # Extended range: σv/σϵ from 0.05 to 0.25
        s2n_rows: List[Dict[str, Any]] = []
        sigma_v_ref = 0.10
        sigma_eps_values = [0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 2.00]
        for sigma_eps in sigma_eps_values:
            cfg_k1 = replace(_override_cfg(cfg, "bull_k1"), observation_sigma=float(sigma_eps))
            cfg_k2 = replace(_override_cfg(cfg, "bull_k2"), observation_sigma=float(sigma_eps))
            k1_abs_mispricing_mean: List[float] = []
            k1_ret_std: List[float] = []
            d_abs_mispricing_mean: List[float] = []
            d_abs_mispricing_p95: List[float] = []
            d_abs_mispricing_p99: List[float] = []
            d_ret_std: List[float] = []
            d_abs_ret_ac1: List[float] = []
            d_xs_belief_std_mean: List[float] = []
            d_xs_action_std_mean: List[float] = []
            for i in range(seeds_val):
                s1 = run_once(cfg_k1, seed=33_000 + i)[0]
                s2 = run_once(cfg_k2, seed=33_000 + i)[0]
                k1_abs_mispricing_mean.append(float(s1.get("abs_mispricing_mean", 0.0)))
                k1_ret_std.append(float(s1.get("ret_std", 0.0)))
                d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                d_abs_mispricing_p95.append(float(s2.get("abs_mispricing_p95", 0.0) - s1.get("abs_mispricing_p95", 0.0)))
                d_abs_mispricing_p99.append(float(s2.get("abs_mispricing_p99", 0.0) - s1.get("abs_mispricing_p99", 0.0)))
                d_ret_std.append(float(s2.get("ret_std", 0.0) - s1.get("ret_std", 0.0)))
                d_abs_ret_ac1.append(float(s2.get("abs_ret_ac1", 0.0) - s1.get("abs_ret_ac1", 0.0)))
                d_xs_belief_std_mean.append(float(s2.get("xs_belief_std_mean", 0.0) - s1.get("xs_belief_std_mean", 0.0)))
                d_xs_action_std_mean.append(float(s2.get("xs_action_std_mean", 0.0) - s1.get("xs_action_std_mean", 0.0)))
            snr = float(sigma_v_ref) / max(float(sigma_eps), 1e-12)
            s2n_rows.append(
                {
                    "sigma_eps": float(sigma_eps),
                    "snr": snr,
                    "k1_abs_mispricing_mean": k1_abs_mispricing_mean,
                    "k1_ret_std": k1_ret_std,
                    "d_abs_mispricing_mean": d_abs_mispricing_mean,
                    "d_abs_mispricing_p95": d_abs_mispricing_p95,
                    "d_abs_mispricing_p99": d_abs_mispricing_p99,
                    "d_ret_std": d_ret_std,
                    "d_abs_ret_ac1": d_abs_ret_ac1,
                    "d_xs_belief_std_mean": d_xs_belief_std_mean,
                    "d_xs_action_std_mean": d_xs_action_std_mean,
                }
            )
        s2n_tex = _latex_table_signal_noise_sweep(s2n_rows)
        s2n_path = table_root / "table_signal_to_noise_sweep.tex"
        s2n_path.write_text(s2n_tex, encoding="utf-8")
        created_paths.append(str(s2n_path))

        # Figures: effect of signal-to-noise on mispricing + volatility (bull regime, k=3-1).
        xs = [float(r["sigma_eps"]) for r in s2n_rows]
        y_mis = []
        ci_mis = []
        y_vol = []
        ci_vol = []
        for r in s2n_rows:
            m, ci = _mean_ci(r.get("d_abs_mispricing_mean", []))
            v, vci = _mean_ci(r.get("d_ret_std", []))
            y_mis.append(m)
            ci_mis.append(ci)
            y_vol.append(v)
            ci_vol.append(vci)
        _plot_sweep(
            fig_root / "fig_signal_to_noise_sweep_mispricing.png",
            xs,
            y_mis,
            ci_mis,
            title="Signal-noise sweep: ΔE[|p-v|] (k=3-1, bull)",
            xlabel="$\\sigma_\\epsilon$ (idio signal noise)",
            ylabel="ΔE[|p-v|]",
        )
        _plot_sweep(
            fig_root / "fig_signal_to_noise_sweep_volatility.png",
            xs,
            y_vol,
            ci_vol,
            title="Signal-noise sweep: $\\Delta\\,\\mathrm{std}(r)$ (k=3-1, bull)",
            xlabel="$\\sigma_\\epsilon$ (idio signal noise)",
            ylabel="$\\Delta\\,\\mathrm{std}(r)$",
        )
        created_paths.extend(
            [
                str(fig_root / "fig_signal_to_noise_sweep_mispricing.png"),
                str(fig_root / "fig_signal_to_noise_sweep_volatility.png"),
            ]
        )

        # 4) Time-step sensitivity (paired k=3-1 deltas; bull regime).
        seeds_dt = min(seeds_val, 20)
        total_time = int(args.T)
        dt_rows: List[Dict[str, Any]] = []
        for dt_val in [1.00, 0.50, 0.25]:
            steps = int(round(total_time / dt_val))
            cfg_k1 = replace(_override_cfg(cfg, "bull_k1"), dt=dt_val, horizon=steps)
            cfg_k2 = replace(_override_cfg(cfg, "bull_k2"), dt=dt_val, horizon=steps)
            d_abs_mispricing_mean: List[float] = []
            d_xs_belief_std_mean: List[float] = []
            d_xs_action_std_mean: List[float] = []
            d_abs_mispricing_p99: List[float] = []
            for i in range(seeds_dt):
                s1 = run_once(cfg_k1, seed=40_000 + i)[0]
                s2 = run_once(cfg_k2, seed=40_000 + i)[0]
                d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                d_xs_belief_std_mean.append(float(s2.get("xs_belief_std_mean", 0.0) - s1.get("xs_belief_std_mean", 0.0)))
                d_xs_action_std_mean.append(float(s2.get("xs_action_std_mean", 0.0) - s1.get("xs_action_std_mean", 0.0)))
                d_abs_mispricing_p99.append(float(s2.get("abs_mispricing_p99", 0.0) - s1.get("abs_mispricing_p99", 0.0)))
            dt_rows.append(
                {
                    "dt": dt_val,
                    "d_abs_mispricing_mean": d_abs_mispricing_mean,
                    "d_xs_belief_std_mean": d_xs_belief_std_mean,
                    "d_xs_action_std_mean": d_xs_action_std_mean,
                    "d_abs_mispricing_p99": d_abs_mispricing_p99,
                }
            )
        dt_tex = _latex_table_dt_sensitivity(dt_rows)
        dt_path = table_root / "table_dt_sensitivity.tex"
        dt_path.write_text(dt_tex, encoding="utf-8")
        created_paths.append(str(dt_path))

        # 5) Stress diagnostic: when can mispricing become large?
        # Focus on weak anchoring (low kappa) and high noise trading (high phi=noise_sigma).
        # Extended sweep: ση from 0.25 to 2.0 for amplification analysis
        seeds_stress = min(seeds_val, 20)
        # Expanded sweep grid (paper-facing): vary phi=noise_sigma more broadly; keep a weak-anchoring slice.
        stress_cases: List[Tuple[str, float, float]] = [
            ("Baseline", 0.005, 0.50),
            ("Weak anchoring", 0.001, 0.50),
            ("High noise", 0.005, 1.00),
            ("Weak+high", 0.001, 1.00),
            ("Low noise", 0.005, 0.25),
            ("Mid-high noise", 0.005, 0.75),
            ("Very high noise", 0.005, 1.50),
            ("Extreme noise", 0.005, 2.00),
            ("Very weak anchoring", 0.0005, 0.50),
        ]
        stress_rows: List[Dict[str, Any]] = []
        for label, kappa, sigma_eta in stress_cases:
            base_k1 = _override_cfg(cfg, "bull_k1")
            base_k3 = _override_cfg(cfg, "bull_k2")
            cfg_k1 = replace(
                base_k1,
                sigma_p2=float(sigma_eta) ** 2,
                market=replace(base_k1.market, kappa=float(kappa), noise_sigma=float(sigma_eta)),
            )
            cfg_k3 = replace(
                base_k3,
                sigma_p2=float(sigma_eta) ** 2,
                market=replace(base_k3.market, kappa=float(kappa), noise_sigma=float(sigma_eta)),
            )
            k1_vals: Dict[str, List[float]] = {k: [] for k, _ in [("abs_mispricing_mean", ""), ("abs_mispricing_p95", ""), ("ret_std", "")]}
            k3_vals: Dict[str, List[float]] = {k: [] for k, _ in [("abs_mispricing_mean", ""), ("abs_mispricing_p95", ""), ("ret_std", "")]}
            for i in range(seeds_stress):
                s1 = run_once(cfg_k1, seed=50_000 + i)[0]
                s3 = run_once(cfg_k3, seed=50_000 + i)[0]
                for key in k1_vals:
                    if key in s1:
                        k1_vals[key].append(float(s1[key]))
                    if key in s3:
                        k3_vals[key].append(float(s3[key]))
            stress_rows.append(
                {
                    "label": label,
                    "kappa": float(kappa),
                    "sigma_eta": float(sigma_eta),
                    "k1_abs_mispricing_mean": k1_vals["abs_mispricing_mean"],
                    "k3_abs_mispricing_mean": k3_vals["abs_mispricing_mean"],
                    "k1_abs_mispricing_p95": k1_vals["abs_mispricing_p95"],
                    "k3_abs_mispricing_p95": k3_vals["abs_mispricing_p95"],
                    "k1_ret_std": k1_vals["ret_std"],
                    "k3_ret_std": k3_vals["ret_std"],
                }
            )
        stress_tex = _latex_table_mispricing_stress(stress_rows)
        stress_path = table_root / "table_mispricing_stress.tex"
        stress_path.write_text(stress_tex, encoding="utf-8")
        created_paths.append(str(stress_path))

        # 6) Kappa sweep (anchoring strength): paired k=3-1 deltas at fixed impact + noise.
        # Extended range: κ from 0.001 to 0.02 for amplification analysis
        kappa_values = [0.001, 0.002, 0.003, 0.005, 0.010, 0.020]
        lam_ref = 0.20
        sigma_eta_ref = 0.50
        seeds_kappa = min(seeds_val, 20)
        kappa_rows: List[Dict[str, Any]] = []
        for kappa in kappa_values:
            base_k1 = _override_cfg(cfg, "bull_k1")
            base_k2 = _override_cfg(cfg, "bull_k2")
            cfg_k1 = replace(
                base_k1,
                sigma_p2=float(sigma_eta_ref) ** 2,
                market=replace(base_k1.market, kappa=float(kappa), impact=float(lam_ref), noise_sigma=float(sigma_eta_ref)),
            )
            cfg_k2 = replace(
                base_k2,
                sigma_p2=float(sigma_eta_ref) ** 2,
                market=replace(base_k2.market, kappa=float(kappa), impact=float(lam_ref), noise_sigma=float(sigma_eta_ref)),
            )
            d_abs_mispricing_mean: List[float] = []
            d_abs_mispricing_p95: List[float] = []
            d_abs_mispricing_p99: List[float] = []
            d_ret_std: List[float] = []
            d_abs_ret_ac1: List[float] = []
            k1_abs_mispricing_mean: List[float] = []
            k1_ret_std: List[float] = []
            for i in range(seeds_kappa):
                s1 = run_once(cfg_k1, seed=55_000 + i)[0]
                s2 = run_once(cfg_k2, seed=55_000 + i)[0]
                k1_abs_mispricing_mean.append(float(s1.get("abs_mispricing_mean", 0.0)))
                k1_ret_std.append(float(s1.get("ret_std", 0.0)))
                d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                d_abs_mispricing_p95.append(float(s2.get("abs_mispricing_p95", 0.0) - s1.get("abs_mispricing_p95", 0.0)))
                d_abs_mispricing_p99.append(float(s2.get("abs_mispricing_p99", 0.0) - s1.get("abs_mispricing_p99", 0.0)))
                d_ret_std.append(float(s2.get("ret_std", 0.0) - s1.get("ret_std", 0.0)))
                d_abs_ret_ac1.append(float(s2.get("abs_ret_ac1", 0.0) - s1.get("abs_ret_ac1", 0.0)))
            kappa_rows.append(
                {
                    "kappa": float(kappa),
                    "ratio": float(lam_ref) / max(float(kappa), 1e-12),
                    "k1_abs_mispricing_mean": k1_abs_mispricing_mean,
                    "k1_ret_std": k1_ret_std,
                    "d_abs_mispricing_mean": d_abs_mispricing_mean,
                    "d_abs_mispricing_p95": d_abs_mispricing_p95,
                    "d_abs_mispricing_p99": d_abs_mispricing_p99,
                    "d_ret_std": d_ret_std,
                    "d_abs_ret_ac1": d_abs_ret_ac1,
                }
            )
        kappa_tex = _latex_table_kappa_sweep(kappa_rows)
        kappa_path = table_root / "table_kappa_sweep.tex"
        kappa_path.write_text(kappa_tex, encoding="utf-8")
        created_paths.append(str(kappa_path))

        # 6b) Targeted “joint move” grid: one small grid over (κ, λ, σ_η).
        # Goal: identify where k-effects on price-level mispricing become non-negligible (if anywhere),
        # else document a negative result (k loads onto disagreement/intensity, not price-level mispricing).
        joint_kappa = [0.001, 0.005, 0.020]
        joint_lam = [0.10, 0.20, 0.50]
        joint_sigma_eta = [0.25, 0.50, 1.50]
        seeds_joint = min(seeds_val, 20)

        joint_rows: List[Dict[str, Any]] = []
        # For heatmap plotting (slice by σ_η): store mean/CI matrices for key deltas.
        mis_mean_by_sigma: Dict[float, np.ndarray] = {}
        mis_ci_by_sigma: Dict[float, np.ndarray] = {}
        dis_mean_by_sigma: Dict[float, np.ndarray] = {}
        dis_ci_by_sigma: Dict[float, np.ndarray] = {}

        any_non_negligible = False
        non_negligible_cells: List[Tuple[float, float, float]] = []

        for sigma_eta in joint_sigma_eta:
            mis_mean = np.zeros((len(joint_kappa), len(joint_lam)), dtype=float)
            mis_ci = np.zeros((len(joint_kappa), len(joint_lam)), dtype=float)
            dis_mean = np.zeros((len(joint_kappa), len(joint_lam)), dtype=float)
            dis_ci = np.zeros((len(joint_kappa), len(joint_lam)), dtype=float)

            for i_k, kappa in enumerate(joint_kappa):
                for j_l, lam in enumerate(joint_lam):
                    base_k1 = _override_cfg(cfg, "bull_k1")
                    base_k3 = _override_cfg(cfg, "bull_k2")
                    cfg_k1 = replace(
                        base_k1,
                        sigma_p2=float(sigma_eta) ** 2,
                        market=replace(base_k1.market, kappa=float(kappa), impact=float(lam), noise_sigma=float(sigma_eta)),
                    )
                    cfg_k3 = replace(
                        base_k3,
                        sigma_p2=float(sigma_eta) ** 2,
                        market=replace(base_k3.market, kappa=float(kappa), impact=float(lam), noise_sigma=float(sigma_eta)),
                    )

                    # Paired per-seed deltas.
                    k1_abs_mispricing_mean: List[float] = []
                    k1_ret_std: List[float] = []
                    d_abs_mispricing_mean: List[float] = []
                    d_xs_belief_std_mean: List[float] = []
                    d_xs_action_mean_abs_mean: List[float] = []
                    d_demand_std: List[float] = []
                    d_ret_std: List[float] = []

                    for s in range(seeds_joint):
                        s1 = run_once(cfg_k1, seed=90_000 + s)[0]
                        s3 = run_once(cfg_k3, seed=90_000 + s)[0]

                        k1_abs_mispricing_mean.append(float(s1.get("abs_mispricing_mean", 0.0)))
                        k1_ret_std.append(float(s1.get("ret_std", 0.0)))
                        d_abs_mispricing_mean.append(float(s3.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                        d_xs_belief_std_mean.append(float(s3.get("xs_belief_std_mean", 0.0) - s1.get("xs_belief_std_mean", 0.0)))
                        d_xs_action_mean_abs_mean.append(float(s3.get("xs_action_mean_abs_mean", 0.0) - s1.get("xs_action_mean_abs_mean", 0.0)))
                        d_demand_std.append(float(s3.get("demand_std", 0.0) - s1.get("demand_std", 0.0)))
                        d_ret_std.append(float(s3.get("ret_std", 0.0) - s1.get("ret_std", 0.0)))

                    # Row for LaTeX table.
                    joint_rows.append(
                        {
                            "sigma_eta": float(sigma_eta),
                            "kappa": float(kappa),
                            "lam": float(lam),
                            "k1_abs_mispricing_mean": k1_abs_mispricing_mean,
                            "k1_ret_std": k1_ret_std,
                            "d_abs_mispricing_mean": d_abs_mispricing_mean,
                            "d_xs_belief_std_mean": d_xs_belief_std_mean,
                            "d_xs_action_mean_abs_mean": d_xs_action_mean_abs_mean,
                            "d_demand_std": d_demand_std,
                            "d_ret_std": d_ret_std,
                        }
                    )

                    # Heatmap cell summaries.
                    dmis_mean, dmis_ci = _mean_ci(d_abs_mispricing_mean)
                    ddis_mean, ddis_ci = _mean_ci(d_xs_belief_std_mean)
                    mis_mean[i_k, j_l] = dmis_mean
                    mis_ci[i_k, j_l] = dmis_ci
                    dis_mean[i_k, j_l] = ddis_mean
                    dis_ci[i_k, j_l] = ddis_ci

                    # Non-negligible criterion (per plan): CI excludes 0 AND >= 5% of baseline.
                    base_mis, _ = _mean_ci(k1_abs_mispricing_mean)
                    stat_sig = (abs(dmis_mean) > dmis_ci) and (dmis_ci > 0.0)
                    econ_sig = abs(dmis_mean) >= 0.05 * max(base_mis, 0.0)
                    if stat_sig and econ_sig:
                        any_non_negligible = True
                        non_negligible_cells.append((float(sigma_eta), float(kappa), float(lam)))

            mis_mean_by_sigma[float(sigma_eta)] = mis_mean
            mis_ci_by_sigma[float(sigma_eta)] = mis_ci
            dis_mean_by_sigma[float(sigma_eta)] = dis_mean
            dis_ci_by_sigma[float(sigma_eta)] = dis_ci

        # Table: joint grid deltas (bull regime, paired k=3-1).
        joint_tex = _latex_table_joint_move_kappa_lambda_sigma_eta(joint_rows)
        joint_path = table_root / "table_joint_move_kappa_lambda_sigma_eta.tex"
        joint_path.write_text(joint_tex, encoding="utf-8")
        created_paths.append(str(joint_path))

        # Figures: 3-slice grids for mispricing and disagreement (bull regime, k=3-1).
        _plot_joint_grid_heatmaps(
            fig_root / "fig_joint_move_grid_mispricing.png",
            sigma_eta_values=[float(x) for x in joint_sigma_eta],
            kappa_values=[float(x) for x in joint_kappa],
            lam_values=[float(x) for x in joint_lam],
            mean_by_sigma=mis_mean_by_sigma,
            ci_by_sigma=mis_ci_by_sigma,
            title="Targeted joint grid: ΔE[|p−v|] (k=3−1, bull)",
            cbar_label="ΔE[|p−v|]",
        )
        _plot_joint_grid_heatmaps(
            fig_root / "fig_joint_move_grid_disagreement.png",
            sigma_eta_values=[float(x) for x in joint_sigma_eta],
            kappa_values=[float(x) for x in joint_kappa],
            lam_values=[float(x) for x in joint_lam],
            mean_by_sigma=dis_mean_by_sigma,
            ci_by_sigma=dis_ci_by_sigma,
            title="Targeted joint grid: ΔE[σ_i( v̂_i )] (k=3−1, bull)",
            cbar_label="ΔE[σ_i( v̂_i )]",
        )
        created_paths.extend(
            [
                str(fig_root / "fig_joint_move_grid_mispricing.png"),
                str(fig_root / "fig_joint_move_grid_disagreement.png"),
            ]
        )

        # Paper-ready note: negative result vs flagged region(s).
        note_lines: List[str] = []
        note_lines.append("Targeted joint-move grid over (κ, λ, σ_η) with paired seeds (bull regime).")
        note_lines.append("Non-negligible criterion: 95% CI excludes 0 AND |ΔE[|p−v|]| ≥ 5% of baseline E[|p−v|] in the same cell.")
        if any_non_negligible:
            note_lines.append("")
            note_lines.append("Result: non-negligible k-effects on price-level mispricing appear in the following cell(s) (σ_η, κ, λ):")
            for (se, kap, lam) in non_negligible_cells:
                note_lines.append(f"- phi={se:.2f}, kappa={kap:.3f}, lambda={lam:.2f}")
            note_lines.append("")
            note_lines.append("Interpretation: k materially affects level mispricing only in these parameter regions; elsewhere it primarily loads onto disagreement/intensity.")
        else:
            note_lines.append("")
            note_lines.append("Negative result: across the full 3×3×3 joint grid, k=3−1 differences in E[|p−v|] are statistically indistinguishable from zero and economically small (≤5% of baseline), while disagreement/intensity metrics move.")
            note_lines.append("In this architecture, k mainly loads onto disagreement/intensity, not price-level mispricing.")
        note_path = table_root / "note_joint_move_k_effects.txt"
        note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
        created_paths.append(str(note_path))

        # 7) Regime strength sweep: vary |mu_v| and compute paired k=3-1 deltas (bull and bear).
        mu_values = [0.01, 0.02, 0.03, 0.05]
        seeds_mu = min(seeds_val, 20)
        mu_rows: List[Dict[str, Any]] = []
        for sign, regime in [(+1.0, "Bull"), (-1.0, "Bear")]:
            for mu0 in mu_values:
                mu = float(sign * mu0)
                base_k1 = _override_cfg(cfg, "bull_k1" if sign > 0 else "bear_k1")
                base_k2 = _override_cfg(cfg, "bull_k2" if sign > 0 else "bear_k2")
                cfg_k1 = replace(base_k1, fundamental_mu=mu)
                cfg_k2 = replace(base_k2, fundamental_mu=mu)
                d_abs_mispricing_mean: List[float] = []
                d_abs_mispricing_p95: List[float] = []
                d_abs_mispricing_p99: List[float] = []
                d_ret_std: List[float] = []
                d_abs_ret_ac1: List[float] = []
                k1_abs_mispricing_mean: List[float] = []
                k1_ret_std: List[float] = []
                for i in range(seeds_mu):
                    s1 = run_once(cfg_k1, seed=66_000 + i)[0]
                    s2 = run_once(cfg_k2, seed=66_000 + i)[0]
                    k1_abs_mispricing_mean.append(float(s1.get("abs_mispricing_mean", 0.0)))
                    k1_ret_std.append(float(s1.get("ret_std", 0.0)))
                    d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0) - s1.get("abs_mispricing_mean", 0.0)))
                    d_abs_mispricing_p95.append(float(s2.get("abs_mispricing_p95", 0.0) - s1.get("abs_mispricing_p95", 0.0)))
                    d_abs_mispricing_p99.append(float(s2.get("abs_mispricing_p99", 0.0) - s1.get("abs_mispricing_p99", 0.0)))
                    d_ret_std.append(float(s2.get("ret_std", 0.0) - s1.get("ret_std", 0.0)))
                    d_abs_ret_ac1.append(float(s2.get("abs_ret_ac1", 0.0) - s1.get("abs_ret_ac1", 0.0)))
                mu_rows.append(
                    {
                        "regime": regime,
                        "mu": mu,
                        "k1_abs_mispricing_mean": k1_abs_mispricing_mean,
                        "k1_ret_std": k1_ret_std,
                        "d_abs_mispricing_mean": d_abs_mispricing_mean,
                        "d_abs_mispricing_p95": d_abs_mispricing_p95,
                        "d_abs_mispricing_p99": d_abs_mispricing_p99,
                        "d_ret_std": d_ret_std,
                        "d_abs_ret_ac1": d_abs_ret_ac1,
                    }
                )
        mu_tex = _latex_table_regime_strength_sweep(mu_rows)
        mu_path = table_root / "table_regime_strength_sweep.tex"
        mu_path.write_text(mu_tex, encoding="utf-8")
        created_paths.append(str(mu_path))

        # 8) Heterogeneous k: compare same-mean distributions in a bull regime.
        seeds_khet = min(seeds_val, 20)
        base = _override_cfg(cfg, "bull_k1")
        # Use a common mean k=2.0 for comparability across distributions.
        khet_cases = [
            ("Fixed k=2.0", {"k_dist": "fixed", "k_init": 2.0, "k_min": 2.0, "k_max": 2.0, "k_std": 0.0}),
            ("Uniform[1,3] (mean 2)", {"k_dist": "uniform", "k_init": 2.0, "k_min": 1.0, "k_max": 3.0, "k_std": 0.0}),
            ("Normal(2,0.5) clipped[1,3]", {"k_dist": "normal", "k_init": 2.0, "k_min": 1.0, "k_max": 3.0, "k_std": 0.5}),
        ]
        khet_rows: List[Dict[str, Any]] = []
        for label, kw in khet_cases:
            cfg_case = replace(
                base,
                k_dist=str(kw["k_dist"]),
                k_init=float(kw["k_init"]),
                k_min=float(kw["k_min"]),
                k_max=float(kw["k_max"]),
                k_std=float(kw["k_std"]),
            )
            abs_mispricing_mean: List[float] = []
            abs_mispricing_p95: List[float] = []
            abs_mispricing_p99: List[float] = []
            ret_std: List[float] = []
            k_std: List[float] = []
            for i in range(seeds_khet):
                s = run_once(cfg_case, seed=77_000 + i)[0]
                abs_mispricing_mean.append(float(s.get("abs_mispricing_mean", 0.0)))
                abs_mispricing_p95.append(float(s.get("abs_mispricing_p95", 0.0)))
                abs_mispricing_p99.append(float(s.get("abs_mispricing_p99", 0.0)))
                ret_std.append(float(s.get("ret_std", 0.0)))
                k_std.append(float(s.get("k_std", 0.0)))
            khet_rows.append(
                {
                    "label": label,
                    "abs_mispricing_mean": abs_mispricing_mean,
                    "abs_mispricing_p95": abs_mispricing_p95,
                    "abs_mispricing_p99": abs_mispricing_p99,
                    "ret_std": ret_std,
                    "k_std": k_std,
                }
            )
        khet_tex = _latex_table_heterogeneous_k(khet_rows)
        khet_path = table_root / "table_heterogeneous_k.tex"
        khet_path.write_text(khet_tex, encoding="utf-8")
        created_paths.append(str(khet_path))

        # 9) Myopic vs Intertemporal policy comparison: show that k comparative statics are unchanged
        # Comparability: same seeds (88_000 + i) for all variants so shocks are paired across policy/k.
        seeds_comp = min(seeds_val, 20)
        comp_variants = ["bull_k1_myopic", "bull_k2_myopic", "bull_k1_intertemporal", "bull_k2_intertemporal"]
        comp_samples: Dict[str, List[Dict[str, Any]]] = {}
        comp_series_by_variant: Dict[str, List[List[Dict[str, Any]]]] = {}
        for variant in comp_variants:
            vcfg = _override_cfg(cfg, variant)
            runs_comp = []
            series_list: List[List[Dict[str, Any]]] = []
            for i in range(seeds_comp):
                summary, series = run_once(vcfg, seed=88_000 + i)
                runs_comp.append(summary)
                series_list.append(series)
            comp_samples[variant] = runs_comp
            comp_series_by_variant[variant] = series_list
        
        # Figures: 2x2 grids for mispricing and belief dispersion; policy decomposition
        _plot_2x2_grid(
            fig_root / "fig_myopic_intertemporal_mispricing.png",
            comp_series_by_variant,
            key="mispricing_true",
            title="Mean |mispricing| by policy and $k$",
            ylabel="$\\mathbb{E}[|p-v|]$",
            transform="abs",
        )
        _plot_2x2_grid(
            fig_root / "fig_myopic_intertemporal_disagreement.png",
            comp_series_by_variant,
            key="xs_belief_std",
            title="Belief dispersion by policy and $k$",
            ylabel="$\\sigma_i(\\hat v_i)$",
        )
        _plot_policy_decomposition(
            fig_root / "fig_myopic_intertemporal_policy_decomposition.png",
            comp_series_by_variant,
        )
        created_paths.extend(
            [
                str(fig_root / "fig_myopic_intertemporal_mispricing.png"),
                str(fig_root / "fig_myopic_intertemporal_disagreement.png"),
                str(fig_root / "fig_myopic_intertemporal_policy_decomposition.png"),
            ]
        )
        
        # Generate comparison table: myopic vs intertemporal for k=1 and k=3
        comp_table_lines = [
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Policy & $k$ & $\\mathbb{E}[|p-v|]$ & $q_{0.95}(|p-v|)$ & $\\mathrm{std}(r)$ \\\\",
            "\\hline",
        ]
        for variant in comp_variants:
            runs = comp_samples.get(variant, [])
            policy_label = "Intertemporal" if variant.endswith("_intertemporal") else "Myopic"
            k_val = "1" if "k1" in variant else "3"
            abs_mis_mean, ci_mis = _mean_ci([float(r.get("abs_mispricing_mean", 0.0)) for r in runs])
            abs_mis_p95, ci_p95 = _mean_ci([float(r.get("abs_mispricing_p95", 0.0)) for r in runs])
            ret_std, ci_ret = _mean_ci([float(r.get("ret_std", 0.0)) for r in runs])
            comp_table_lines.append(
                f"{policy_label} & ${k_val}$ & ${abs_mis_mean:.3f}\\pm{ci_mis:.3f}$ & "
                f"${abs_mis_p95:.3f}\\pm{ci_p95:.3f}$ & ${ret_std:.3f}\\pm{ci_ret:.3f}$ \\\\"
            )
        
        # Add delta rows: k=3-1 for each policy
        for policy_type, suffix in [("Myopic", "_myopic"), ("Intertemporal", "_intertemporal")]:
            k1_runs = comp_samples.get(f"bull_k1{suffix}", [])
            k3_runs = comp_samples.get(f"bull_k2{suffix}", [])
            if k1_runs and k3_runs:
                d_mis_mean, d_ci_mis = _mean_ci([
                    float(k3_runs[i].get("abs_mispricing_mean", 0.0)) - float(k1_runs[i].get("abs_mispricing_mean", 0.0))
                    for i in range(min(len(k1_runs), len(k3_runs)))
                ])
                d_mis_p95, d_ci_p95 = _mean_ci([
                    float(k3_runs[i].get("abs_mispricing_p95", 0.0)) - float(k1_runs[i].get("abs_mispricing_p95", 0.0))
                    for i in range(min(len(k1_runs), len(k3_runs)))
                ])
                d_ret_std, d_ci_ret = _mean_ci([
                    float(k3_runs[i].get("ret_std", 0.0)) - float(k1_runs[i].get("ret_std", 0.0))
                    for i in range(min(len(k1_runs), len(k3_runs)))
                ])
                comp_table_lines.append(
                    f"$\\Delta$ {policy_type} ($k=3-1$) & --- & "
                    f"${d_mis_mean:.3f}\\pm{d_ci_mis:.3f}$ & "
                    f"${d_mis_p95:.3f}\\pm{d_ci_p95:.3f}$ & "
                    f"${d_ret_std:.3f}\\pm{d_ci_ret:.3f}$ \\\\"
                )
        
        comp_table_lines.extend(["\\hline", "\\end{tabular}"])
        comp_table_tex = "\n".join(comp_table_lines) + "\n"
        comp_table_path = table_root / "table_myopic_vs_intertemporal.tex"
        comp_table_path.write_text(comp_table_tex, encoding="utf-8")
        created_paths.append(str(comp_table_path))

        # 10) k→mispricing extension: impaired-arbitrage regime (state-dependent κ only)
        # α grid: κ(t) = κ₀ exp(−α|p−v|); α=0 is baseline.
        seeds_ext = min(seeds_val, 20)
        alpha_values = [0.0, 0.05, 0.1, 0.2]
        ext_rows: List[Dict[str, Any]] = []
        for alpha in alpha_values:
            ampl = {
                "state_dependent_kappa": alpha > 0,
                "kappa_decay_rate": float(alpha) if alpha > 0 else 0.1,
                "overconfidence_dependent_impact": False,
            }
            base_k1 = _override_cfg(cfg, "bull_k1")
            base_k2 = _override_cfg(cfg, "bull_k2")
            cfg_k1 = replace(base_k1, amplification=ampl)
            cfg_k2 = replace(base_k2, amplification=ampl)
            k1_abs_mispricing_mean: List[float] = []
            k1_abs_mispricing_median: List[float] = []
            k1_abs_mispricing_p95: List[float] = []
            d_abs_mispricing_mean: List[float] = []
            d_abs_mispricing_median: List[float] = []
            d_abs_mispricing_p95: List[float] = []
            for i in range(seeds_ext):
                s1 = run_once(cfg_k1, seed=95_000 + i)[0]
                s2 = run_once(cfg_k2, seed=95_000 + i)[0]
                k1_abs_mispricing_mean.append(float(s1.get("abs_mispricing_mean", 0.0)))
                k1_abs_mispricing_median.append(float(s1.get("abs_mispricing_median", s1.get("abs_mispricing_mean", 0.0))))
                k1_abs_mispricing_p95.append(float(s1.get("abs_mispricing_p95", 0.0)))
                d_abs_mispricing_mean.append(float(s2.get("abs_mispricing_mean", 0.0)) - float(s1.get("abs_mispricing_mean", 0.0)))
                d_abs_mispricing_median.append(
                    float(s2.get("abs_mispricing_median", s2.get("abs_mispricing_mean", 0.0)))
                    - float(s1.get("abs_mispricing_median", s1.get("abs_mispricing_mean", 0.0)))
                )
                d_abs_mispricing_p95.append(float(s2.get("abs_mispricing_p95", 0.0)) - float(s1.get("abs_mispricing_p95", 0.0)))
            ext_rows.append(
                {
                    "alpha": float(alpha),
                    "k1_abs_mispricing_mean": k1_abs_mispricing_mean,
                    "k1_abs_mispricing_p95": k1_abs_mispricing_p95,
                    "d_abs_mispricing_mean": d_abs_mispricing_mean,
                    "d_abs_mispricing_median": d_abs_mispricing_median,
                    "d_abs_mispricing_p95": d_abs_mispricing_p95,
                }
            )
        # LaTeX table for extension
        ext_lines = [
            "\\begin{tabular}{rcccccc}",
            "\\hline",
            "$\\alpha$ & $\\E[|p-v|]$ ($k=1$) & $\\Delta\\E[|p-v|]$ & $\\Delta q_{0.5}$ & $\\Delta q_{0.95}$ & Flag \\\\",
            "\\hline",
        ]
        for row in ext_rows:
            alpha = float(row["alpha"])
            k1_mean, _ = _mean_ci(row.get("k1_abs_mispricing_mean", []))
            d_mean, d_ci = _mean_ci(row.get("d_abs_mispricing_mean", []))
            d_med, _ = _mean_ci(row.get("d_abs_mispricing_median", []))
            d_p95, _ = _mean_ci(row.get("d_abs_mispricing_p95", []))
            flag = _economic_flag(k1_mean, d_mean, 0.0, 0.0, rel_mis=0.05, abs_mis=0.05)
            ext_lines.append(
                f"{alpha:.2f} & ${k1_mean:.3f}$ & ${d_mean:.3f}\\pm{d_ci:.3f}$ & "
                f"${d_med:.3f}$ & ${d_p95:.3f}$ & {flag} \\\\"
            )
        ext_lines.extend(["\\hline", "\\end{tabular}"])
        ext_path = table_root / "table_extension_state_dependent.tex"
        ext_path.write_text("\n".join(ext_lines) + "\n", encoding="utf-8")
        created_paths.append(str(ext_path))

        # Figure: Δ mispricing vs α
        xs_alpha = [float(r["alpha"]) for r in ext_rows]
        y_dmis, ci_dmis = [], []
        for r in ext_rows:
            m, c = _mean_ci(r.get("d_abs_mispricing_mean", []))
            y_dmis.append(m)
            ci_dmis.append(c)
        _plot_sweep(
            fig_root / "fig_extension_state_dependent.png",
            xs_alpha,
            y_dmis,
            ci_dmis,
            title="Impaired-arbitrage extension: $\\Delta\\mathbb{E}[|p-v|]$ ($k=3-1$) vs $\\alpha$",
            xlabel="$\\alpha$ (kappa decay rate)",
            ylabel="$\\Delta\\mathbb{E}[|p-v|]$",
        )
        created_paths.append(str(fig_root / "fig_extension_state_dependent.png"))
    elif args.mode == "baseline_regimes":
        # Ensure LaTeX includes don't fail if sensitivity is skipped.
        _write_placeholder_table(table_root / "table_convergence.tex", "Convergence check")
        _write_placeholder_table(table_root / "table_lambda_sensitivity.tex", "Sensitivity check")
        _write_placeholder_table(table_root / "table_signal_to_noise_sweep.tex", "Signal-to-noise sweep")
        _write_placeholder_table(table_root / "table_price_filter.tex", "Price-filter robustness")
        _write_placeholder_table(table_root / "table_dt_sensitivity.tex", "Time-step sensitivity")
        _write_placeholder_table(table_root / "table_mispricing_stress.tex", "Mispricing stress diagnostic")
        _write_placeholder_table(table_root / "table_kappa_sweep.tex", "Kappa sweep")
        _write_placeholder_table(table_root / "table_regime_strength_sweep.tex", "Regime strength sweep")
        _write_placeholder_table(table_root / "table_heterogeneous_k.tex", "Heterogeneous k")
        _write_placeholder_table(table_root / "table_myopic_vs_intertemporal.tex", "Myopic vs Intertemporal comparison")
        _write_placeholder_table(table_root / "table_extension_state_dependent.tex", "k→mispricing extension")

    print("Created paths:")
    for path in created_paths:
        print(f"- {path}")

    print("\nAggregated metrics:")
    for variant, metrics in summaries.items():
        excluded_keys = {"event_study_k_decay", "event_study_k_mean", "event_study_k_abs", "event_study_k_by_correctness"}
        line = f"{variant}: " + ", ".join(f"{k}={metrics[k]:.4f}" for k in metrics if k not in excluded_keys and isinstance(metrics[k], (int, float)))
        print(line)
    
    if args.mode == "learning_variants":
        # Compute separation summary
        print("\n" + "=" * 80)
        print("Separation Summary:")
        print("=" * 80)
        baseline = summaries.get("baseline_no_learning", {})
        memory_only = summaries.get("memory_only", {})
        rl_only = summaries.get("rl_only", {})
        full_learning = summaries.get("full_learning", {})

        metrics_to_compare = ["abs_ret_ac1", "k_std", "demand_std", "abs_mispricing_mean"]

        print("\nMemory effect (memory_only - baseline_no_learning):")
        for metric in metrics_to_compare:
            baseline_val = baseline.get(metric, 0.0)
            memory_val = memory_only.get(metric, 0.0)
            diff = memory_val - baseline_val
            print(f"  {metric}: {baseline_val:.4f} -> {memory_val:.4f} (diff={diff:+.4f})")

        print("\nFull learning effect (full_learning - rl_only):")
        for metric in metrics_to_compare:
            rl_val = rl_only.get(metric, 0.0)
            full_val = full_learning.get(metric, 0.0)
            diff = full_val - rl_val
            print(f"  {metric}: {rl_val:.4f} -> {full_val:.4f} (diff={diff:+.4f})")

        # Regime-break adaptation metrics (only relevant for learning_variants configs)
        if cfg.experiment and cfg.experiment.regime_break_enabled:
            print("\n" + "=" * 80)
            print("Regime-Break Adaptation Metrics:")
            print("=" * 80)
        
        # Plot mispricing over time
        if "rl_only" in agg_series_by_variant and "full_learning" in agg_series_by_variant:
            t_break = cfg.experiment.t_break
            plt.figure()
            for variant in ["rl_only", "full_learning"]:
                series = agg_series_by_variant[variant]
                times = [row.get("t", 0) for row in series]
                mispricing = [abs(row.get("mispricing_true", 0.0)) for row in series]
                plt.plot(times, mispricing, label=variant)
            plt.axvline(x=t_break, color='r', linestyle='--', label='Regime Break')
            plt.xlabel("Time")
            plt.ylabel("|Mispricing|")
            plt.title("Mispricing Response to Regime Break")
            plt.legend()
            plt.tight_layout()
            fig_path = fig_root / "fig_regime_break_mispricing.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=150)
            plt.close()
            created_paths.append(str(fig_path))
        
        # Create LaTeX table
        break_metrics = ["mispricing_overshoot", "recovery_time_mispricing", "vol_jump", "cluster_jump"]
        rl_break = {k: rl_only.get(k, None) for k in break_metrics}
        full_break = {k: full_learning.get(k, None) for k in break_metrics}
        
        if any(v is not None for v in rl_break.values()) or any(v is not None for v in full_break.values()):
            lines = [
                "\\begin{tabular}{lcccc}",
                "\\hline",
                "Variant & Overshoot & Recovery Time & Vol Jump & Cluster Jump \\\\",
                "\\hline",
            ]
            for variant, metrics in [("rl\\_only", rl_break), ("full\\_learning", full_break)]:
                vals = []
                for k in break_metrics:
                    v = metrics.get(k)
                    if v is None:
                        vals.append("---")
                    else:
                        if k == "recovery_time_mispricing":
                            vals.append(f"{v:.1f}")
                        else:
                            vals.append(f"{v:.4f}")
                lines.append(f"{variant} & {' & '.join(vals)} \\\\")
            lines.extend(["\\hline", "\\end{tabular}"])
            
            table_tex = "\n".join(lines) + "\n"
            table_path = table_root / "table_regime_break.tex"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            table_path.write_text(table_tex, encoding="utf-8")
            created_paths.append(str(table_path))
            
        # Print regime-break metrics robustly
        base = ["mispricing_overshoot", "recovery_time_mispricing", "vol_jump", "cluster_jump"]

        def _get_metric(agg, name):
            # Try common encodings
            candidates = [
                name,
                f"regime_break_{name}",
                f"regimebreak_{name}",
                f"rb_{name}",
            ]
            for c in candidates:
                if c in agg:
                    return c, agg[c]
            # Fallback: any key that endswith _<name>
            for k in agg.keys():
                if k.endswith("_" + name):
                    return k, agg[k]
            return None, None

        # Print per-variant metrics if available
        # Read from JSON files to ensure we have all keys (including regime-break metrics)
        aggregated = {}
        for variant_name in summaries.keys():
            json_path = out_root / f"summary_{variant_name}.json"
            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as f:
                    aggregated[variant_name] = json.load(f)
            else:
                aggregated[variant_name] = summaries[variant_name]
        for variant_name, agg in aggregated.items():
            found_any = False
            for nm in base:
                k, v = _get_metric(agg, nm)
                if k is not None:
                    if not found_any:
                        print(f"{variant_name}:")
                        found_any = True
                    if nm == "recovery_time_mispricing":
                        print(f"  {nm}: {v:.1f} (from key '{k}')")
                    else:
                        print(f"  {nm}: {v:.4f} (from key '{k}')")
            if not found_any:
                # Helpful debug line
                dbg = [k for k in agg.keys() if ("break" in k.lower() or "overshoot" in k.lower() or "recovery" in k.lower() or "vol_jump" in k.lower() or "cluster_jump" in k.lower())]
                print(f"{variant_name}: (no regime-break metrics found; candidate keys={dbg[:20]})")

        # Print separation (full_learning - rl_only) if both exist
        if "full_learning" in aggregated and "rl_only" in aggregated:
            print("\nRegime-break separation (full_learning - rl_only):")
            a = aggregated["rl_only"]
            b = aggregated["full_learning"]
            for nm in base:
                ka, va = _get_metric(a, nm)
                kb, vb = _get_metric(b, nm)
                if va is not None and vb is not None:
                    print(f"  {nm}: {va:.4f} -> {vb:.4f} (diff={vb - va:+.4f})")

    if args.sensitivity and args.mode == "learning_variants":
        base_memory_only = _override_cfg(cfg, "memory_only")
        memory_cfg = base_memory_only.memory
        cfg_low = replace(base_memory_only, memory=replace(memory_cfg, h1=0.0))
        cfg_high = replace(base_memory_only, memory=replace(memory_cfg, h1=memory_cfg.h1 * 2.0))
        runs_low = [run_once(cfg_low, seed=900 + i)[0] for i in range(3)]
        runs_high = [run_once(cfg_high, seed=950 + i)[0] for i in range(3)]
        agg_low = aggregate_runs(runs_low)
        agg_high = aggregate_runs(runs_high)
        norm_low = np.array(agg_low.get("event_study_k_norm", [0.0]), dtype=float)
        norm_high = np.array(agg_high.get("event_study_k_norm", [0.0]), dtype=float)
        halflife_low = agg_low.get("event_study_k_halflife", 50.0)
        halflife_high = agg_high.get("event_study_k_halflife", 50.0)
        H = len(norm_low) - 1
        # Compute persistence as mean over horizons h=10..H of normalized curve
        persistence_low = float(np.mean(norm_low[10 : H + 1])) if H >= 10 else 0.0
        persistence_high = float(np.mean(norm_high[10 : H + 1])) if H >= 10 else 0.0
        # PASS if persistence_high > persistence_low by at least max(0.05 * persistence_low, 1e-3)
        # AND halflife_high >= halflife_low
        rel_threshold = persistence_low * 0.05
        abs_threshold = 1e-3
        threshold = max(rel_threshold, abs_threshold)
        norm_passes = (persistence_high - persistence_low) >= threshold
        halflife_passes = halflife_high >= halflife_low
        status = "PASS" if (norm_passes and halflife_passes) else "FAIL"
        print(
            f"\nSanity check {status}: higher h1 increases normalized persistence "
            f"(norm_low={persistence_low:.4f}, norm_high={persistence_high:.4f}, "
            f"hl_low={halflife_low:.2f}, hl_high={halflife_high:.2f})."
        )
    else:
        print("\nSanity check: sensitivity toggle not run.")


if __name__ == "__main__":
    main()
