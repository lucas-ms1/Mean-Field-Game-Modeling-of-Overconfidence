"""
One-asset moment match: estimate (κ, ση, λ) from a liquid index/ETF.

Operationalizes Section 6.1 (Parameter identification and calibration strategy).
Outputs: estimates JSON, regime classification, optional regime figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys

# Windows consoles often default to a non-UTF-8 code page (e.g., cp1252),
# which can crash argparse help output if it contains Greek symbols (κ, λ, σ).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf


LN2 = math.log(2)


def _load_paper_baseline_params(config_path: Path) -> Dict[str, float]:
    """
    Load paper baseline parameters from a SimulationConfig-style YAML.

    Expected keys: market.kappa, market.impact, market.noise_sigma.
    Interprets market.noise_sigma as the price-noise volatility φ in dp.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid YAML config (expected mapping at root).")
    market = data.get("market", {})
    if not isinstance(market, dict):
        raise ValueError("Invalid YAML config (expected 'market' mapping).")
    return {
        "kappa": float(market["kappa"]),
        "lambda": float(market["impact"]),
        "phi": float(market["noise_sigma"]),
    }


def _synthetic_spy_like(n_days: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic SPY-like data for reproducible vignette when yfinance unavailable."""
    rng = np.random.default_rng(seed)
    kappa_true = 0.004
    sigma_eta_true = 0.012
    v = np.zeros(n_days + 1)
    p = np.zeros(n_days + 1)
    for t in range(n_days):
        v[t + 1] = v[t] + 0.0001 * rng.standard_normal()
        p[t + 1] = p[t] + kappa_true * (v[t] - p[t]) + sigma_eta_true * rng.standard_normal()
    price = 400 * np.exp(p)
    volume = 50e6 * np.exp(0.5 * rng.standard_normal(n_days + 1))
    volume = np.maximum(volume, 1e6)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n_days + 1, freq="B")[: n_days + 1]
    return pd.DataFrame({"price": price, "volume": volume}, index=idx)


def fetch_data(ticker: str = "SPY", years: int = 8) -> pd.DataFrame:
    """Fetch daily price and volume for the ticker."""
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=years * 365)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, timeout=30)
    except Exception:
        return pd.DataFrame(columns=["price", "volume"])
    if df.empty or len(df) < 100:
        return pd.DataFrame(columns=["price", "volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if "Close" not in df.columns or "Volume" not in df.columns:
        return pd.DataFrame(columns=["price", "volume"])
    return df[["Close", "Volume"]].rename(columns={"Close": "price", "Volume": "volume"})


def estimate_kappa(df: pd.DataFrame, ma_window: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Estimate κ from mean-reversion half-life of price deviations from slow MA.

    y_t = log(P_t) - log(MA_t) ≈ (p - v) proxy
    AR(1): y_t = ρ y_{t-1} + ε_t
    Half-life h = -ln(0.5)/ln(ρ), κ = ln(2)/h
    """
    if ma_window is None:
        ma_window = min(200, max(5, len(df) // 4))
    price = df["price"].astype(float).values
    ma = pd.Series(price).rolling(ma_window, min_periods=ma_window).mean().values
    price_slice = price[ma_window:]
    ma_slice = ma[ma_window:]
    valid = (~np.isnan(ma_slice)) & (ma_slice > 1e-12)
    y = (price_slice[valid] - ma_slice[valid]) / ma_slice[valid]
    if len(y) < 5:
        return float("nan"), float("nan"), float("nan")
    # AR(1): y_t = ρ y_{t-1}
    y_lag = y[:-1]
    y_curr = y[1:]
    rho = np.dot(y_lag, y_curr) / (np.dot(y_lag, y_lag) + 1e-12)
    rho = np.clip(rho, 0.01, 0.999)
    half_life = -LN2 / math.log(rho)
    kappa = LN2 / half_life
    return kappa, half_life, float(rho)


def estimate_sigma_r(df: pd.DataFrame) -> Tuple[float, float]:
    """Estimate return volatility sigma_r from std(log returns) per day; also annualized."""
    if df.empty or len(df) < 2:
        return float("nan"), float("nan")
    returns = np.log(df["price"] / df["price"].shift(1)).dropna()
    if len(returns) < 2:
        return float("nan"), float("nan")
    sigma_r = float(returns.std())
    sigma_r_ann = sigma_r * (252 ** 0.5)  # annualized
    return sigma_r, sigma_r_ann


def estimate_lambda(df: pd.DataFrame, baseline_lambda: float = 0.20) -> float:
    """
    Estimate λ from Cov(r, V) / Var(V) where V is demeaned volume proxy.

    Without signed order flow, use turnover (volume / price) or demeaned log-volume
    as demand proxy. Scale to be comparable to baseline λ.
    """
    returns = np.log(df["price"] / df["price"].shift(1)).dropna()
    # Use log-volume as proxy (demeaned)
    vol = np.log(df["volume"].values + 1)
    vol = vol[1:]  # align with returns
    returns = returns.values
    if len(returns) != len(vol):
        n = min(len(returns), len(vol))
        returns = returns[:n]
        vol = vol[:n]
    if len(returns) < 10:
        return float("nan")
    vol = vol - np.mean(vol)
    cov_rv = np.cov(returns, vol)[0, 1]
    var_v = np.var(vol)
    if var_v < 1e-12:
        return float("nan")
    # λ ∝ Cov(r,V)/Var(V); scale so typical SPY gives order-of-magnitude similar to 0.20
    # Raw ratio is in (return units)/(log-vol units). Normalize by std(r) to get dimensionless.
    raw_lambda = cov_rv / var_v
    std_r = np.std(returns)
    # Dimensionally: λ in model is (price change per unit demand). Our proxy gives
    # sensitivity of return to log-volume. Scale by price level to get comparable magnitude.
    # Heuristic: multiply by 10 to bring typical values into [0.1, 0.5] range.
    lambda_hat = raw_lambda * 10.0
    lambda_hat = np.clip(lambda_hat, 0.01, 1.0)
    return float(lambda_hat)


def classify_regime(kappa: float, sigma_r: Optional[float] = None) -> str:
    """
    Map κ to baseline vs stress (regime by anchoring strength only).

    Baseline: κ ∈ [0.002, 0.02] (high anchoring).
    Stress: κ < 0.001 (weak anchoring).
    """
    if kappa is None or (isinstance(kappa, float) and np.isnan(kappa)):
        return "indeterminate"
    if kappa < 0.001:
        return "stress (weak anchoring)"
    if 0.002 <= kappa <= 0.02:
        return "baseline"
    return "borderline"


def run_estimation(ticker: str = "SPY", years: int = 8, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run full estimation and return results dict."""
    if df is None:
        df = fetch_data(ticker=ticker, years=years)
    mean_price = float(np.nanmean(df["price"].astype(float).values)) if df is not None and (not df.empty) and "price" in df.columns else float("nan")
    scale_s = (1.0 / mean_price) if mean_price > 1e-12 else float("nan")
    kappa, half_life, rho = estimate_kappa(df)
    sigma_r, sigma_r_ann = estimate_sigma_r(df)
    phi = (sigma_r / scale_s) if (scale_s > 1e-12 and (not np.isnan(sigma_r))) else float("nan")
    regime = classify_regime(kappa, sigma_r)
    def _safe(x):
        if x is None or (isinstance(x, (int, float)) and (np.isnan(x) or np.isinf(x))):
            return None
        return float(x)

    return {
        "ticker": ticker,
        "n_obs": int(len(df)),
        "start_date": str(df.index[0].date()) if len(df) else None,
        "end_date": str(df.index[-1].date()) if len(df) else None,
        "kappa": _safe(kappa),
        "half_life_days": _safe(half_life),
        "ar1_rho": _safe(rho),
        "mean_price": _safe(mean_price),
        "scale_s": _safe(scale_s),
        "sigma_r": _safe(sigma_r),
        "sigma_r_ann": _safe(sigma_r_ann),
        "phi": _safe(phi),
        "regime": regime,
        "baseline": {"kappa": 0.005, "half_life_days": 139},
    }


def plot_regime(
    results: Dict[str, Any],
    out_path: Path,
) -> None:
    """Plot (κ, σr) with baseline/stress regions by anchoring strength. σr is calibration target."""
    fig, ax = plt.subplots(figsize=(5, 4))
    kappa = results.get("kappa")
    sigma_r = results.get("sigma_r")
    ticker = results.get("ticker", "SPY")

    # Regime by κ only: baseline κ∈[0.002,0.02], stress κ<0.001
    k_baseline = np.array([0.002, 0.02, 0.02, 0.002])
    s_baseline = np.array([0, 0, 0.025, 0.025])  # σr range for shading
    paper_kappa, paper_sigma_r = 0.005, 0.01
    xlim, ylim = (0, 0.025), (0, 0.025)

    ax.fill(k_baseline, s_baseline, alpha=0.3, color="green", label="Baseline (high anchoring)")
    ax.axvspan(0, 0.001, alpha=0.2, color="red", label="Stress (weak anchoring)")
    ax.plot([paper_kappa], [paper_sigma_r], "go", markersize=10, label="Model baseline")

    if kappa is not None and sigma_r is not None and not np.isnan(kappa):
        ax.plot(kappa, sigma_r, "k*", markersize=14, label=f"{ticker}")
        ax.annotate(
            f"{ticker}\n$\\kappa$={kappa:.4f}\n$\\sigma_r$={sigma_r:.4f}",
            xy=(kappa, sigma_r),
            xytext=(kappa + 0.001, sigma_r + 0.003),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax.set_xlabel(r"$\kappa$ (mean-reversion, per day)")
    ax.set_ylabel(r"$\sigma_r$ (return vol, calibration target)")
    ax.set_title("One-asset moment match: regime by anchoring strength")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-asset moment match vignette")
    parser.add_argument("--ticker", default="SPY", help="ETF ticker (default: SPY)")
    parser.add_argument("--years", type=int, default=8, help="Years of history")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip regime figure")
    parser.add_argument("--csv", type=str, default=None, help="Load from CSV (columns: Date, Close, Volume) when yfinance unavailable")
    parser.add_argument("--skip-rows", type=int, default=None, help="Skip N rows at start of CSV (e.g. MacroTrends header)")
    parser.add_argument(
        "--paper-config",
        type=str,
        default=None,
        help="Paper baseline config (YAML). Used for the Baseline column and the normalization λ.",
    )
    parser.add_argument(
        "--plan-b",
        action="store_true",
        help="Calibrate phi by simulation root-finding (variance-consistent Plan B), not just sigma_r/scale_s.",
    )
    parser.add_argument(
        "--plan-b-sim-config",
        type=str,
        default=None,
        help="Simulation config YAML used for Plan B (default: paper_baseline.yaml).",
    )
    parser.add_argument("--plan-b-burn", type=int, default=2000, help="Burn-in increments for Plan B moments.")
    parser.add_argument("--plan-b-sample", type=int, default=20000, help="Sample increments for Plan B moments.")
    parser.add_argument(
        "--plan-b-seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds for Plan B (common random numbers).",
    )
    parser.add_argument("--plan-b-max-iter", type=int, default=18, help="Max bisection iterations for Plan B.")
    parser.add_argument("--plan-b-tol-abs", type=float, default=1e-4, help="Absolute tolerance on daily sigma_r in Plan B.")
    parser.add_argument("--plan-b-tol-rel", type=float, default=5e-3, help="Relative tolerance on daily sigma_r in Plan B.")
    args = parser.parse_args()

    # Resolve output directory: paper_artifacts/moment_match relative to project root
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        proj_root = Path(__file__).resolve().parents[2]
        out_dir = proj_root / "paper_artifacts" / "moment_match"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = None
    if args.csv:
        csv_path = Path(args.csv)
        if csv_path.exists():
            skip = args.skip_rows
            if skip is None:
                # Auto-detect MacroTrends-style header (skip until we find date,close,volume)
                with open(csv_path, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        low = line.lower()
                        if "date" in low and ("close" in low or "close/last" in low) and "volume" in low:
                            skip = i
                            break
                    else:
                        skip = 0
            df = pd.read_csv(csv_path, skiprows=range(skip) if skip else 0, index_col=0, parse_dates=True)
            # Handle common column names: close, Close, Close/Last, Adj Close
            close_col = next((c for c in ["close", "Close", "Close/Last", "Adj Close"] if c in df.columns), None)
            vol_col = next((c for c in ["volume", "Volume"] if c in df.columns), None)
            if close_col and vol_col:
                df = df.rename(columns={close_col: "price", vol_col: "volume"})[["price", "volume"]]
                # Strip $ from price if present
                if df["price"].dtype == object:
                    df["price"] = df["price"].astype(str).str.replace("$", "", regex=False).astype(float)
            else:
                df = None
    if df is None:
        df = fetch_data(ticker=args.ticker, years=args.years)
    min_rows = 10 if args.csv else 100
    if df is not None and (df.empty or len(df) < min_rows):
        if not args.csv:
            print("Warning: yfinance returned insufficient data (rate limit?). Using synthetic SPY-like data.")
        else:
            print(f"Warning: CSV has only {len(df)} rows; need at least {min_rows}. Using synthetic fallback.")
        df = _synthetic_spy_like(n_days=2000)
    results = run_estimation(ticker=args.ticker, years=args.years, df=df)

    # Paper baseline parameters (Table "Baseline" column + λ normalization)
    default_paper_cfg = Path(__file__).resolve().parents[2] / "code" / "configs" / "paper_baseline.yaml"
    paper_cfg_path = Path(args.paper_config) if args.paper_config else default_paper_cfg
    try:
        paper = _load_paper_baseline_params(paper_cfg_path)
    except Exception:
        paper = {"kappa": 0.005, "lambda": 0.20, "phi": 0.50}
    baseline_kappa = float(paper["kappa"])
    baseline_lambda = float(paper["lambda"])
    baseline_phi = float(paper["phi"])
    baseline_hl = (LN2 / baseline_kappa) if baseline_kappa > 1e-12 else float("nan")
    baseline_sigma_eta = (baseline_phi / baseline_lambda) if baseline_lambda > 1e-12 else float("nan")

    phi_naive = float(results.get("phi")) if results.get("phi") is not None else float("nan")
    phi_hat = phi_naive
    plan_b_payload = None
    if args.plan_b:
        # Import simulation code only when needed.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../code
        from dataclasses import replace as _replace

        from mfg_oc.config import load_config
        from mfg_oc.phi_plan_b import calibrate_phi_bisection, render_variance_share_table_tex

        sim_cfg_path = Path(args.plan_b_sim_config) if args.plan_b_sim_config else default_paper_cfg
        sim_cfg = load_config(sim_cfg_path)

        burn = int(args.plan_b_burn)
        sample = int(args.plan_b_sample)
        horizon_needed = burn + sample + 2  # ensure sufficient increments under logged-path alignment
        sim_cfg = _replace(sim_cfg, horizon=horizon_needed)

        # Use estimated kappa from price-only data when available; keep lambda as a normalization choice.
        kappa_hat = results.get("kappa")
        kappa_used = float(kappa_hat) if kappa_hat is not None else float(sim_cfg.market.kappa)
        impact_used = float(baseline_lambda)

        seeds = []
        for part in str(args.plan_b_seeds).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
        if not seeds:
            seeds = [0, 1, 2, 3, 4]

        sigma_r_target = float(results.get("sigma_r")) if results.get("sigma_r") is not None else float("nan")
        scale_s = float(results.get("scale_s")) if results.get("scale_s") is not None else float("nan")
        if not (np.isfinite(sigma_r_target) and np.isfinite(scale_s) and scale_s > 0.0):
            print("Warning: Plan B requested but sigma_r/scale_s is not finite. Falling back to naive phi.")
        else:
            plan_b_payload = calibrate_phi_bisection(
                base_cfg=sim_cfg,
                sigma_r_target=sigma_r_target,
                scale_s=scale_s,
                seeds=seeds,
                burn=burn,
                sample=sample,
                kappa=kappa_used,
                impact=impact_used,
                phi_min=0.0,
                phi_max=phi_naive if np.isfinite(phi_naive) else None,
                tol_abs=float(args.plan_b_tol_abs),
                tol_rel=float(args.plan_b_tol_rel),
                max_iter=int(args.plan_b_max_iter),
                log_metrics=False,
            )
            phi_star = plan_b_payload.get("phi_star")
            if phi_star is None:
                print(f"Warning: Plan B bracketing failed ({plan_b_payload.get('status')}). Falling back to naive phi.")
            else:
                phi_hat = float(phi_star)
                try:
                    m = plan_b_payload.get("moments", {})
                    tex = render_variance_share_table_tex(m)
                    (out_dir / "table_moment_match_phi_variance_shares.tex").write_text(tex, encoding="utf-8")
                except Exception as e:
                    print(f"Warning: failed to write variance-share table: {e}")

    # Overwrite the headline phi to match the paper narrative: Plan B if enabled, else naive.
    results["phi"] = float(phi_hat) if np.isfinite(phi_hat) else None
    results["phi_naive"] = float(phi_naive) if np.isfinite(phi_naive) else None
    if plan_b_payload is not None:
        results["phi_plan_b"] = float(phi_hat) if np.isfinite(phi_hat) else None
        results["phi_plan_b_status"] = str(plan_b_payload.get("status"))

    sigma_eta_implied = (phi_hat / baseline_lambda) if baseline_lambda > 1e-12 else float("nan")

    # Write JSON
    json_path = out_dir / f"estimates_{args.ticker}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write LaTeX table fragments for vignette
    def _tex(v, fmt=".4f"):
        if v is None:
            return "---"
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "---"
        return f"{v:{fmt}}"
    (out_dir / "table_moment_match_kappa.tex").write_text(_tex(results.get("kappa")))
    (out_dir / "table_moment_match_sigma_r.tex").write_text(_tex(results.get("sigma_r")))
    (out_dir / "table_moment_match_sigma_r_ann.tex").write_text(_tex(results.get("sigma_r_ann")))
    (out_dir / "table_moment_match_hl.tex").write_text(_tex(results.get("half_life_days"), ".1f"))
    (out_dir / "table_moment_match_regime.tex").write_text(results.get("regime", "---"))
    (out_dir / "table_moment_match_mean_price.tex").write_text(_tex(results.get("mean_price"), ".2f"))
    (out_dir / "table_moment_match_scale_s.tex").write_text(_tex(results.get("scale_s"), ".6f"))
    (out_dir / "table_moment_match_phi.tex").write_text(_tex(results.get("phi"), ".3f"))
    (out_dir / "table_moment_match_phi_naive.tex").write_text(_tex(results.get("phi_naive"), ".3f"))

    (out_dir / "table_moment_match_kappa_baseline.tex").write_text(_tex(baseline_kappa, ".3f"))
    (out_dir / "table_moment_match_hl_baseline.tex").write_text(_tex(baseline_hl, ".0f"))
    (out_dir / "table_moment_match_phi_baseline.tex").write_text(_tex(baseline_phi, ".2f"))
    (out_dir / "table_moment_match_lambda_norm.tex").write_text(_tex(baseline_lambda, ".2f"))
    (out_dir / "table_moment_match_sigma_eta_implied.tex").write_text(_tex(sigma_eta_implied, ".2f"))
    (out_dir / "table_moment_match_sigma_eta_baseline.tex").write_text(_tex(baseline_sigma_eta, ".2f"))
    n = results.get("n_obs")
    n_tex = (f"{n:,}".replace(",", r"{,}") if n is not None and n >= 1000 else str(n)) if n is not None else "---"
    (out_dir / "table_moment_match_n_obs.tex").write_text(n_tex)
    start = results.get("start_date")
    end = results.get("end_date")
    start_yr = start[:4] if start and len(start) >= 4 else "---"
    end_yr = end[:4] if end and len(end) >= 4 else "---"
    (out_dir / "table_moment_match_start.tex").write_text(start_yr)
    (out_dir / "table_moment_match_end.tex").write_text(end_yr)

    if not args.no_plot:
        fig_path = out_dir / "fig_moment_match_regime.png"
        plot_regime(results, fig_path)

    print(f"Estimates written to {json_path}")
    print(f"  kappa = {results.get('kappa')}, sigma_r = {results.get('sigma_r')}, half-life = {results.get('half_life_days')} days")
    print(f"  Regime: {results.get('regime')}")


if __name__ == "__main__":
    main()
