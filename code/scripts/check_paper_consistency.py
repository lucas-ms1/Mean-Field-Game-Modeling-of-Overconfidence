"""
Lightweight paper consistency checks.

Checks:
- duplicate LaTeX labels / duplicate \\tag{...}
- placeholder text scan ("TODO", "(calibrate)", etc.)
- Table 8 arithmetic identity: h ↔ kappa via h = ln(2)/kappa
- (optional-but-default) ban "all figures/tables" phrasing that is easy to falsify
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import yaml

LN2 = math.log(2.0)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_tex_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.tex"))


def _parse_float_from_fragment(path: Path) -> Optional[float]:
    s = _read_text(path).strip()
    if s in {"---", ""}:
        return None
    # Strip common LaTeX wrappers and keep first number.
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _check_duplicate_labels(tex_files: list[Path]) -> list[str]:
    label_re = re.compile(r"\\label\{([^}]+)\}")
    labels: dict[str, list[Path]] = defaultdict(list)
    for f in tex_files:
        for lab in label_re.findall(_read_text(f)):
            labels[lab].append(f)
    errors: list[str] = []
    for lab, files in sorted(labels.items()):
        if len(files) > 1:
            where = ", ".join(str(p) for p in files)
            errors.append(f"Duplicate label '{lab}' in: {where}")
    return errors


def _check_duplicate_tags(tex_files: list[Path]) -> list[str]:
    tag_re = re.compile(r"\\tag\{([^}]+)\}")
    tags: dict[str, list[Path]] = defaultdict(list)
    for f in tex_files:
        for tag in tag_re.findall(_read_text(f)):
            tags[tag].append(f)
    errors: list[str] = []
    if tags:
        for tag, files in sorted(tags.items()):
            where = ", ".join(str(p) for p in files)
            errors.append(f"Manual \\\\tag{{{tag}}} found (ban tags for robust numbering): {where}")
        return errors
    for tag, files in sorted(tags.items()):
        if len(files) > 1:
            where = ", ".join(str(p) for p in files)
            errors.append(f"Duplicate \\\\tag{{{tag}}} in: {where}")
    return errors


def _check_placeholders(paths: list[Path]) -> list[str]:
    patterns = [
        r"\bTODO\b",
        r"\bFIXME\b",
        r"\(calibrate\)",
        r"\bPlaceholder\b",
    ]
    rx = re.compile("|".join(patterns))
    errors: list[str] = []
    for p in paths:
        text = _read_text(p)
        for m in rx.finditer(text):
            snippet = text[max(0, m.start() - 20) : min(len(text), m.end() + 20)].replace("\n", " ")
            errors.append(f"Placeholder match in {p}: ...{snippet}...")
            break  # one hit per file is enough
    return errors


def _check_all_figures_claim(tex_files: list[Path]) -> list[str]:
    bad_phrases = [
        "behind all reported figures and tables",
        "behind all figures and tables",
        "all reported figures and tables",
    ]
    errors: list[str] = []
    for f in tex_files:
        low = _read_text(f).lower()
        for phrase in bad_phrases:
            if phrase in low:
                errors.append(f"Banned phrasing '{phrase}' found in {f}")
    return errors


def _check_table8_identities(artifacts_dir: Path) -> list[str]:
    """
    Check that the moment-match fragments satisfy the half-life identity:
      h ≈ ln(2)/kappa

    We check both the estimated pair and the baseline pair (if fragments exist).
    """
    mm = artifacts_dir / "moment_match"
    if not mm.exists():
        return [f"Missing moment_match directory: {mm}"]

    def chk(kappa_path: Path, hl_path: Path, name: str, tol: float) -> Optional[str]:
        kappa = _parse_float_from_fragment(kappa_path)
        hl = _parse_float_from_fragment(hl_path)
        if kappa is None or hl is None:
            return f"Missing numeric values for {name}: kappa={kappa} hl={hl}"
        implied = LN2 / kappa if kappa > 1e-12 else float("inf")
        if abs(implied - hl) > tol:
            return f"{name} half-life mismatch: ln2/kappa={implied:.3f} vs hl={hl:.3f} (tol={tol})"
        return None

    errors: list[str] = []
    est_err = chk(mm / "table_moment_match_kappa.tex", mm / "table_moment_match_hl.tex", "Estimate", tol=2.0)
    if est_err:
        errors.append(est_err)

    k_b = mm / "table_moment_match_kappa_baseline.tex"
    h_b = mm / "table_moment_match_hl_baseline.tex"
    if k_b.exists() and h_b.exists():
        base_err = chk(k_b, h_b, "Baseline", tol=2.0)
        if base_err:
            errors.append(base_err)
    return errors


def _check_paper_baseline_config(config_path: Path) -> list[str]:
    if not config_path.exists():
        return [f"Missing paper baseline config: {config_path}"]
    data = yaml.safe_load(_read_text(config_path))
    if not isinstance(data, dict) or "market" not in data:
        return [f"Invalid baseline config (missing market): {config_path}"]
    market = data.get("market", {})
    errors: list[str] = []
    for key in ("kappa", "impact", "noise_sigma"):
        if key not in market:
            errors.append(f"Baseline config missing market.{key}: {config_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper consistency checks")
    parser.add_argument("--paper-dir", type=str, default="paper/current/ieee")
    parser.add_argument("--artifacts-dir", type=str, default="paper/current/paper_artifacts")
    parser.add_argument("--baseline-config", type=str, default="code/configs/paper_baseline.yaml")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    artifacts_dir = Path(args.artifacts_dir)
    baseline_cfg = Path(args.baseline_config)

    tex_files = list(_iter_tex_files(paper_dir))
    if not tex_files:
        print(f"No .tex files found under {paper_dir}", file=sys.stderr)
        return 2

    errors: list[str] = []
    errors.extend(_check_duplicate_labels(tex_files))
    errors.extend(_check_duplicate_tags(tex_files))
    errors.extend(_check_all_figures_claim(tex_files))

    # Placeholder scan over paper tex + artifact tex fragments.
    placeholder_paths: list[Path] = []
    placeholder_paths.extend(tex_files)
    if artifacts_dir.exists():
        placeholder_paths.extend(sorted(artifacts_dir.rglob("*.tex")))
    errors.extend(_check_placeholders(placeholder_paths))

    errors.extend(_check_paper_baseline_config(baseline_cfg))
    errors.extend(_check_table8_identities(artifacts_dir))

    if errors:
        print("FAIL: consistency checks found issues:\n", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1

    print("OK: paper consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
