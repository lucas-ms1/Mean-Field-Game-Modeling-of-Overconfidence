# Code (simulation + replication scripts)

This folder contains the Python code used to generate the contents of `../paper_artifacts/` (run-level CSVs, summary JSONs, figures, and LaTeX tables).

## Install

```bash
python -m venv .venv
# activate the venv:
# - Windows PowerShell: .venv\Scripts\Activate.ps1
# - macOS/Linux:       source .venv/bin/activate
pip install -e .
```

## Generate paper artifacts

```bash
python -m scripts.build_paper_artifacts --help
python -m scripts.build_paper_artifacts --config configs/paper_baseline.yaml --mode baseline_regimes --seeds 50
```

## Moment-match vignette (SPY)

```bash
python -m scripts.estimate_moment_match --help
python -m scripts.estimate_moment_match --ticker SPY
```
