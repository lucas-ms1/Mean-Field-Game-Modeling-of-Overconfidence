# Mean Field Game Modeling of Overconfidence

This repository contains:
- `paper/`: LaTeX sources for the paper.
- `code/`: Python simulation + artifact-generation pipeline.
- `paper_artifacts/`: Generated figures, tables, and run-level outputs referenced by the LaTeX sources.
- `paper.pdf`: Prebuilt PDF (for convenience).

## Quickstart (rebuild paper artifacts)

From a shell:

```bash
cd code
python -m venv .venv
# activate the venv:
# - Windows PowerShell: .venv\Scripts\Activate.ps1
# - macOS/Linux:       source .venv/bin/activate
pip install -e .
python -m scripts.build_paper_artifacts --help
python -m scripts.build_paper_artifacts --config configs/paper_baseline.yaml --mode baseline_regimes --seeds 50
python -m scripts.estimate_moment_match --help
```

Artifacts are written to `paper_artifacts/` by default.

## Build the PDF

The LaTeX sources expect `paper_artifacts/` to be present one directory above `paper/`.
If you have a LaTeX distribution installed, you can compile from `paper/` (example):

```bash
cd paper
pdflatex ieee_main.tex
bibtex ieee_main
pdflatex ieee_main.tex
pdflatex ieee_main.tex
```
