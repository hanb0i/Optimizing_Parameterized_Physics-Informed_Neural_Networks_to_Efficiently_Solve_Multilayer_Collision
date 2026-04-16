# `graphs/`: Paper Figures and Ablation Placeholders

This folder contains reproducible scripts to generate IEEE-style, paper-ready figures (or clearly marked placeholders when required artifacts are missing).

## Quick Start

From the repo root:

- Generate all figures:
  - `python3 graphs/make_all_graphs.py`
- Geometry and boundary conditions schematic:
  - `python3 graphs/scripts/plot_geometry_bc.py`
- Ablation table (placeholder until ablation results exist):
  - `python3 graphs/scripts/plot_ablation_table.py`
- Error heatmap (uses the trained 3-layer PINN checkpoint):
  - `python3 graphs/scripts/plot_error_heatmap.py`

All figures are written to `graphs/figures/` as both PNG and PDF.

## Ablation Runner (Generates Real Results)

Run the three-layer ablation sweep (trains and evaluates multiple variants on the
same random-interior FEM-referenced protocol used by the generalization study):

`python3 graphs/scripts/run_ablation_three_layer.py`

Outputs:

- `graphs/data/ablation_results.csv` (schema: `variant,mean_mae,worst_mae`, values are percent)
- Per-variant checkpoints and evaluator visualizations under `graphs/data/ablation_runs/`
- Logs under `graphs/data/ablation_logs/`

Then render the ablation table figure:

`python3 graphs/scripts/plot_ablation_table.py`

Notes:

- `mean_mae` / `worst_mae` are top-surface `u_z` MAE percentages over the
  random-interior evaluation cases, not the older exhaustive corner sweep.
- The full-framework row automatically uses
  `graphs/data/three_layer_compliance_calibration.json` when that artifact is present.
- The runner sets `PINN_WARM_START=0` by default for fairness.
- Use `--skip-train` to reuse existing checkpoints.
- Use `--epochs-soap` and `--device` to control runtime and device.

## Required Artifacts

### Error heatmap

- Requires a trained checkpoint at `PINN_MODEL_PATH` (defaults to `pinn-workflow/pinn_model.pth`).
- Optional overrides:
  - `PINN_ERROR_CASE` (label only)
  - `PINN_ERROR_E` (comma list, e.g. `1.0,10.0,10.0`)
  - `PINN_ERROR_T` (comma list, e.g. `0.10,0.02,0.02`)
