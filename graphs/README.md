# `graphs/`: Paper Figures and Ablation Placeholders

This folder contains reproducible scripts to generate IEEE-style, paper-ready figures (or clearly marked placeholders when required artifacts are missing).

## Quick Start

From the repo root:

- Geometry and boundary conditions schematic:
  - `python3 graphs/scripts/plot_geometry_bc.py`
- Ablation table (placeholder until ablation results exist):
  - `python3 graphs/scripts/plot_ablation_table.py`
- Error heatmap (uses the trained 3-layer PINN checkpoint):
  - `python3 graphs/scripts/plot_error_heatmap.py`
- Surrogate verification scatter (uses surrogate workflow outputs):
  - `python3 graphs/scripts/plot_surrogate_verification.py`

All figures are written to `graphs/figures/` as both PNG and PDF.

## Ablation Runner (Generates Real Results)

Run the three-layer ablation sweep (trains and evaluates multiple variants):

`python3 graphs/scripts/run_ablation_three_layer.py`

Outputs:

- `graphs/data/ablation_results.csv` (schema: `variant,mean_mae,worst_mae`, values are percent)
- Per-variant checkpoints and evaluator visualizations under `graphs/data/ablation_runs/`
- Logs under `graphs/data/ablation_logs/`

Then render the ablation table figure:

`python3 graphs/scripts/plot_ablation_table.py`

Notes:

- The runner sets `PINN_WARM_START=0` by default for fairness.
- Use `--skip-train` to reuse existing checkpoints.
- Use `--epochs-soap` and `--device` to control runtime and device.

## Required Artifacts (If You Want Non-Placeholder Plots)

### Error heatmap

- Requires a trained checkpoint at `PINN_MODEL_PATH` (defaults to `pinn-workflow/pinn_model.pth`).
- Optional overrides:
  - `PINN_ERROR_CASE` (label only)
  - `PINN_ERROR_E` (comma list, e.g. `1.0,10.0,10.0`)
  - `PINN_ERROR_T` (comma list, e.g. `0.10,0.02,0.02`)

### Surrogate verification scatter

Requires:

- `pinn-workflow/surrogate_workflow/outputs/phase1_dataset.npz`
- `pinn-workflow/surrogate_workflow/outputs/surrogate_model.pt`

These are produced by the surrogate workflow:

- `python3 pinn-workflow/surrogate_workflow/run_phase1.py --regenerate --n-samples 200 --max-epochs 1200 --device cpu`
- `python3 pinn-workflow/surrogate_workflow/verify_phase1.py --device cpu`

If these files are missing, the script emits a placeholder plot and prints what is needed.

