# Phase 1 Surrogate Workflow

Builds a differentiable surrogate that maps design parameters to a single-impact response
metric using the baseline FEA solver.

## What it does

- Generates a dataset via the baseline FEA solver
- Trains an MLP surrogate on normalized inputs/outputs
- Validates with hold-out accuracy, prediction vs truth, and trend fidelity
- Runs an optimization safety check against the baseline solver

## Run

```bash
python surrogate_workflow/run_phase1.py --regenerate
```

Outputs are saved in `surrogate_workflow/outputs`:

- `phase1_dataset.npz` (raw + normalized data + scalers)
- `surrogate_model.pt` (model + scalers)
- `plots/pred_vs_truth.png`
- `plots/trend_fidelity.png`
- `phase1_summary.txt`

## Diagnostics (Phase 1)

Run the diagnostic checklist (PDE summary, mesh sweep, surrogate validation):

```bash
python surrogate_workflow/diagnostics_phase1.py
```

By default, the mesh sweep runs 4 resolutions (20x20x6, 30x30x9, 40x40x12, 50x50x15)
and reports normalized errors vs the finest mesh.

You can override meshes:

```bash
python surrogate_workflow/diagnostics_phase1.py --mesh 30,30,10 --mesh 60,60,20
```

Reports are written to `surrogate_workflow/outputs/phase1_diagnostics.txt`.

## Customize

Edit `surrogate_workflow/config.py`:

- `DESIGN_PARAMS`, `DESIGN_RANGES`
- `N_SAMPLES` (>= 200 for 1-2 params)
- network/training settings
