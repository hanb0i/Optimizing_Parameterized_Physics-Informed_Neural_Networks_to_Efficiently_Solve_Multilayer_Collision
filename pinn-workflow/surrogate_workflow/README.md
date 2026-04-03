# Phase 1 Surrogate Workflow (Three-Layer PINN Baseline)

Trains a small MLP surrogate that maps three-layer design parameters to a scalar response
metric computed from the trained parametric PINN.

## What it does

- Generates a dataset by querying the PINN over a load-patch grid and extracting peak top-surface deflection
- Trains an MLP surrogate on normalized inputs/outputs
- Validates with hold-out accuracy, prediction vs truth, and trend fidelity
- Runs a simple optimization safety check

## Run

```bash
python pinn-workflow/surrogate_workflow/run_phase1.py --regenerate
```

Or run the repo wrapper (sets `MPLCONFIGDIR` and keeps PINN compliance scaling consistent):

```bash
./scripts/run_three_layer_surrogate.sh
```

Override dataset/training size for quick smoke runs:

```bash
python pinn-workflow/surrogate_workflow/run_phase1.py --regenerate --n-samples 50 --max-epochs 300
```

Recommended (higher accuracy) run on a GPU box:

```bash
export SURROGATE_DEVICE=cuda
./scripts/run_three_layer_surrogate.sh --n-samples 2000 --max-epochs 6000
```

Outputs are saved in `pinn-workflow/surrogate_workflow/outputs` (ignored by git).
