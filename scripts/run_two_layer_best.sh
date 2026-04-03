#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=.mplconfig
export PYTHONPYCACHEPREFIX=.pycache

# Two-layer PINN lives in `pinn-workflow-2layer/` and is evaluated against the 2-layer FEM.
export PINN_TWO_LAYER_WORKFLOW_DIR="${PINN_TWO_LAYER_WORKFLOW_DIR:-pinn-workflow-2layer}"

python3 compare_two_layer_pinn_fem.py
