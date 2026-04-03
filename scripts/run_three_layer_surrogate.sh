#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=.mplconfig
export PYTHONPYCACHEPREFIX=.pycache

# Keep surrogate evaluation consistent with the tuned 3-layer PINN sweep settings.
export PINN_E_COMPLIANCE_POWER=0.95 PINN_PDE_DECOMPOSE_BY_LAYER=1 PINN_THICKNESS_COMPLIANCE_ALPHA=3 PINN_DISPLACEMENT_COMPLIANCE_SCALE=1 \
  PINN_DATA_E_VALUES=1.0,10.0 PINN_DATA_T1_VALUES=0.02,0.10 PINN_DATA_T2_VALUES=0.02,0.10 PINN_DATA_T3_VALUES=0.02,0.10 \
  PINN_EVAL_E_VALUES=1.0,10.0 PINN_EVAL_T1_VALUES=0.02,0.10 PINN_EVAL_T2_VALUES=0.02,0.10 PINN_EVAL_T3_VALUES=0.02,0.10 \
  PINN_FEM_NE_X=10 PINN_FEM_NE_Y=10 PINN_FEM_NE_Z=4

# Baseline response grid resolution used when querying the PINN under the load patch.
# Lower is faster; higher is more accurate.
export SURROGATE_TOP_NX="${SURROGATE_TOP_NX:-11}"

# Surrogate tuning knobs (aim: <5% rel error on the 3-layer extreme-case checks).
export SURROGATE_ANCHOR_REPEAT="${SURROGATE_ANCHOR_REPEAT:-40}"
export SURROGATE_HIDDEN_UNITS="${SURROGATE_HIDDEN_UNITS:-512}"
export SURROGATE_HIDDEN_LAYERS="${SURROGATE_HIDDEN_LAYERS:-5}"
export SURROGATE_ACTIVATION="${SURROGATE_ACTIVATION:-relu}"
export SURROGATE_LEARNING_RATE="${SURROGATE_LEARNING_RATE:-5e-4}"
export SURROGATE_PATIENCE="${SURROGATE_PATIENCE:-800}"
export SURROGATE_TARGET_REL_ERR_PCT="${SURROGATE_TARGET_REL_ERR_PCT:-5.0}"

python3 pinn-workflow/surrogate_workflow/run_phase1.py --regenerate "${@}"
python3 pinn-workflow/surrogate_workflow/verify_phase1.py
