import os

import numpy as np

import pinn_config as pc

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output paths
OUTPUT_DIR = os.path.join(ROOT_DIR, "surrogate_workflow", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATASET_PATH = os.path.join(OUTPUT_DIR, "phase1_dataset.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "surrogate_model.pt")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "phase1_summary.txt")

# Design parameters for the two-layer plate PINN.
DESIGN_PARAMS = ["E1", "t1", "E2", "t2"]
DESIGN_RANGES = {
    "E1": (float(pc.E_RANGE[0]), float(pc.E_RANGE[1])),
    "t1": (float(pc.T1_RANGE[0]), float(pc.T1_RANGE[1])),
    "E2": (float(pc.E_RANGE[0]), float(pc.E_RANGE[1])),
    "t2": (float(pc.T2_RANGE[0]), float(pc.T2_RANGE[1])),
}

# Dataset generation
N_SAMPLES = 500
SEED = 7
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1

# Model hyperparameters
HIDDEN_LAYERS = 4
HIDDEN_UNITS = 128
ACTIVATION = "tanh"

# Training
LEARNING_RATE = 1e-3
MAX_EPOCHS = 3000
BATCH_SIZE = 64
PATIENCE = 200
MIN_DELTA = 1e-6

# Validation and sweeps
TREND_SWEEP_PARAM = "E1"
TREND_SWEEP_POINTS = 50
OPT_CANDIDATES = 1000
TREND_ANCHOR_POINTS = 25

def mid_design() -> np.ndarray:
    return np.array([(DESIGN_RANGES[name][0] + DESIGN_RANGES[name][1]) * 0.5 for name in DESIGN_PARAMS], dtype=float)

