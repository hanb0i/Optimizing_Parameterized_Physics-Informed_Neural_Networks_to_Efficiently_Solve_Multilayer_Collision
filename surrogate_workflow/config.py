import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output paths
OUTPUT_DIR = os.path.join(ROOT_DIR, "surrogate_workflow", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATASET_PATH = os.path.join(OUTPUT_DIR, "phase1_dataset.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "surrogate_model.pt")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "phase1_summary.txt")

# Design parameters (1-3 params recommended for Phase 1)
DESIGN_PARAMS = ["E2", "E3"]
DESIGN_RANGES = {
    "E2": (200.0, 500.0),
    "E3": (200.0, 500.0),
}

BASE_LAYER_E = 360.0
BASE_NU = 0.3

# Geometry and loading (matches FEA workflow)
GEOMETRY = {
    "Lx": 1.0,
    "Ly": 1.0,
    "H": 0.1,
}
LAYER_INTERFACES = [0.0, GEOMETRY["H"] / 3.0, 2.0 * GEOMETRY["H"] / 3.0, GEOMETRY["H"]]
LOAD_PATCH = {
    "x_start": 0.33,
    "x_end": 0.67,
    "y_start": 0.33,
    "y_end": 0.67,
    "pressure": 0.1,
}

# Dataset generation
N_SAMPLES = 200
SEED = 7
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15

# Model hyperparameters
HIDDEN_LAYERS = 3
HIDDEN_UNITS = 64
ACTIVATION = "tanh"

# Training
LEARNING_RATE = 1e-3
MAX_EPOCHS = 2000
BATCH_SIZE = 64
PATIENCE = 200
MIN_DELTA = 1e-5

# Validation and checks
TREND_SWEEP_PARAM = "E2"
TREND_SWEEP_POINTS = 25
OPT_CANDIDATES = 1000
TREND_ANCHOR_POINTS = 25
