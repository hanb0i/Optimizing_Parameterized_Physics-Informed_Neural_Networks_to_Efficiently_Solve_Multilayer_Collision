import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output paths
OUTPUT_DIR = os.path.join(ROOT_DIR, "surrogate_workflow", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATASET_PATH = os.path.join(OUTPUT_DIR, "phase1_dataset.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "surrogate_model.pt")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "phase1_summary.txt")

# Design parameters: Full 5D Parametric Space
DESIGN_PARAMS = ["E", "thickness", "restitution", "friction", "impact_velocity"]
DESIGN_RANGES = {
    "E": (1.0, 10.0),
    "thickness": (0.05, 0.15),
    "restitution": (0.1, 0.9),
    "friction": (0.0, 0.6),
    "impact_velocity": (0.2, 2.0)
}

# Geometry and loading (standardized across workflow)
GEOMETRY = {
    "Lx": 1.0,
    "Ly": 1.0,
    "H": 0.1,
}

# Dataset generation
N_SAMPLES = 225
SEED = 7
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1

# Model hyperparameters
HIDDEN_LAYERS = 3
HIDDEN_UNITS = 64
ACTIVATION = "tanh"

# Training
LEARNING_RATE = 1e-3
MAX_EPOCHS = 3000
BATCH_SIZE = 32
PATIENCE = 300
MIN_DELTA = 1e-6

# Validation and sweeps
TREND_SWEEP_PARAM = "E"
TREND_SWEEP_POINTS = 50
OPT_CANDIDATES = 1000
TREND_ANCHOR_POINTS = 25
