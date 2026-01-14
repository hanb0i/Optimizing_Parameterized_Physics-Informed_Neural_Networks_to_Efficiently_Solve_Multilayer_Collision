import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
Layer_Interfaces = [0.0, H]

# --- Material Properties ---
E_vals = [1.0]  # Normalized
nu_vals = [0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 1.0  # Load magnitude
LOAD_PATCH_X = [Lx/3, 2*Lx/3]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_SOAP = 2000
EPOCHS_SSBFGS = 30

# SOAP optimizer
SOAP_PRECONDITION_FREQUENCY = 10

# SciPy self-scaled BFGS optimizer
SS_BFGS_METHOD = "BFGS"
SS_BFGS_VARIANT = "SSBFGS_AB"
SS_BFGS_MAXITER = 1
SS_BFGS_GTOL = 0.0
SS_BFGS_INITIAL_SCALE = False

# Loss Weights
WEIGHTS = {
    'pde': 1.0,
    'bc': 1.0,
    'load': 1.0,
}

# Sampling
N_INTERIOR = 10000
N_BOUNDARY = 2000
