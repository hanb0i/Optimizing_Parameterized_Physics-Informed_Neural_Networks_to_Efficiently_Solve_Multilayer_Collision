# ==========================================
# PINN CONFIGURATION
# ==========================================
# This file contains all global hyperparameters, material constants,
# and training settings for the Three-Layered Collision PINN model.

import torch
import numpy as np

# --- Geometry Dimensions ---
# The physical domain is a rectangular prism 1.0 x 1.0 x 0.1
Lx = 1.0  # Length in X
Ly = 1.0  # Length in Y
H = 0.1   # Total height in Z
Layer_Interfaces = [0.0, H]  # Z-coordinates of layer interfaces (Single layer currently)

# --- Material Properties ---
# Normalized Young's Modulus and Poisson's Ratio
E_vals = [1.0]   # Young's Modulus for the layer
nu_vals = [0.3]  # Poisson's ratio

# Helper function to convert Engineering constants (E, nu) to Lame parameters (lambda, mu)
# Lambda (lm) and Mu (mu) are used in the Navier-Cauchy elasticity equations.
def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

# Pre-compute Lame parameters for each layer
Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading Conditions ---
# A pressure load is applied to a patch on the top surface.
p0 = 1.0  # Load magnitude (pressure)
# The load is applied within these X and Y bounds (middle third of the plate)
LOAD_PATCH_X = [Lx/3, 2*Lx/3] 
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3  # Initial learning rate for Adam/SOAP
EPOCHS_SOAP = 2000    # Number of epochs for the pre-training phase (SOAP optimizer)
EPOCHS_SSBFGS = 30    # Number of steps for the fine-tuning phase (L-BFGS)

# SOAP optimizer settings
SOAP_PRECONDITION_FREQUENCY = 10  # How often to update the SOAP preconditioner

# SciPy self-scaled BFGS optimizer settings
SS_BFGS_METHOD = "BFGS"
SS_BFGS_VARIANT = "SSBFGS_AB"  # Specific variant of limited-memory BFGS
SS_BFGS_MAXITER = 1            # Iterations per step (kept low as we step manually)
SS_BFGS_GTOL = 0.0             # Gradient tolerance
SS_BFGS_INITIAL_SCALE = False  # Whether to use initial scaling

# Loss Weights
# Weights to balance the different objectives in the loss function
WEIGHTS = {
    'pde': 1.0,   # Weight for Physics (Navier-Cauchy) residuals
    'bc': 1.0,    # Weight for Boundary Conditions (Clamped sides)
    'load': 1.0,  # Weight for the Traction Load on top
}

# Sampling Strategy
N_INTERIOR = 10000  # Number of collocation points inside the domain (for PDE)
N_BOUNDARY = 2000   # Number of points on the boundaries (for BCs)

# Output Scaling
# Critical parameter to scale the Neural Network output to the expected physical magnitude.
# Without this, the network struggles to reach the correct displacement amplitude (~0.28).
OUTPUT_SCALE = 3.55
