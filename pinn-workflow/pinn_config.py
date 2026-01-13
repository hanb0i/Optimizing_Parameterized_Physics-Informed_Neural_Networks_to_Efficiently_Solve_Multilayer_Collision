
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
# Single layer (homogeneous material)
# z goes from 0 to H
Layer_Interfaces = [0.0, H]

# --- Material Properties ---
# Young's Modulus (E) and Poisson's Ratio (nu)
# Single layer to match FEM
E_vals = [1.0] # Normalized
nu_vals = [0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 1.0 # Load magnitude

# --- Unit-consistent loss scaling ---
# div(sigma) has units of stress/length; scale by a characteristic length.
PDE_LENGTH_SCALE = H

# --- Boundary condition handling ---
# Use hard mask early for shape, then switch to soft BCs for magnitude.
USE_HARD_SIDE_BC = True
HARD_BC_EPOCHS = 1000

# Load patch boundaries (normalized coordinates)
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # [0.333, 0.667]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # [0.333, 0.667]

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 2000 # Increased to enforce load and reduce underfit
EPOCHS_LBFGS = 300 # L-BFGS fine-tuning steps; resampling here should help convergence.
# SOAP optimizer
SOAP_PRECONDITION_FREQUENCY = 10 # Lower = more frequent curvature updates; higher = cheaper but less responsive
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 1.0,    # Increased from 1.0
    'bc': 1.0,      # Reduced, as hard constraint handles side BCs now
    'load': 1.0, # Heavily increased to match traction target
    'energy': 1.0, # Energy/compliance balance
    'interface_u': 1.0 
}
# Loss weight ramp: load-first to raise displacement while preserving shape.
WEIGHT_RAMP_EPOCHS = 800
LOAD_WEIGHT_START = 8.0
PDE_WEIGHT_START = 0.2
ENERGY_WEIGHT_START = 0.0
# Sampling
N_INTERIOR = 10000 # Per layer
N_SIDES = 2000  # Clamped side faces
N_TOP_LOAD = 4000  # Load patch (more points to boost displacement)
N_TOP_FREE = 2000  # Top free surface
N_BOTTOM = 2000  # Bottom free surface
UNDER_PATCH_FRACTION = 0.8 # Fraction of interior points sampled under the load patch

# Fourier Features
FOURIER_DIM = 0 # Number of Fourier frequencies
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling

# Output Scaling
