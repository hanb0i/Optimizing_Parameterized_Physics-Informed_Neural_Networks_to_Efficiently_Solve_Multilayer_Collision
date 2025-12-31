
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

# Load patch boundaries (normalized coordinates)
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # [0.333, 0.667]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # [0.333, 0.667]

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 2000 # Increased to enforce load and reduce underfit
EPOCHS_LBFGS = 100 # Increased from 500. Resampling here. Should help convergence. 
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 1.0,    # Increased from 1.0
    'bc': 1.0,      # Reduced, as hard constraint handles side BCs now
    'load': 1.0, # Heavily increased to match traction target
    'interface_u': 1.0 
}
# Sampling
N_INTERIOR = 10000 # Per layer
N_BOUNDARY = 2000  # Per face type

# Fourier Features
FOURIER_DIM = 0 # Number of Fourier frequencies
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling
