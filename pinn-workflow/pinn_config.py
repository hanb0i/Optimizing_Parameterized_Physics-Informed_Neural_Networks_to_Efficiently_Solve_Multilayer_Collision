
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
# Layer interfaces (assuming equal thickness for simplicity unless specified)
# z goes from 0 to H.
# Layer 1: 0 to H/3
# Layer 2: H/3 to 2H/3
# Layer 3: 2H/3 to H
Layer_Interfaces = [0.0, H/3, 2*H/3, H]

# --- Material Properties ---
# Young's Modulus (E) and Poisson's Ratio (nu)
# Can be different per layer
E_vals = [360.0, 360.0, 360.0] # Match FEA material
nu_vals = [0.3, 0.3, 0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 0.1 # Load magnitude

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 1000 # Optimal balance (Knee point at ~200-500)
EPOCHS_LBFGS = 1500 # Testing user's hypothesis (1000-2000 range)
WEIGHTS = {
    'pde': 1.0,    # Balanced with Load
    'bc': 100.0,    # Strong constraint (Trial 12)
    'load': 100.0,   # Natural Physics (Trial 12)
    'interface_u': 100.0, # Balanced (Trial 12)
    'interface_t': 1.0   # Matches Traction (Trial 12)
}
# Sampling
N_INTERIOR = 2000 # Standard resolution (Trial 12)
N_BOUNDARY = 2000  # Standard resolution

# Fourier Features
FOURIER_DIM = 64 # Number of Fourier frequencies
FOURIER_SCALE = 5.0 # Increased to capture sharp load edges
OUTPUT_SCALE = 1.0 # Removed scaling to allow natural physics-driven magnitude
