
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
p0 = 0.1 # Load magnitude (Matched to FEA: 0.1)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3 # Restore standard LR for Hard Constraints
EPOCHS_ADAM = 3000 # Optimized based on convergence analysis
EPOCHS_LBFGS = 2000 
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 0.0,     # Zero physics to force deformation
    'bc': 1.0,      
    'load': 10000.0, # Maximum forcing
    'interface_u': 100.0,
    'interface_stress': 10.0 
}
# Sampling
N_INTERIOR = 8000 # Standard sampling for speed
N_BOUNDARY = 2000

# Model Architecture
HIDDEN_LAYERS = 5
HIDDEN_UNITS = 64

# Fourier Features
FOURIER_DIM = 64 
FOURIER_SCALE = 1.5
OUTPUT_SCALE = 0.03 # Reduced for small displacement (Target ~0.0016)
