import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height (baseline thickness)
# Single layer (homogeneous material)
# z goes from 0 to H
Layer_Interfaces = [0.0, H]

# --- Material Properties ---
# Young's Modulus (E) and Poisson's Ratio (nu)
# Single layer to match FEM
E_vals = [1.0] # Normalized
nu_vals = [0.3]
# Parameterized PINN settings (do not alter baseline values)
E_RANGE = [1.0, 10.0]
THICKNESS_RANGE = [0.05, 0.15]
RESTITUTION_RANGE = [0.1, 0.9]
FRICTION_RANGE = [0.0, 0.6]
IMPACT_VELOCITY_RANGE = [0.2, 2.0]
PARAM_DIM = 5

# Optional: explicit E sweep values for `verify_parametric_pinn.py`.
# If not set, it uses `np.linspace(E_RANGE[0], E_RANGE[1], PINN_VERIFY_E_STEPS)`.
# VERIFY_E_SWEEP_VALUES = np.linspace(E_RANGE[0], E_RANGE[1], 10).tolist()
# Optional: explicit restitution/friction sweep values for verification.
# VERIFY_RESTITUTION_SWEEP_VALUES = np.linspace(RESTITUTION_RANGE[0], RESTITUTION_RANGE[1], 7).tolist()
# VERIFY_FRICTION_SWEEP_VALUES = np.linspace(FRICTION_RANGE[0], FRICTION_RANGE[1], 7).tolist()
# VERIFY_IMPACT_VELOCITY_SWEEP_VALUES = np.linspace(IMPACT_VELOCITY_RANGE[0], IMPACT_VELOCITY_RANGE[1], 7).tolist()

# Reference parameter values for parity with baseline FEA (which has no restitution/friction).
RESTITUTION_REF = 0.5
FRICTION_REF = 0.3
IMPACT_VELOCITY_REF = 1.0

# Inference-time compliance correction for E:
# Use u = v / E^p instead of v / E (p=1.0). This can help slightly reduce
# high-E under/over-shoot without retraining.
E_COMPLIANCE_POWER = 0.973

# --- Parametric compliance scaling ---
# Many plate-like problems scale strongly with thickness (often ~ 1/t^3).
# We apply a simple thickness-aware scaling in the physics layer:
#   u = (v / E) * (H / t)^alpha
# where H is the baseline thickness (config.H) and t is the sampled thickness.
# Set alpha=0.0 to disable.
THICKNESS_COMPLIANCE_ALPHA = 1.234

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

# --- Network Architecture ---
LAYERS = 4
NEURONS = 64

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 2000
EPOCHS_LBFGS = 0
# SOAP optimizer
SOAP_PRECONDITION_FREQUENCY = 10 # Lower = more frequent curvature updates; higher = cheaper but less responsive
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 5.0,    # Reverted to 5.0 (Optimal: 0.4% Error at E=1, 10% at E=10)
    'bc': 0.7,      # Slightly softer sides so load can gather more budget
    'load': 5.0, # Optimal load weight
    'energy': 0.63, # Per user request
    'impact_invariance': 0.0,  # Set >0 only for neutral-parameter mode
    'impact_contact': 0.0002,   # Reduced to preserve FEA parity in no-supervision mode
    'friction_coulomb': 0.001,  # Reduced to preserve FEA parity in no-supervision mode
    'friction_stick': 0.0005,   # Reduced to preserve FEA parity in no-supervision mode
    'interface_u': 1.0,
    'data': 1.0
}

# Loss weight ramp: load-first to raise displacement while preserving shape.
WEIGHT_RAMP_EPOCHS = 0
LOAD_WEIGHT_START = WEIGHTS['load']
PDE_WEIGHT_START = WEIGHTS['pde']
ENERGY_WEIGHT_START = WEIGHTS['energy']
# Force soft side boundary conditions from the beginning.
FORCE_SOFT_SIDE_BC_FROM_START = True
SOFT_MODE_PDE_WEIGHT_SCALE = 3.0
SOFT_MODE_LOAD_WEIGHT_SCALE = 1.0
# Sampling
N_INTERIOR = 15000 # Per layer
N_SIDES = 2000  # Clamped side faces
N_TOP_LOAD = 6000  # Load patch (more points to boost displacement)
N_TOP_FREE = 2000  # Top free surface
N_BOTTOM = 2000  # Bottom free surface
UNDER_PATCH_FRACTION = 0.95 # More interior points focus under the load patch

#Resampling/perturbation control
SAMPLING_NOISE_SCALE = 0.08  # Larger perturbations widen coverage while still sampling residual-rich zones.

# Auxiliary load-patch average displacement penalty
LOAD_PATCH_UZ_TARGET = -0.05  # Encourage the mean vertical deflection on the load patch
LOAD_PATCH_UZ_WEIGHT = 0.02   # Keep the auxiliary penalty small so shape stays intact

# Fourier Features
FOURIER_DIM = 0 # Number of Fourier frequencies
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling

# Hybrid / Parametric Training Data
N_DATA_POINTS = 9000
DATA_E_VALUES = [1.0, 5.0, 10.0]
DATA_THICKNESS_VALUES = [0.05, 0.1, 0.15]
USE_SUPERVISION_DATA = False

# --- Explicit impact/friction physics controls ---
# When enabled, restitution/friction influence boundary losses directly.
USE_EXPLICIT_IMPACT_PHYSICS = True
# If True, keeps restitution/friction neutral (used before explicit physics).
ENFORCE_IMPACT_INVARIANCE = False
# Restitution-coupled load amplification gain.
IMPACT_RESTITUTION_GAIN = 0.03
# Impact-velocity gain for dynamic traction amplification.
IMPACT_VELOCITY_GAIN = 0.03
