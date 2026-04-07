import os
import torch
import numpy as np

Lx = 1.0
Ly = 1.0
H = 0.1
NUM_LAYERS = 3
LAYER_THICKNESSES = [H / 3.0, H / 3.0, H / 3.0]
Layer_Interfaces = [0.0, LAYER_THICKNESSES[0], LAYER_THICKNESSES[0] + LAYER_THICKNESSES[1], H]

E_vals = [1.0]
nu_vals = [0.3]
E_RANGE = [1.0, 10.0]
T1_RANGE = [0.02, 0.10]
T2_RANGE = [0.02, 0.10]
T3_RANGE = [0.02, 0.10]
RESTITUTION_RANGE = [0.5, 0.5]
FRICTION_RANGE = [0.3, 0.3]
IMPACT_VELOCITY_RANGE = [1.0, 1.0]
PARAM_DIM = 9

RESTITUTION_REF = 0.5
FRICTION_REF = 0.3
IMPACT_VELOCITY_REF = 1.0

E_COMPLIANCE_POWER = 0.95
DISPLACEMENT_COMPLIANCE_SCALE = 1.0
THICKNESS_COMPLIANCE_ALPHA = 3.0

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

p0 = 1.0

FEM_NE_X = 10
FEM_NE_Y = 10
FEM_NE_Z = 4

PDE_LENGTH_SCALE = H

USE_HARD_SIDE_BC = True
HARD_BC_EPOCHS = 1000

LOAD_PATCH_X = [Lx/3, 2*Lx/3]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]

LAYERS = 4
NEURONS = 64
INTERFACE_FEATURE_BETA = 20.0

LEARNING_RATE = 1e-3
EPOCHS_SOAP = 400
EPOCHS_LBFGS = 0
SOAP_PRECONDITION_FREQUENCY = 10

WEIGHTS = {
    'pde': 10.0,
    'bc': 0.7,
    'load': 5.0,
    'energy': 0.63,
    'impact_invariance': 0.0,
    'impact_contact': 0.0002,
    'friction_coulomb': 0.001,
    'friction_stick': 0.0005,
    'interface_u': 300.0,
    'data': 400.0
}

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default

def _env_float_list(name: str, default):
    val = os.getenv(name)
    if val is None:
        return default
    out = []
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            return default
    return out if out else default

DISPLACEMENT_COMPLIANCE_SCALE = _env_float("PINN_DISPLACEMENT_COMPLIANCE_SCALE", DISPLACEMENT_COMPLIANCE_SCALE)
THICKNESS_COMPLIANCE_ALPHA = _env_float("PINN_THICKNESS_COMPLIANCE_ALPHA", THICKNESS_COMPLIANCE_ALPHA)
E_COMPLIANCE_POWER = _env_float("PINN_E_COMPLIANCE_POWER", E_COMPLIANCE_POWER)

for _k, _env in [
    ("pde", "PINN_W_PDE"),
    ("interface_u", "PINN_W_INTERFACE_U"),
    ("load", "PINN_W_LOAD"),
    ("bc", "PINN_W_BC"),
    ("data", "PINN_W_DATA"),
]:
    if _env in os.environ:
        WEIGHTS[_k] = _env_float(_env, float(WEIGHTS.get(_k, 0.0)))

PDE_DECOMPOSE_BY_LAYER = _env_flag("PINN_PDE_DECOMPOSE_BY_LAYER", True)

WEIGHT_RAMP_EPOCHS = 0
LOAD_WEIGHT_START = WEIGHTS['load']
PDE_WEIGHT_START = WEIGHTS['pde']
ENERGY_WEIGHT_START = WEIGHTS['energy']
FORCE_SOFT_SIDE_BC_FROM_START = True
SOFT_MODE_PDE_WEIGHT_SCALE = 3.0
SOFT_MODE_LOAD_WEIGHT_SCALE = 1.0

N_INTERIOR = 15000
N_SIDES = 2000
N_TOP_LOAD = 6000
N_TOP_FREE = 2000
N_BOTTOM = 2000
N_INTERFACE = _env_int("PINN_N_INTERFACE", 16000)
UNDER_PATCH_FRACTION = 0.95
INTERFACE_SAMPLE_FRACTION = _env_float("PINN_INTERFACE_SAMPLE_FRACTION", 0.75)
INTERFACE_BAND = 0.05 * H
PATCH_CENTER_BIAS_FRACTION = 0.8
PATCH_CENTER_BIAS_SHAPE = 3.0

SAMPLING_NOISE_SCALE = 0.08

LOAD_PATCH_UZ_TARGET = -0.05
LOAD_PATCH_UZ_WEIGHT = 0.02

FOURIER_DIM = 0
FOURIER_SCALE = 1.0

N_DATA_POINTS = _env_int("PINN_N_DATA_POINTS", 36000)
DATA_E_VALUES = [1.0, 10.0]
DATA_T1_VALUES = [0.02, 0.10]
DATA_T2_VALUES = [0.02, 0.10]
DATA_T3_VALUES = [0.02, 0.10]
EVAL_E_VALUES = [1.0, 10.0]
EVAL_T1_VALUES = [0.02, 0.10]
EVAL_T2_VALUES = [0.02, 0.10]
EVAL_T3_VALUES = [0.02, 0.10]
USE_SUPERVISION_DATA = True

SUPERVISION_THICKNESS_POWER = 3.0

DATA_E_VALUES = _env_float_list("PINN_DATA_E_VALUES", DATA_E_VALUES)
DATA_T1_VALUES = _env_float_list("PINN_DATA_T1_VALUES", DATA_T1_VALUES)
DATA_T2_VALUES = _env_float_list("PINN_DATA_T2_VALUES", DATA_T2_VALUES)
DATA_T3_VALUES = _env_float_list("PINN_DATA_T3_VALUES", DATA_T3_VALUES)

EVAL_E_VALUES = _env_float_list("PINN_EVAL_E_VALUES", EVAL_E_VALUES)
EVAL_T1_VALUES = _env_float_list("PINN_EVAL_T1_VALUES", EVAL_T1_VALUES)
EVAL_T2_VALUES = _env_float_list("PINN_EVAL_T2_VALUES", EVAL_T2_VALUES)
EVAL_T3_VALUES = _env_float_list("PINN_EVAL_T3_VALUES", EVAL_T3_VALUES)

FEM_NE_X = _env_int("PINN_FEM_NE_X", FEM_NE_X)
FEM_NE_Y = _env_int("PINN_FEM_NE_Y", FEM_NE_Y)
FEM_NE_Z = _env_int("PINN_FEM_NE_Z", FEM_NE_Z)

USE_EXPLICIT_IMPACT_PHYSICS = False
ENFORCE_IMPACT_INVARIANCE = False
IMPACT_RESTITUTION_GAIN = 0.03
IMPACT_VELOCITY_GAIN = 0.03
