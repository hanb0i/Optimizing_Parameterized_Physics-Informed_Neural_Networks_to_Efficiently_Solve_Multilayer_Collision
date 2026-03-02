import os
import sys
import numpy as np
import torch

from surrogate_workflow import config

# Add parent directory to path to import model and pinn_config
PINN_DIR = os.path.dirname(os.path.abspath(__file__)) # surrogate_workflow dir
BASE_DIR = os.path.dirname(PINN_DIR) # pinn-workflow dir

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import model
import pinn_config as pc
import physics

# Cache for the loaded PINN
_PINN_MODEL = None
_DEVICE = torch.device("cpu")

def _get_pinn():
    global _PINN_MODEL, _DEVICE
    if _PINN_MODEL is not None:
        return _PINN_MODEL, _DEVICE
    
    _DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _PINN_MODEL = model.MultiLayerPINN().to(_DEVICE)
    model_path = os.path.join(BASE_DIR, "pinn_model.pth")
    if os.path.exists(model_path):
        _PINN_MODEL.load_state_dict(torch.load(model_path, map_location=_DEVICE, weights_only=False), strict=False)
        print(f"Loaded PINN for surrogate from {model_path}")
    else:
        print(f"Warning: PINN model not found at {model_path}")
    _PINN_MODEL.eval()
    return _PINN_MODEL, _DEVICE

def compute_response(mu):
    """
    mu:
      - legacy: [E, thickness, restitution, friction, impact_velocity]
      - 2-layer: [E1, t1, E2, t2, restitution, friction, impact_velocity]
      - 3-layer: [E1, t1, E2, t2, E3, t3, restitution, friction, impact_velocity]
    Returns: Peak vertical displacement magnitude |Uz_max|
    """
    pinn, device = _get_pinn()

    L = int(getattr(pc, "NUM_LAYERS", 2))
    if len(mu) == 5:
        E_val, t_val, r_val, mu_fric, v0_val = mu
        E_list = [float(E_val)] * L
        t_list = [float(t_val) / float(L)] * L
    elif L == 2 and len(mu) == 7:
        E1, t1, E2, t2, r_val, mu_fric, v0_val = mu
        E_list = [float(E1), float(E2)]
        t_list = [float(t1), float(t2)]
    elif L == 3 and len(mu) == 9:
        E1, t1, E2, t2, E3, t3, r_val, mu_fric, v0_val = mu
        E_list = [float(E1), float(E2), float(E3)]
        t_list = [float(t1), float(t2), float(t3)]
    else:
        raise ValueError(f"Expected mu length 5, {(7 if L==2 else 9)}; got {len(mu)} (NUM_LAYERS={L})")

    t_val = float(sum(t_list))
    
    # Grid search for peak displacement on top surface
    nx = 11
    x = np.linspace(0.35, 0.65, nx)
    y = np.linspace(0.35, 0.65, nx)
    X, Y = np.meshgrid(x, y)
    Xf, Yf = X.flatten(), Y.flatten()
    
    Zf = np.ones_like(Xf) * float(t_val)
    Rf = np.ones_like(Xf) * r_val
    MFf = np.ones_like(Xf) * mu_fric
    Vf = np.ones_like(Xf) * v0_val
    
    cols = [Xf, Yf, Zf]
    for Ei, ti in zip(E_list, t_list):
        cols.append(np.ones_like(Xf) * float(Ei))
        cols.append(np.ones_like(Xf) * float(ti))
    cols.extend([Rf, MFf, Vf])
    pts = np.stack(cols, axis=1).astype(np.float32, copy=False)
    
    with torch.no_grad():
        x_t = torch.tensor(pts, dtype=torch.float32).to(device)
        v = pinn(x_t)
        u = physics.decode_u(v, x_t).cpu().numpy()
    
    uz = u[:, 2]
    return float(np.abs(np.min(uz)))
