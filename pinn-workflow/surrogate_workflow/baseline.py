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
      - 3-layer: [E1, t1, E2, t2, E3, t3, restitution, friction, impact_velocity]
    Returns: Peak vertical displacement magnitude |Uz_max|
    """
    pinn, device = _get_pinn()
    
    if len(mu) == 5:
        E_val, t_val, r_val, mu_fric, v0_val = mu
        E1, E2, E3 = E_val, E_val, E_val
        t1 = t_val / 3.0
        t2 = t_val / 3.0
        t3 = t_val / 3.0
    elif len(mu) == 9:
        E1, t1, E2, t2, E3, t3, r_val, mu_fric, v0_val = mu
        t_val = float(t1) + float(t2) + float(t3)
        E_val = float(E3)
    else:
        raise ValueError(f"Expected mu length 5 or 9, got {len(mu)}")
    
    # Grid search for peak displacement on top surface
    nx = 11
    x = np.linspace(0.35, 0.65, nx)
    y = np.linspace(0.35, 0.65, nx)
    X, Y = np.meshgrid(x, y)
    Xf, Yf = X.flatten(), Y.flatten()
    
    Zf = np.ones_like(Xf) * t_val
    E1f = np.ones_like(Xf) * float(E1)
    E2f = np.ones_like(Xf) * float(E2)
    E3f = np.ones_like(Xf) * float(E3)
    t1f = np.ones_like(Xf) * float(t1)
    t2f = np.ones_like(Xf) * float(t2)
    t3f = np.ones_like(Xf) * float(t3)
    Rf = np.ones_like(Xf) * r_val
    MFf = np.ones_like(Xf) * mu_fric
    Vf = np.ones_like(Xf) * v0_val
    
    # Input layout: [x,y,z,E1,t1,E2,t2,E3,t3,r,mu,v0]
    pts = np.stack([Xf, Yf, Zf, E1f, t1f, E2f, t2f, E3f, t3f, Rf, MFf, Vf], axis=1)
    
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
    
    uz = v[:, 2]
    # Apply trained scaling parameters from pinn_config.py
    t_scale = (float(pc.H) / t_val) ** float(pc.THICKNESS_COMPLIANCE_ALPHA)
    u_final = (uz / (E_val ** float(pc.E_COMPLIANCE_POWER))) * t_scale
    
    return float(np.abs(np.min(u_final)))
