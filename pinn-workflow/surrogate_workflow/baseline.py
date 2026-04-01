import os
import sys

import numpy as np
import torch

# Add pinn-workflow to path to import model and pinn_config when run as a script.
SURROGATE_DIR = os.path.dirname(os.path.abspath(__file__))
PINN_DIR = os.path.dirname(SURROGATE_DIR)
if PINN_DIR not in sys.path:
    sys.path.insert(0, PINN_DIR)

import model  # noqa: E402
import pinn_config as pc  # noqa: E402


_PINN_MODEL = None
_DEVICE = None


def _u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    e_scale = 0.5 * (pts[:, 3:4] + pts[:, 5:6])
    t_scale = pts[:, 4:5] + pts[:, 6:7]
    e_pow = float(getattr(pc, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(pc, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(pc, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(pc, "H", 1.0))
    return scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _get_device() -> torch.device:
    requested = os.getenv("SURROGATE_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(pinn: torch.nn.Module, device: torch.device) -> None:
    override = os.getenv("PINN_MODEL_PATH")
    model_path = override or os.path.join(PINN_DIR, "pinn_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PINN checkpoint not found at {model_path}")
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(model_path, map_location=device)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)


def _get_pinn():
    global _PINN_MODEL, _DEVICE
    if _PINN_MODEL is not None:
        return _PINN_MODEL, _DEVICE
    _DEVICE = _get_device()
    pinn = model.MultiLayerPINN().to(_DEVICE)
    _load_checkpoint(pinn, _DEVICE)
    pinn.eval()
    _PINN_MODEL = pinn
    return _PINN_MODEL, _DEVICE


def compute_response(mu: np.ndarray) -> float:
    """
    mu: [E1, t1, E2, t2]
    Returns: peak downward vertical displacement on the top surface (positive scalar).
    """
    pinn, device = _get_pinn()
    e1_val, t1_val, e2_val, t2_val = [float(v) for v in mu]
    thickness = float(t1_val) + float(t2_val)

    nx = int(os.getenv("SURROGATE_TOP_NX", "11"))
    x = np.linspace(float(pc.LOAD_PATCH_X[0]), float(pc.LOAD_PATCH_X[1]), nx)
    y = np.linspace(float(pc.LOAD_PATCH_Y[0]), float(pc.LOAD_PATCH_Y[1]), nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = np.full_like(x_flat, thickness, dtype=float)

    r_ref = float(getattr(pc, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(pc, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(pc, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack(
        [
            x_flat,
            y_flat,
            z_flat,
            np.full_like(x_flat, e1_val),
            np.full_like(x_flat, t1_val),
            np.full_like(x_flat, e2_val),
            np.full_like(x_flat, t2_val),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )

    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
    u = _u_from_v(v, pts)
    uz = u[:, 2]
    return float(-np.min(uz))

