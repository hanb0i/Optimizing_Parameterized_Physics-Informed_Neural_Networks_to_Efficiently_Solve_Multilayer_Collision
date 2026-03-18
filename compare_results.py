
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), 'pinn-workflow'))
import pinn_config as config
import model
from scipy.interpolate import RegularGridInterpolator

def _u_from_v(v, pts):
    e_scale = 0.5 * (pts[:, 3:4] + pts[:, 4:5])
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    return scale * v / (e_scale ** e_pow)

def compare():
    print("Loading FEA Solution...")
    data = np.load("fea_solution.npy", allow_pickle=True).item()
    X_fea = data['x'] # (nx, ny, nz)
    Y_fea = data['y']
    Z_fea = data['z']
    U_fea = data['u'] # (nx, ny, nz, 3)
    
    # Grid axes
    x_axis = X_fea[:, 0, 0]
    y_axis = Y_fea[0, :, 0]
    z_axis = Z_fea[0, 0, :]
    
    print(f"FEA Grid: {len(x_axis)}x{len(y_axis)}x{len(z_axis)}")
    
    # 1. Generate PINN Predictions on same Grid
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pinn = model.MultiLayerPINN().to(device)
    env_model_path = os.environ.get("PINN_MODEL_PATH")
    if env_model_path:
        model_path = env_model_path
    else:
        candidates = []
        candidates.extend(glob("*.pth"))
        candidates.extend(glob(os.path.join(os.path.dirname(__file__), "pinn-workflow", "*.pth")))
        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            print("Model not found, cannot plot.")
            return
        model_path = max(candidates, key=os.path.getmtime)

    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
        sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
        pinn.load_state_dict(sd, strict=False)
        print(f"Loaded PINN model from {model_path}")
    except Exception as e:
        print(f"PINN model not found or error loading: {e}")
        return
        
    pinn.eval()
    
    pts = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1) # (N, 3)
    e1_ones = np.ones((pts.shape[0], 1)) * config.E_vals[0]
    e2_ones = np.ones((pts.shape[0], 1)) * config.E_vals[0]
    r_ones = np.ones((pts.shape[0], 1)) * float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ones = np.ones((pts.shape[0], 1)) * float(getattr(config, "FRICTION_REF", 0.3))
    v0_ones = np.ones((pts.shape[0], 1)) * float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.hstack([pts, e1_ones, e2_ones, r_ones, mu_ones, v0_ones])
    
    # Single layer - query all points at once
    U_pinn_flat = np.zeros((pts.shape[0], 3), dtype=pts.dtype)
    
    with torch.no_grad():
        pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
        v_pinn_flat = pinn(pts_tensor, 0).cpu().numpy()
        U_pinn_flat = _u_from_v(v_pinn_flat, pts)
            
    U_pinn = U_pinn_flat.reshape(U_fea.shape)
    
    # 2. Compute Metrics
    # U_z at top surface
    u_z_fea_top = U_fea[:, :, -1, 2]
    u_z_pinn_top = U_pinn[:, :, -1, 2]
    
    abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    denom = np.max(np.abs(u_z_fea_top))
    if denom > 0:
        mae_pct = (mae / denom) * 100.0
    else:
        mae_pct = 0.0
    
    print(f"Comparison Results (Top Surface u_z):")
    print(f"MAE: {mae:.6f}")
    print(f"MAE % of max |FEA u_z|: {mae_pct:.2f}%")
    print(f"Max Error: {max_err:.6f}")
    print(f"Peak Deflection FEA: {u_z_fea_top.min():.6f}")
    print(f"Peak Deflection PINN: {u_z_pinn_top.min():.6f}")
    
    # 3. Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Contours
    # FEA
    c1 = axes[0].contourf(X_fea[:,:,0], Y_fea[:,:,0], u_z_fea_top, levels=50, cmap='jet')
    axes[0].set_title("FEA Displacement u_z (Top)")
    plt.colorbar(c1, ax=axes[0])
    
    # PINN
    c2 = axes[1].contourf(X_fea[:,:,0], Y_fea[:,:,0], u_z_pinn_top, levels=50, cmap='jet')
    axes[1].set_title("PINN Displacement u_z (Top)")
    plt.colorbar(c2, ax=axes[1])
    
    # Error
    c3 = axes[2].contourf(X_fea[:,:,0], Y_fea[:,:,0], abs_diff, levels=50, cmap='magma')
    axes[2].set_title("Absolute Error |FEA - PINN|")
    plt.colorbar(c3, ax=axes[2])
    
    plt.savefig("comparison_top.png")
    print("Saved comparison_top.png")
    
    # Cross section
    # y index middle
    mid_y = U_fea.shape[1] // 2
    
    xz_X = X_fea[:, mid_y, :]
    xz_Z = Z_fea[:, mid_y, :]
    xz_Uz_fea = U_fea[:, mid_y, :, 2]
    xz_Uz_pinn = U_pinn[:, mid_y, :, 2]
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    c4 = axes2[0].contourf(xz_X, xz_Z, xz_Uz_fea, levels=50, cmap='jet')
    axes2[0].set_title("FEA Cross Section u_z")
    plt.colorbar(c4, ax=axes2[0])
    
    c5 = axes2[1].contourf(xz_X, xz_Z, xz_Uz_pinn, levels=50, cmap='jet')
    axes2[1].set_title("PINN Cross Section u_z")
    plt.colorbar(c5, ax=axes2[1])
    
    plt.savefig("comparison_cross_section.png")
    print("Saved comparison_cross_section.png")

if __name__ == "__main__":
    compare()
