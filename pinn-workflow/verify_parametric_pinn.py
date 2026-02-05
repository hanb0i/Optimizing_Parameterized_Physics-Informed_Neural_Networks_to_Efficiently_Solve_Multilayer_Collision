
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver

def run_fea(E_val):
    print(f"Running FEA for E={E_val}...")
    # Mock config for FEA solver
    cfg = {
        'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': config.H},
        'material': {'E': E_val, 'nu': config.nu_vals[0]},
        'load_patch': {
            'pressure': config.p0,
            'x_start': config.LOAD_PATCH_X[0]/config.Lx,
            'x_end': config.LOAD_PATCH_X[1]/config.Lx,
            'y_start': config.LOAD_PATCH_Y[0]/config.Ly,
            'y_end': config.LOAD_PATCH_Y[1]/config.Ly
        }
    }
    x, y, z, u = fem_solver.solve_fem(cfg)
    return x, y, z, u

def main():
    E_test_values = [1.0, 5.0, 10.0]
    results = {}

    # Load PINN
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(REPO_ROOT, "pinn_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
        
    print(f"Loading model from: {model_path}")
    pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    pinn.eval()

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    # Row 0: FEA, Row 1: PINN, Row 2: Error
    # Cols: E=1, E=5, E=10

    for idx, E_val in enumerate(E_test_values):
        # 1. Run FEA
        x_nodes, y_nodes, z_nodes, u_fea_grid = run_fea(E_val)
        
        # Extract Top Surface for Visualization
        # u_fea_grid shape: (nx, ny, nz, 3)
        u_z_fea_top = u_fea_grid[:, :, -1, 2].T # Transpose for pcolormesh (y, x)
        
        # Meshgrid for plotting
        X, Y = np.meshgrid(x_nodes, y_nodes)
        
        # 2. Run PINN
        # Create grid points matching FEA nodes
        nx, ny = len(x_nodes), len(y_nodes)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = np.ones_like(X_flat) * config.H
        E_flat = np.ones_like(X_flat) * E_val
        
        # Prepare input (N, 4)
        input_pts = np.stack([X_flat, Y_flat, Z_flat, E_flat], axis=1)
        input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            v_pinn_flat = pinn(input_tensor).cpu().numpy()
            
        # Physics compliance scaling: u = v / E
        u_pinn_flat = v_pinn_flat / (E_val)
            
        u_z_pinn_top = u_pinn_flat[:, 2].reshape(ny, nx)
        
        # 3. Compute Error
        abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
        mae = np.mean(abs_diff)
        max_err = np.max(abs_diff)
        peak_fea = np.min(u_z_fea_top)
        peak_pinn = np.min(u_z_pinn_top)
        
        print(f"\n--- Results for E={E_val} ---")
        print(f"Peak Deflection FEA: {peak_fea:.6f}")
        print(f"Peak Deflection PINN: {peak_pinn:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Max Error: {max_err:.6f}")
        
        # 4. Plot
        # Row 0: FEA
        ax_fea = axes[0, idx]
        c1 = ax_fea.contourf(X, Y, u_z_fea_top, levels=50, cmap="jet")
        ax_fea.set_title(f"FEA (E={E_val})\nPeak: {peak_fea:.5f}")
        plt.colorbar(c1, ax=ax_fea)
        
        # Row 1: PINN
        ax_pinn = axes[1, idx]
        c2 = ax_pinn.contourf(X, Y, u_z_pinn_top, levels=50, cmap="jet")
        ax_pinn.set_title(f"PINN (E={E_val})\nPeak: {peak_pinn:.5f}")
        plt.colorbar(c2, ax=ax_pinn)
        
        # Row 2: Error
        ax_err = axes[2, idx]
        c3 = ax_err.contourf(X, Y, abs_diff, levels=50, cmap="magma")
        ax_err.set_title(f"Error (MAE: {mae:.5f})")
        plt.colorbar(c3, ax=ax_err)

    plt.tight_layout()
    result_path = os.path.join(REPO_ROOT, "parametric_verification.png")
    plt.savefig(result_path)
    print(f"\nVerification plot saved to: {result_path}")
    # plt.show()

if __name__ == "__main__":
    main()
