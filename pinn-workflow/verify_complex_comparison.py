
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT_DIR)
FEA_DIR = os.path.join(REPO_ROOT, "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")

sys.path.append(ROOT_DIR)
sys.path.append(FEA_SOLVER_DIR)

import pinn_config as config
try:
    from model import MultiLayerPINN
except ImportError:
    from model import PINN as MultiLayerPINN
    
# Import complex solver
# It's not in __init__.py presumably, so direct import
import fem_solver_complex

def run_complex_comparison():
    print("=== Complex Geometry Comparison: PINN vs Voxel-FEA ===")
    
    # 1. Setup Voxel Grid for FEA
    # Use config parameters, but maybe coarser mesh for speed if needed
    ne_x = 40
    ne_y = 40
    ne_z = 20 # Z-resolution needs to be decent to capture layers
    
    Lx, Ly, H = config.Lx, config.Ly, config.H
    
    # Material Definitions (Indices: 0=Bot, 1=Core, 2=Top)
    # Void is -1
    materials = [
        {'E': config.LAYER_E_VALS[0], 'nu': config.LAYER_NU_VALS[0]}, # Bot
        {'E': config.LAYER_E_VALS[1], 'nu': config.LAYER_NU_VALS[1]}, # Core
        {'E': config.LAYER_E_VALS[2], 'nu': config.LAYER_NU_VALS[2]}  # Top
    ]
    
    # Create Material Grid
    material_grid = np.full((ne_x, ne_y, ne_z), -1, dtype=int)
    
    dx = Lx / ne_x
    dy = Ly / ne_y
    dz = H / ne_z
    
    print(f"Generating Voxel Mesh ({ne_x}x{ne_y}x{ne_z})...")
    
    # Populate params for computing z_top
    # Re-implement dent logic here or import config?
    # Config has 'get_domain_height' but it expects torch tensors.
    # Let's implement numpy version locally for safety/speed.
    def get_z_top(x, y):
        # Center dims
        cx, cy = config.Lx/2, config.Ly/2
        r2 = (x - cx)**2 + (y - cy)**2
        dent = config.DENT_DEPTH * np.exp(-r2 / (2 * config.DENT_WIDTH**2))
        return config.H - dent

    # Iterate elements
    for k in range(ne_z):
        z_center = (k + 0.5) * dz
        for j in range(ne_y):
            y_center = (j + 0.5) * dy
            for i in range(ne_x):
                x_center = (i + 0.5) * dx
                
                z_limit = get_z_top(x_center, y_center)
                
                if z_center > z_limit:
                    material_grid[i,j,k] = -1 # Void
                else:
                    # Determine Layer
                    # Use Relative Z logic from config/physics
                    z_rel = z_center / z_limit
                    
                    if z_rel <= config.LAYER_Z_RATIOS[0]:
                        material_grid[i,j,k] = 0 # Bot
                    elif z_rel <= config.LAYER_Z_RATIOS[1]:
                        material_grid[i,j,k] = 1 # Core
                    else:
                        material_grid[i,j,k] = 2 # Top

    # 2. Run FEA
    fea_cfg = {
        'geometry': {'Lx': Lx, 'Ly': Ly, 'H': H},
        'mesh': {'ne_x': ne_x, 'ne_y': ne_y, 'ne_z': ne_z},
        'material': materials,
        'load_patch': {
            'x_start': config.LOAD_PATCH_X[0], 'x_end': config.LOAD_PATCH_X[1],
            'y_start': config.LOAD_PATCH_Y[0], 'y_end': config.LOAD_PATCH_Y[1],
            'pressure': config.p0
        },
        'use_soft_mask': True
    }
    
    try:
        x_nodes, y_nodes, z_nodes, u_fea_grid = fem_solver_complex.solve_fem_complex(fea_cfg, material_grid)
        print("FEA Solved.")
    except Exception as e:
        print(f"FEA Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Pinn Prediction on Same Grid
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerPINN().to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "pinn_model.pth"), map_location=device))
    
    # Construct Nodes for PINN query
    # x_nodes: (nx,), y_nodes: (ny,), z_nodes: (nz,)
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    
    # Flatten
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    Z_flat = Z_grid.flatten()
    
    # Convert to Torch
    t_X = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(1).to(device)
    t_Y = torch.tensor(Y_flat, dtype=torch.float32).unsqueeze(1).to(device)
    t_Z = torch.tensor(Z_flat, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Reconstruct E, t, etc inputs
    # Use torch-based z_top logic
    t_z_top = config.get_domain_height(t_X, t_Y)
    t_rel_z = t_Z / (t_z_top + 1e-8)
    
    # E logic
    t_E = torch.zeros_like(t_X)
    # Mask active
    active_mask = (t_Z <= t_z_top)
    
    # Assign E based on layers (same logic as physics.py)
    mask_bot = (t_rel_z <= config.LAYER_Z_RATIOS[0])
    mask_mid = (t_rel_z > config.LAYER_Z_RATIOS[0]) & (t_rel_z <= config.LAYER_Z_RATIOS[1])
    mask_top = (t_rel_z > config.LAYER_Z_RATIOS[1])
    
    t_E[mask_bot] = config.LAYER_E_VALS[0]
    t_E[mask_mid] = config.LAYER_E_VALS[1]
    t_E[mask_top] = config.LAYER_E_VALS[2]
    
    # Standard params
    t_T = t_z_top # Local thickness
    t_R = torch.full_like(t_X, 0.5)
    t_Mu = torch.full_like(t_X, 0.3)
    t_V0 = torch.full_like(t_X, 1.0)
    
    inputs = torch.cat([t_X, t_Y, t_Z, t_E, t_T, t_R, t_Mu, t_V0], dim=1)
    
    with torch.no_grad():
        v_pred = model(inputs, 0)
        u_pinn_flat = v_pred
        
    u_pinn_z = u_pinn_flat[:, 2].cpu().numpy().reshape(nx, ny, nz)
    
    # Apply invalid mask to PINN (where z > z_top)
    z_top_np = get_z_top(X_grid, Y_grid)
    invalid_mask = Z_grid > z_top_np
    u_pinn_z[invalid_mask] = np.nan
    
    # FEA Result is already grid. Mask void?
    # FEA u_grid is (nx, ny, nz, 3)
    # Note: FEA returns displacements for ALL nodes, but inactive nodes should be 0 (fixed).
    # We should mask them for visualization.
    u_fea_z = u_fea_grid[:, :, :, 2]
    u_fea_z[invalid_mask] = np.nan

    # 4. Visualize Cross-Section
    # Pick Y-slice
    y_idx = ny // 2
    y_val = y_nodes[y_idx]
    
    # X vs Z
    X_slice = X_grid[:, y_idx, :]
    Z_slice = Z_grid[:, y_idx, :]
    U_fea_slice = u_fea_z[:, y_idx, :]
    U_pinn_slice = u_pinn_z[:, y_idx, :]
    
    # Calculate Error
    # Only on valid pixels
    valid_slice = ~np.isnan(U_fea_slice)
    # Avoid div by zero
    abs_error = np.abs(U_fea_slice - U_pinn_slice)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # FEA
    c1 = axes[0].pcolormesh(X_slice, Z_slice, U_fea_slice, cmap='jet', shading='auto')
    axes[0].set_title(f"FEA (Voxel) - Z-Displacement @ Y={y_val:.2f}")
    axes[0].axis('equal')
    plt.colorbar(c1, ax=axes[0])
    
    # PINN
    c2 = axes[1].pcolormesh(X_slice, Z_slice, U_pinn_slice, cmap='jet', shading='auto')
    axes[1].set_title(f"PINN - Z-Displacement @ Y={y_val:.2f}")
    axes[1].axis('equal')
    plt.colorbar(c2, ax=axes[1])
    
    # Error
    c3 = axes[2].pcolormesh(X_slice, Z_slice, abs_error, cmap='inferno', shading='auto')
    axes[2].set_title("Absolute Error |FEA - PINN|")
    axes[2].axis('equal')
    plt.colorbar(c3, ax=axes[2])
    
    # Plot Surface Lines
    x_plot = x_nodes
    z_plot = get_z_top(x_plot, y_val)
    for ax in axes:
        ax.plot(x_plot, z_plot, 'k-', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

    plt.tight_layout()
    outfile = os.path.join(ROOT_DIR, "visualization", "complex_comparison.png")
    plt.savefig(outfile)
    print(f"Comparison saved to {outfile}")
    
    # Print metrics
    if np.any(valid_slice):
        mae = np.nanmean(abs_error)
        peak_fea = np.nanmin(U_fea_slice) # displacement is negative z
        peak_pinn = np.nanmin(U_pinn_slice)
        print("\n=== Quantitive Metrics ===")
        print(f"MAE: {mae:.6f}")
        print(f"Peak FEA Z:  {peak_fea:.6f}")
        print(f"Peak PINN Z: {peak_pinn:.6f}")
        print(f"Ratio (PINN/FEA): {peak_pinn/peak_fea:.4f}")

if __name__ == "__main__":
    run_complex_comparison()
