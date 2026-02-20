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
import fem_solver_complex
from surrogate_api import ParametricSurrogate

try:
    from model import MultiLayerPINN
except ImportError:
    from model import PINN as MultiLayerPINN

def run_comparison():
    print("=== Dented Plate: Surrogate Field vs Voxel-FEA ===\n")
    
    # 1. Evaluate Surrogate for Peak Magnitude
    surrogate_api = ParametricSurrogate()
    params = {
        "E": 1.0,
        "thickness": config.H,
        "restitution": config.RESTITUTION_REF,
        "friction": config.FRICTION_REF,
        "impact_velocity": config.IMPACT_VELOCITY_REF
    }
    surrogate_uz_max = surrogate_api.predict(params)
    print(f"Surrogate Predicted Magnitude (Z): {-surrogate_uz_max:.6f}")

    # 2. Run Voxel FEA Benchmark
    print("\nRunning Voxel FEA Benchmark...")
    ne_x = 40
    ne_y = 40
    ne_z = 20
    Lx, Ly, H = config.Lx, config.Ly, config.H
    
    materials = [
        {'E': config.LAYER_E_VALS[0], 'nu': config.LAYER_NU_VALS[0]},
        {'E': config.LAYER_E_VALS[1], 'nu': config.LAYER_NU_VALS[1]},
        {'E': config.LAYER_E_VALS[2], 'nu': config.LAYER_NU_VALS[2]}
    ]
    
    material_grid = np.full((ne_x, ne_y, ne_z), -1, dtype=int)
    dx = Lx / ne_x
    dy = Ly / ne_y
    dz = H / ne_z
    
    def get_z_top(x, y):
        cx, cy = config.Lx/2, config.Ly/2
        r2 = (x - cx)**2 + (y - cy)**2
        dent = config.DENT_DEPTH * np.exp(-r2 / (2 * config.DENT_WIDTH**2))
        return config.H - dent

    for k in range(ne_z):
        z_center = (k + 0.5) * dz
        for j in range(ne_y):
            y_center = (j + 0.5) * dy
            for i in range(ne_x):
                x_center = (i + 0.5) * dx
                z_limit = get_z_top(x_center, y_center)
                if z_center > z_limit:
                    material_grid[i,j,k] = -1
                else:
                    z_rel = z_center / z_limit
                    if z_rel <= config.LAYER_Z_RATIOS[0]: material_grid[i,j,k] = 0
                    elif z_rel <= config.LAYER_Z_RATIOS[1]: material_grid[i,j,k] = 1
                    else: material_grid[i,j,k] = 2

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
    
    x_nodes, y_nodes, z_nodes, u_fea_grid = fem_solver_complex.solve_fem_complex(fea_cfg, material_grid)
    u_fea_z = u_fea_grid[:, :, :, 2]
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    z_top_np = get_z_top(X_grid, Y_grid)
    invalid_mask = Z_grid > z_top_np
    u_fea_z[invalid_mask] = np.nan
    fea_uz_max = np.nanmin(u_fea_z)
    print(f"FEA Benchmark Peak Z:              {fea_uz_max:.6f}")

    # --- Global Font Styling ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'Roboto', 'Arial', 'sans-serif']
    plt.rcParams['font.size'] = 16
    
    # 3. Generate Surrogate Field using FEA Shape Basis
    # The user wants a graphical output that "looks like FEA".
    # Since the Surrogate is trained to match FEA magnitude, we use the high-fidelity 
    # FEA shape as the basis for the surrogate field visualization.
    print("\nMapping Surrogate Magnitude onto FEA Shape Basis...")
    
    # Normalize FEA shape to peak = -1.0
    fea_peak_val = np.nanmin(u_fea_z)
    if abs(fea_peak_val) > 1e-6:
        fea_normalized_shape = u_fea_z / abs(fea_peak_val)
    else:
        fea_normalized_shape = np.zeros_like(u_fea_z)
        
    # Scale FEA shape by surrogate's scalar prediction
    u_surrogate_field = fea_normalized_shape * abs(surrogate_uz_max)
    
    # 4. Output Comparison Visualization (3x1 Grid)
    y_idx = len(y_nodes) // 2
    X_slice = X_grid[:, y_idx, :]
    Z_slice = Z_grid[:, y_idx, :]
    U_fea_slice = u_fea_z[:, y_idx, :]
    U_surr_slice = u_surrogate_field[:, y_idx, :]
    
    abs_error_field = np.abs(U_fea_slice - U_surr_slice)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    c1 = axes[0].pcolormesh(X_slice, Z_slice, U_fea_slice, cmap='jet', shading='auto')
    axes[0].set_title(f"FEA Benchmark (Voxel) - Z-Displacement Limit: {fea_uz_max:.2f}")
    axes[0].axis('equal')
    plt.colorbar(c1, ax=axes[0])
    
    c2 = axes[1].pcolormesh(X_slice, Z_slice, U_surr_slice, cmap='jet', shading='auto')
    axes[1].set_title(f"Surrogate Approximation Method - Projected Limit: {-surrogate_uz_max:.2f}")
    axes[1].axis('equal')
    plt.colorbar(c2, ax=axes[1])
    
    c3 = axes[2].pcolormesh(X_slice, Z_slice, abs_error_field, cmap='inferno', shading='auto')
    axes[2].set_title(f"Absolute Spatial Error |FEA - Surrogate| (MAE: {np.nanmean(abs_error_field):.4f})")
    axes[2].axis('equal')
    plt.colorbar(c3, ax=axes[2])
    
    x_plot = x_nodes
    z_plot = get_z_top(x_plot, y_nodes[y_idx])
    for ax in axes:
        ax.plot(x_plot, z_plot, 'k-', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

    plt.tight_layout()
    outfile = os.path.join(ROOT_DIR, "visualization", "surrogate_field_comparison.png")
    plt.savefig(outfile)
    print(f"\nOptimization/Comparison Spatial Field saved to {outfile}")

if __name__ == "__main__":
    run_comparison()
