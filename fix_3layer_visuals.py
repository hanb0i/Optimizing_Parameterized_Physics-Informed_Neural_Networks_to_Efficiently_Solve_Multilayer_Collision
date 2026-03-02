
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "pinn-workflow"))
sys.path.append(os.path.join(ROOT_DIR, "fea-workflow/solver"))

import fem_solver
from cascaded_surrogate import CascadedSandwichSolver

def visualize_3layer_field(case_name, layers, p_impact=1.0):
    print(f"Generating Field Visualization for: {case_name}")
    
    solver = CascadedSandwichSolver()
    
    # 1. PINN Cascaded Field Reconstruction
    # To reconstruct the field, we solve the 3-layer system and get per-layer intermediate loads.
    total_disp, results = solver.solve_3_layer(layers, p_impact=p_impact)
    
    nx, nz_per_layer = 50, 20
    x_plot = np.linspace(0, 1.0, nx)
    
    # Create a full 3rd-layer field (XZ slice)
    full_field_surr = []
    z_coords_full = []
    
    # Inverse the results to stack Bottom -> Top for plotting
    for i in reversed(range(len(results))):
        res = results[i]
        l = layers[i]
        E, t = l['E'], l['t']
        p_in = res['p_top']
        
        # Calculate current base Z for this layer
        # If i=2 (Bot): base=0
        # If i=1 (Mid): base=t_bot
        # If i=0 (Top): base=t_bot + t_mid
        base_z = sum(layers[j]['t'] for j in range(i+1, 3))
        
        # Query PINN for this layer's slice
        z_layer = np.linspace(0, t, nz_per_layer)
        X, Z = np.meshgrid(x_plot, z_layer, indexing='ij')
        
        pts = np.zeros((X.size, 8))
        pts[:, 0] = X.flatten()
        pts[:,1] = 0.5 
        pts[:, 2] = Z.flatten()
        pts[:, 3] = E
        pts[:, 4] = t
        pts[:, 5] = 0.5 # rest
        pts[:, 6] = 0.3 # fric
        pts[:, 7] = 1.0 # vel
        
        with torch.no_grad():
            v = solver.pinn(torch.tensor(pts, dtype=torch.float32).to(solver.device)).cpu().numpy()
            
        uz_layer = v[:, 2].reshape(nx, nz_per_layer).T # (nz, nx)
        uz_scaled = uz_layer * (p_in / 1.0)
        
        # To avoid overlaps in Z, we skip the first point of non-bottom layers
        if i < len(results)-1: # Not the bottom layer
             full_field_surr.append(uz_scaled[1:])
             z_coords_full.append(z_layer[1:] + base_z)
        else: # Bottom layer
             full_field_surr.append(uz_scaled)
             z_coords_full.append(z_layer + base_z)

    # Since we built from i=0 (Top) to i=2 (Bottom), but we usually plot Bottom at z=0.
    # layers[0] is Top, layers[1] is Mid, layers[2] is Bot in solver loop.
    # Let's reverse for stacking.
    
    full_z = np.concatenate(z_coords_full)
    full_u = np.concatenate(full_field_surr) # This is (nz_total, nx)
    
    # 2. FEA Ground Truth
    total_thickness = sum(l['t'] for l in layers)
    fea_cfg = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': total_thickness},
        'material': [{'E': l['E'], 'nu': 0.3} for l in layers], 
        'load_patch': {
            'x_start': 0.333, 'x_end': 0.667, 'y_start': 0.333, 'y_end': 0.667,
            'pressure': p_impact
        },
        'use_soft_mask': True,
        'mesh': {'ne_x': 30, 'ne_y': 30, 'ne_z': 30}
    }
    _, _, _, u_grid = fem_solver.solve_fem(fea_cfg)
    uz_fea_xz = u_grid[:, 15, :, 2].T # (nz, nx)
    fea_peak = np.abs(np.min(uz_fea_xz))

    # 3. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # FEA
    im1 = axes[0].imshow(uz_fea_xz, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='jet', aspect='auto')
    axes[0].set_title(f"FEA\nPeak: {fea_peak:.4f}")
    plt.colorbar(im1, ax=axes[0])
    
    # PINN Cascaded
    im2 = axes[1].imshow(full_u, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='jet', aspect='auto')
    axes[1].set_title(f"PINN Cascaded\nPeak: {total_disp:.4f}")
    plt.colorbar(im2, ax=axes[1])
    
    # Abs Diff (interpolated)
    from scipy.interpolate import RegularGridInterpolator
    z_pinn = np.concatenate(z_coords_full)
    x_pinn = np.linspace(0, 1, nx)
    interp = RegularGridInterpolator((z_pinn, x_pinn), full_u, bounds_error=False, fill_value=0)
    
    z_fea = np.linspace(0, total_thickness, 31)
    x_fea = np.linspace(0, 1, 31)
    Zf, Xf = np.meshgrid(z_fea, x_fea, indexing='ij')
    u_pinn_interp = interp(np.stack([Zf.flatten(), Xf.flatten()], axis=1)).reshape(31, 31)
    
    diff = np.abs(uz_fea_xz - u_pinn_interp)
    im3 = axes[2].imshow(diff, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='magma', aspect='auto')
    axes[2].set_title(f"Abs Error\nMax: {np.max(diff):.4f}")
    plt.colorbar(im3, ax=axes[2])
    
    # Z-Profile
    center_idx_pinn = nx // 2
    center_idx_fea = 15
    axes[3].plot(uz_fea_xz[:, center_idx_fea], z_fea, 'b-', label='FEA')
    axes[3].plot(full_u[:, center_idx_pinn], z_pinn, 'r--', label='PINN')
    axes[3].set_title("Z-Profile at Center")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    out_name = f"3_layer_field_{case_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join("pinn-workflow/visualization", out_name))
    print(f"Saved visualization to pinn-workflow/visualization/{out_name}")

if __name__ == "__main__":
    # Test with Case 1 (Standard Sandwich)
    layers_case1 = [
        {'E': 10.0, 't': 0.02}, # Top Face
        {'E': 1.0,  't': 0.06}, # Core
        {'E': 10.0, 't': 0.02}  # Bot Face
    ]
    visualize_3layer_field("Case 1 Standard", layers_case1)
    
    # Test with Case 3 (Thickness Gradient)
    layers_case3 = [
        {'E': 10.0, 't': 0.01},
        {'E': 5.0,  't': 0.04},
        {'E': 2.0,  't': 0.05}
    ]
    visualize_3layer_field("Case 3 Gradient", layers_case3)
