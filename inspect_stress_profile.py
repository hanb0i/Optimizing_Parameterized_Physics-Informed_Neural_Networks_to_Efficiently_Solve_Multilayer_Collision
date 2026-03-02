
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project paths
sys.path.append(os.path.join(os.getcwd(), 'pinn-workflow'))
sys.path.append(os.path.join(os.getcwd(), 'fea-workflow/solver'))
import fem_solver

def inspect_stress_profile():
    layers = [
        {'E': 10.0, 'thickness': 0.02}, # Top Face
        {'E': 1.0,  'thickness': 0.06}, # Core
        {'E': 10.0, 'thickness': 0.02}  # Bot Face
    ]
    
    total_thickness = sum(l['thickness'] for l in layers)
    
    fea_cfg = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': total_thickness},
        'material': [{'E': l['E'], 'nu': 0.3} for l in layers], 
        'load_patch': {
            'x_start': 0.333, 'x_end': 0.667,
            'y_start': 0.333, 'y_end': 0.667,
            'pressure': 1.0
        },
        'use_soft_mask': True,
        'mesh': {'ne_x': 20, 'ne_y': 20, 'ne_z': 50} # High Z-res for profile
    }
    
    print("Running FEA for stress profile...")
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(fea_cfg)
    
    # Extract sigma_zz along the center-line (0.5, 0.5, z)
    # fem_solver.solve_fem returns u_grid. We need to compute stress.
    # Actually, let's look at u_z first. 
    # But for "Force Transfer", sigma_zz is the key.
    
    # Helper to get mesh indices for center
    ne_x, ne_y, ne_z = 20, 20, 50
    ix, iy = ne_x // 2, ne_y // 2
    
    # We can use the strain calculation from physics (adapted for numpy)
    # delta_z = H / ne_z
    # sigma_zz = E * duz/dz (simplified for profile check)
    
    u_z_profile = u_grid[ix, iy, :, 2]
    z_coords = np.linspace(0, total_thickness, ne_z + 1)
    
    # duz/dz
    duz_dz = np.diff(u_z_profile) / np.diff(z_coords)
    
    # sigma_zz profile per layer
    sigma_zz = []
    current_z = 0.0
    for l in layers:
        t = l['thickness']
        E = l['E']
        # Find indices in this layer
        mask = (z_coords[:-1] >= current_z) & (z_coords[:-1] < current_z + t)
        sigma_zz.extend(E * duz_dz[mask])
        current_z += t
        
    plt.figure(figsize=(8, 6))
    plt.plot(z_coords[:-1], np.abs(sigma_zz), 'o-', label='|sigma_zz| (Pressure)')
    plt.xlabel('Z-Coordinate (Height)')
    plt.ylabel('Effective Pressure')
    plt.title('Pressure Transfer through 3-Layer Sandwich (FEA)')
    plt.axvline(0.02, color='r', linestyle='--', label='Interface 1')
    plt.axvline(0.08, color='g', linestyle='--', label='Interface 2')
    plt.grid(True)
    plt.legend()
    plt.savefig('stress_profile_fea.png')
    
    # Print interface pressures
    # Sigma at Z=0.1 (Top) should be ~1.0
    # Sigma at Z=0.08 (Face1-Core)
    # Sigma at Z=0.02 (Core-Face2)
    # Sigma at Z=0.0 (Bottom)
    
    print(f"Top Surface Pressure (Z=0.1): {abs(sigma_zz[-1]):.4f}")
    # Interface core-top
    idx1 = int(0.08 / (total_thickness / ne_z))
    print(f"Interface 1 Pressure (Z=0.08): {abs(sigma_zz[idx1]):.4f}")
    # Interface bottom-core
    idx2 = int(0.02 / (total_thickness / ne_z))
    print(f"Interface 2 Pressure (Z=0.02): {abs(sigma_zz[idx2]):.4f}")
    print(f"Bottom Surface Pressure (Z=0.0): {abs(sigma_zz[0]):.4f}")

if __name__ == "__main__":
    inspect_stress_profile()
