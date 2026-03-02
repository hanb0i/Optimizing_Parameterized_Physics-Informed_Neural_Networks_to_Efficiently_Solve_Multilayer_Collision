
import sys
import os
import torch
import numpy as np

# Add project paths
sys.path.append(os.path.join(os.getcwd(), 'pinn-workflow'))
sys.path.append(os.path.join(os.getcwd(), 'fea-workflow/solver'))
import fem_solver

def get_3layer_fea_truth():
    layers = [
        {'E': 10.0, 'thickness': 0.02}, # Bottom
        {'E': 1.0,  'thickness': 0.06}, # Middle
        {'E': 10.0, 'thickness': 0.02}  # Top
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
        'mesh': {'ne_x': 15, 'ne_y': 15, 'ne_z': 15} # Fast mesh for truth check
    }
    
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(fea_cfg)
    u_z = u_grid[:, :, :, 2]
    peak_defl = np.abs(np.min(u_z))
    print(f"FEA_GROUND_TRUTH: {peak_defl:.6f}")

if __name__ == "__main__":
    get_3layer_fea_truth()
