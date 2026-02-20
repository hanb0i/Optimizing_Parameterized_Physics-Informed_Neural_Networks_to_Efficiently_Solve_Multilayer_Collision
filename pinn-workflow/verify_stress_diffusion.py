import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Global Font Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'Roboto', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 16

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT_DIR)
FEA_DIR = os.path.join(REPO_ROOT, "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")

sys.path.append(ROOT_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(FEA_DIR)
sys.path.append(FEA_SOLVER_DIR)

from surrogate_api import ParametricSurrogate
import fem_solver
import pinn_config as pc

def solve_stress_diffusion(layers, p0, load_patch_x, load_patch_y):
    """
    Calculates the displacement of each layer using a stress-diffusion model.
    
    Args:
        layers: List of dicts with 'E', 'thickness', etc.
        p0: Initial pressure on top surface.
        load_patch_x: (width) of load patch.
        load_patch_y: (length) of load patch.
        
    Returns:
        total_disp: Sum of layer displacements.
        layer_results: List of dicts with details per layer.
    """
    ps = ParametricSurrogate()
    
    current_p = p0
    current_lx = load_patch_x
    current_ly = load_patch_y
    
    total_disp = 0.0
    layer_results = []
    
    print(f"\n--- Stress Diffusion Calculation (P0={p0:.2f}) ---")
    
    # Process from Top to Bottom
    # Note: layers list is usually Bottom -> Top in our config.
    # We need to iterate Top -> Bottom for stress flow.
    
    ordered_layers = layers[::-1] # Reverse to get Top first
    
    for i, layer in enumerate(ordered_layers):
        # 1. Prediction for this layer
        # The surrogate predicts displacement for a standard patch P=1.0.
        # We assume Linear Elasticity: u_actual = u_ref * (P_actual / P_ref)
        
        # Surrogate Params
        params = {
            "E": layer['E'],
            "thickness": layer['thickness'],
            "restitution": 0.5, # Neutral
            "friction": 0.3,    # Neutral
            "impact_velocity": 1.0 # Neutral
        }
        
        # Get Reference Displacement (at P=1.0, Load Lx=0.333)
        # Note: The surrogate was trained on a FIXED patch size (1/3 of domain).
        # Our diffused patch is LARGER.
        # Plate theory: u ~ P * A / D? or u ~ P * L^4 / D?
        # For a fixed pressure P, if patch grows -> Total Force grows -> u grows?
        # OR: u ~ Force / Stiffness. Force = P * Area.
        # If P decreases as Area increases (Force constant), what happens to u?
        # Boussinesq: u ~ Force / r.
        # If Force is constant, but spread over larger r, u should decrease.
        
        # Let's stick to the simplest "Series Spring" first:
        # Layer stiffness k = A * E / t ? No, bending/plate mode.
        # We simply scale the surrogate output linearly by pressure.
        # We ignore the "Shape Effect" of the widening patch for now (or approximate it).
        
        u_ref = ps.predict(params)
        
        # Scale by Pressure Ratio
        u_layer = u_ref * (current_p / 1.0)
        
        # Correction for Diffused Patch Size?
        # If the patch is wider, the plate is "stiffened" less? Or load is more distributed?
        # Distributed load causes LESS peak deflection than concentrated load.
        # Factor ~ 1 / sqrt(Area)?
        # Let's assume u_layer is correct for the mitigated pressure.
        
        total_disp += u_layer
        
        print(f"Layer {i} ({layer['name']}):")
        print(f"  Input Pressure: {current_p:.4f}")
        print(f"  Patch Size: {current_lx:.4f} x {current_ly:.4f}")
        print(f"  Disp (Contribution): {u_layer:.6f}")
        
        layer_results.append({
            'name': layer['name'],
            'p_in': current_p,
            'disp': u_layer,
            'z_start': sum(l['thickness'] for l in layers) - sum(l['thickness'] for l in ordered_layers[:i]),
            'thickness': layer['thickness']
        })
        
        # 2. Diffuse Stress for NEXT layer
        # Spread angle (alpha). 45 degrees -> slope 1.
        # dx = 2 * t * tan(alpha).
        diffusion_slope = 1.0 
        
        dx = 2 * layer['thickness'] * diffusion_slope
        dy = 2 * layer['thickness'] * diffusion_slope
        
        next_lx = current_lx + dx
        next_ly = current_ly + dy
        
        # Conservation of Force: F = P * Area = Constant
        force = current_p * (current_lx * current_ly)
        next_area = next_lx * next_ly
        current_p = force / next_area
        
        current_lx = next_lx
        current_ly = next_ly
        
    return total_disp, layer_results

def run_stress_diffusion_check():
    print("=== Stress-Diffusion Series Model Verification ===")
    
    # 1. Define Layers (Bottom -> Top)
    layers = [
        {'name': 'Bottom', 'E': 10.0, 'thickness': 0.1, 'nu': 0.3},
        {'name': 'Middle', 'E': 1.0,  'thickness': 0.1, 'nu': 0.3},
        {'name': 'Top',    'E': 5.0,  'thickness': 0.1, 'nu': 0.3}
    ]
    
    total_thickness = sum(l['thickness'] for l in layers)
    
    # 2. Run FEA Ground Truth
    print("\nRunning FEA Ground Truth...")
    fea_cfg = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': total_thickness},
        'material': [{'E': l['E'], 'nu': l['nu']} for l in layers], 
        'load_patch': {
            'x_start': 0.333, 'x_end': 0.667,
            'y_start': 0.333, 'y_end': 0.667,
            'pressure': 1.0
        },
        'use_soft_mask': True,
        'mesh': {'ne_x': 30, 'ne_y': 30, 'ne_z': 30}
    }
    
    _, _, z_nodes, u_grid = fem_solver.solve_fem(fea_cfg)
    uz_top = u_grid[:, :, -1, 2] 
    fea_peak = np.abs(np.min(uz_top))
    print(f"FEA Peak Displacement: {fea_peak:.6f}")
    
    # 3. Run Stress Diffusion Model
    patch_w = 0.667 - 0.333
    pred_disp, details = solve_stress_diffusion(layers, 1.0, patch_w, patch_w)
    
    print(f"\nTotal Predicted Displacement (Series Sum): {pred_disp:.6f}")
    
    error = abs(pred_disp - fea_peak)
    rel_error = (error / fea_peak) * 100
    print(f"Error: {error:.6f} ({rel_error:.2f}%)")
    
    # 4. Visualization (Step-wise Comparison)
    plt.figure(figsize=(10, 6))
    
    # FEA Profile (Centerline)
    mid_y = u_grid.shape[1] // 2
    mid_x = u_grid.shape[0] // 2
    u_fea_line = u_grid[mid_x, mid_y, :, 2]
    z_line = np.linspace(0, total_thickness, len(u_fea_line))
    
    plt.plot(u_fea_line, z_line, 'b-', linewidth=3, label='FEA (Ground Truth)')
    
    # Diffusion Model (Step-wise)
    # We know the compression of each layer.
    # We construct the displacement profile from Bottom (0) up.
    
    z_curr = 0.0
    u_curr = 0.0 # Fixed bottom
    
    z_pts = [0.0]
    u_pts = [0.0]
    
    # Iterate Bottom -> Top for plotting integration
    for layer in layers:
        # Find the result for this layer
        res = next(r for r in details if r['name'] == layer['name'])
        
        # Linear interpolation of compression within layer
        u_next = u_curr - res['disp'] # Downward displacement
        
        z_next = z_curr + layer['thickness']
        
        z_pts.append(z_next)
        u_pts.append(u_next)
        
        # Draw layer patch
        color = 'gray'
        if layer['name'] == 'Middle': color = 'silver'
        if layer['name'] == 'Top': color = 'whitesmoke'
        
        plt.axhspan(z_curr, z_next, color=color, alpha=0.2)
        plt.text(np.min(u_fea_line)*0.8, z_curr + 0.02, 
                 f"{layer['name']} (P={res['p_in']:.2f})", fontsize=9)
        
        z_curr = z_next
        u_curr = u_next
        
    plt.plot(u_pts, z_pts, 'r--o', linewidth=2, label='Stress-Diffusion Model')
    
    plt.title(f"Stress-Diffusion Series Verification\nFEA: {fea_peak:.4f} | Diffusion: {pred_disp:.4f} (Err: {rel_error:.1f}%)")
    plt.xlabel("Uz Displacement")
    plt.ylabel("Z Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(ROOT_DIR, "visualization", "stress_diffusion_check.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    run_stress_diffusion_check()
