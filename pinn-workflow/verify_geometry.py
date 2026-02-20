
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Global Font Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'Roboto', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 16
import scipy.interpolate

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import pinn_config as config
try:
    from model import MultiLayerPINN
except ImportError:
    from model import PINN as MultiLayerPINN # Fallback

def run_geometry_verification():
    print("=== Complex Geometry Verification (Parametric Dent) ===")
    
    # 1. Load Trained PINN
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerPINN().to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "pinn_model.pth"), map_location=device))
        print("Loaded PINN model.")
    except Exception as e:
        print(f"Failed to load PINN: {e}")
        return

    # 2. Generate Slice for Visualization (Y = Ly/2)
    # n_x = 100
    # n_z = 50
    x_vals = np.linspace(0, config.Lx, 100)
    # max_possible_z = config.H
    # But z varies. We grid z from [0, 1] relative depth and map to physical z.
    z_rel_vals = np.linspace(0, 1.0, 50)
    
    Y_fixed = config.Ly / 2.0
    
    X_grid, Z_rel_grid = np.meshgrid(x_vals, z_rel_vals)
    
    # Map to Physical Z
    # We need z_top(x, y_fixed)
    # Using config's python function might be tricky if it expects torch tensors.
    # Let's use torch for coordinate generation.
    
    X_torch = torch.tensor(X_grid, dtype=torch.float32).to(device)
    Y_torch = torch.full_like(X_torch, Y_fixed)
    Z_rel_torch = torch.tensor(Z_rel_grid, dtype=torch.float32).to(device)
    
    # Get Z-top surface
    Z_top_torch = config.get_domain_height(X_torch, Y_torch)
    Z_phys_torch = Z_rel_torch * Z_top_torch
    
    # Prepare Input
    # E_param is ignored in Phase 6 geometry mode, pass 0 or 1
    # We need to construct the full input vector: [x, y, z, E, t, r, mu, v0]
    # In Phase 6, E is derived from Z. We can just pass a placeholder and let the model/physics handle it?
    # Wait, the MODEL inputs (11 dims) are: x,y,z,E,t,r,mu,v0... 
    # The NETWORK learns the mapping. 
    # For visualization, we should pass the actual E that the physics enforced, 
    # OR if the network was trained with E as input, we need that.
    # IN PHASE 5/6: The physics calculates E from Z. The Network might still take E as input 
    # if we didn't change input dim. 
    # Let's check model.py input dim. It is 11.
    # During training, 'e' was passed as layer property.
    # So we need to reconstruct the layer property E based on Z_rel_torch.
    
    E_input = torch.zeros_like(X_torch)
    if hasattr(config, "LAYER_Z_RATIOS"):
        # Bot
        mask_bot = Z_rel_torch <= config.LAYER_Z_RATIOS[0]
        E_input[mask_bot] = config.LAYER_E_VALS[0]
        # Core
        mask_mid = (Z_rel_torch > config.LAYER_Z_RATIOS[0]) & (Z_rel_torch < config.LAYER_Z_RATIOS[1])
        E_input[mask_mid] = config.LAYER_E_VALS[1]
        # Top
        mask_top = Z_rel_torch >= config.LAYER_Z_RATIOS[1]
        E_input[mask_top] = config.LAYER_E_VALS[2]
        
    T_input = Z_top_torch # Thickness is local height
    
    # Other params (standard reference)
    R_input = torch.full_like(X_torch, 0.5)
    Mu_input = torch.full_like(X_torch, 0.3)
    V0_input = torch.full_like(X_torch, 1.0)
    
    # Flatten for batch processing
    inputs = torch.stack([
        X_torch.flatten(), Y_torch.flatten(), Z_phys_torch.flatten(),
        E_input.flatten(), T_input.flatten(),
        R_input.flatten(), Mu_input.flatten(), V0_input.flatten()
    ], dim=1)
    
    with torch.no_grad():
        v_pred = model(inputs, 0)
    u_pred = v_pred
        
    # Reshape
    U_z = u_pred[:, 2].reshape(X_grid.shape).cpu().numpy()
    Z_phys = Z_phys_torch.cpu().numpy()
    X_plot = X_torch.cpu().numpy()
    
    # 3. Visualize
    plt.figure(figsize=(10, 6))
    
    # Plot Displacement Contour
    # Use tricontourf if grid is irregular, but here columns are aligned (X constant), Z varies.
    # pcolormesh handles this well.
    plt.pcolormesh(X_plot, Z_phys, U_z, shading='gouraud', cmap='jet')
    plt.colorbar(label='Z-Displacement')
    
    # Plot Boundaries
    plt.plot(x_vals, Z_top_torch[0, :].cpu().numpy(), 'k-', linewidth=2, label='Top Surface (Dent)')
    plt.plot(x_vals, np.zeros_like(x_vals), 'k-', linewidth=2, label='Bottom Surface')
    
    # Plot Layer Interfaces
    if hasattr(config, "LAYER_Z_RATIOS"):
        z_i1 = config.LAYER_Z_RATIOS[0] * Z_top_torch[0, :].cpu().numpy()
        z_i2 = config.LAYER_Z_RATIOS[1] * Z_top_torch[0, :].cpu().numpy()
        plt.plot(x_vals, z_i1, 'w--', alpha=0.5, label='Interface 1')
        plt.plot(x_vals, z_i2, 'w--', alpha=0.5, label='Interface 2')

    plt.title(f"Cross-Section @ Y={Y_fixed:.2f} (Dented Plate)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis('equal') # Important to see true shape
    plt.legend()
    
    outfile = os.path.join(ROOT_DIR, "visualization", "geometry_verification.png")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    print(f"Visualization saved to {outfile}")

if __name__ == "__main__":
    run_geometry_verification()
