import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# pinn-workflow is ROOT_DIR
REPO_ROOT = os.path.dirname(ROOT_DIR)
FEA_DIR = os.path.join(REPO_ROOT, "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")

sys.path.append(ROOT_DIR)
sys.path.append(REPO_ROOT) # For direct fea-workflow imports if needed
sys.path.append(FEA_DIR)
sys.path.append(FEA_SOLVER_DIR)

from surrogate_api import ParametricSurrogate
import fem_solver

def calculate_composite_stiffness(layers):
    """
    Calculates bending stiffness (EI) per unit width for a composite plate.
    Assumes perfect bonding between layers.
    """
    # 1. Calculate neutral axis position (z_bar)
    # z_bar = sum(Ei * Ai * yi) / sum(Ei * Ai)
    # For unit width, Ai = ti * 1.0, yi is center of layer i
    
    numerator = 0.0
    denominator = 0.0
    current_z = 0.0
    
    for l in layers:
        t = l['thickness']
        E = l['E']
        y_center = current_z + t/2.0
        
        numerator += E * t * y_center
        denominator += E * t
        
        current_z += t
        
    z_bar = numerator / denominator
    
    # 2. Calculate Total EI about neutral axis
    # EI = sum( Ei * (Ii + Ai * di^2) )
    EI_total = 0.0
    current_z = 0.0
    
    for l in layers:
        t = l['thickness']
        E = l['E']
        y_center = current_z + t/2.0
        
        I_local = (1.0 * t**3) / 12.0
        d = y_center - z_bar
        
        EI_total += E * (I_local + t * d**2)
        
        current_z += t
        
    return EI_total

def run_3layer_verification():
    print("=== 3-Layer Composite Verification (Bonded) ===")
    
    # 1. Define Layer Properties (Bottom -> Top)
    # Using distinct materials to test composite logic
    layers = [
        {'name': 'Bottom', 'E': 10.0, 'thickness': 0.1, 'nu': 0.3},
        {'name': 'Middle', 'E': 1.0,  'thickness': 0.1, 'nu': 0.3},
        {'name': 'Top',    'E': 5.0,  'thickness': 0.1, 'nu': 0.3}
    ]
    
    total_thickness = sum(l['thickness'] for l in layers)
    
    # 2. Run FEA Ground Truth
    print("\nRunning FEA on 3-Layer Stack...")
    fea_cfg = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': total_thickness},
        'material': [{'E': l['E'], 'nu': l['nu']} for l in layers], 
        'load_patch': {
            'x_start': 0.333, 'x_end': 0.667,
            'y_start': 0.333, 'y_end': 0.667,
            'pressure': 1.0
        },
        'use_soft_mask': True,
        'mesh': {'ne_x': 30, 'ne_y': 30, 'ne_z': 30} # Explicit mesh control
    }
    
    # Solve
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(fea_cfg)
    
    # Extract peak displacement at top surface center
    # Top surface is index -1 in z
    uz_top = u_grid[:, :, -1, 2] 
    fea_peak_disp = np.abs(np.min(uz_top))
    print(f"FEA Peak Displacement: {fea_peak_disp:.6f}")
    
    # 3. Composite Beam Theory Prediction
    print("\nCalculating Composite Stiffness...")
    EI_actual = calculate_composite_stiffness(layers)
    print(f"Total EI (Composite): {EI_actual:.6f}")
    
    # 4. Use Surrogate for a Reference Case
    # We query the surrogate for a "Reference Layer" that is within distribution.
    # Ref: E=5.0, t=0.1.
    ps = ParametricSurrogate()
    
    # Import config to get learned alpha
    # Use relative import or sys.path
    import pinn_config as pc
    alpha = getattr(pc, 'THICKNESS_COMPLIANCE_ALPHA', 1.234)
    print(f"Using Learned Thickness Scaling Alpha: {alpha}")
    
    ref_params = {
        "E": 5.0,
        "thickness": 0.1,
        "restitution": 0.5,
        "friction": 0.3,
        "impact_velocity": 1.0
    }
    
    # Predict displacement for reference layer using Surrogate
    u_ref_surr = ps.predict(ref_params)
    
    # 5. Hybrid Scaling
    # Step A: Equivalent Homogeneous Modulus (E_eq)
    # EI_total = E_eq * (w * H_total^3) / 12
    # E_eq = 12 * EI_total / H_total^3
    E_eq = 12.0 * EI_actual / (total_thickness**3)
    
    # Step B: Scale by Modulus Ratio (u ~ 1/E)
    # The surrogate was trained with u ~ v / E^p, but let's assume linearity for small perturbations
    # or use the learned E_power if available. 
    e_power = getattr(pc, 'E_COMPLIANCE_POWER', 0.973)
    
    scaling_E = (ref_params['E'] / E_eq) ** e_power
    
    # Step C: Scale by Thickness Ratio (u ~ H^-alpha)
    scaling_H = (total_thickness / ref_params['thickness']) ** (-alpha)
    
    predicted_disp = u_ref_surr * scaling_E * scaling_H
    
    print(f"Reference (E={ref_params['E']}, t={ref_params['thickness']}): {u_ref_surr:.6f}")
    print(f"Equivalent E: {E_eq:.6f}")
    print(f"Scaling Factors -> E: {scaling_E:.4f}, H: {scaling_H:.4f}")
    print(f"\nPredicted Composite Disp: {predicted_disp:.6f}")
    
    # 6. Compare Peaks
    error = abs(predicted_disp - fea_peak_disp)
    rel_error = (error / fea_peak_disp) * 100
    print(f"Error: {error:.6f} ({rel_error:.2f}%)")
    
    # 7. Full Field Reconstruction & Visualization
    print("\nReconstructing Full Field for Visualization...")
    
    # Grid for plotting (XZ cross-section at mid-Y)
    ny = u_grid.shape[1]
    mid_y_idx = ny // 2
    
    # FEA Field (Ground Truth)
    uz_fea_xz = u_grid[:, mid_y_idx, :, 2].T # (nz, nx) -> Transpose to (nz, nx) for image
    # Note: imshow expects (rows, cols) where rows=Z, cols=X.
    # u_grid index order: [x, y, z, component]
    # So u_grid[:, mid, :, 2] is (nx, nz). Transpose gives (nz, nx).
    
    # Surrogate "Field" Approximation
    # We need to query the PINN for the *full reference field* on a normalized grid, 
    # then scale it.
    
    # Re-use PINN directly (bypass surrogate wrapper for field access)
    # The surrogate class wraps the peak extraction, but we need the field.
    # We can import `baseline` which has `_get_pinn`.
    from surrogate_workflow import baseline
    pinn_model, device = baseline._get_pinn()
    
    # Create grid for PINN query (matching FEA nodes roughly)
    nx_plot = 50
    nz_plot = 50
    
    x_p = np.linspace(0, 1.0, nx_plot)
    y_p = np.linspace(0.5, 0.5, 1) # Mid-Y
    z_ref = np.linspace(0, ref_params['thickness'], nz_plot) # Reference Height
    
    Xp, Yp, Zp = np.meshgrid(x_p, y_p, z_ref, indexing='ij')
    # Flatten
    pts_flat = np.zeros((Xp.size, 12))
    pts_flat[:, 0] = Xp.flatten()
    pts_flat[:, 1] = Yp.flatten()
    pts_flat[:, 2] = Zp.flatten()
    pts_flat[:, 3] = ref_params['E']  # E1
    pts_flat[:, 4] = ref_params['thickness'] / 3.0  # t1
    pts_flat[:, 5] = ref_params['E']  # E2
    pts_flat[:, 6] = ref_params['thickness'] / 3.0  # t2
    pts_flat[:, 7] = ref_params['E']  # E3
    pts_flat[:, 8] = ref_params['thickness'] / 3.0  # t3
    pts_flat[:, 9] = ref_params['restitution']
    pts_flat[:, 10] = ref_params['friction']
    pts_flat[:, 11] = ref_params['impact_velocity']
    
    with torch.no_grad():
        v_pred = pinn_model(torch.tensor(pts_flat, dtype=torch.float32).to(device)).cpu().numpy()
        
    # Un-normalize U form V
    # u = v * t_scale / E^pow
    # We use the raw V output and apply our Hybrid Scaling directly to it?
    # No, we must decode V to U_ref first using the PINN's training logic.
    
    # From baseline.compute_response:
    # t_scale = (float(pc.H) / t_val) ** float(pc.THICKNESS_COMPLIANCE_ALPHA)
    # u_final = (uz / (E_val ** float(pc.E_COMPLIANCE_POWER))) * t_scale
    
    H_baseline = getattr(pc, 'H', 0.1) # Training baseline H usually 0.1
    t_val = ref_params['thickness']
    t_scale_pinn = (H_baseline / t_val) ** alpha
    
    uz_raw = v_pred[:, 2]
    u_ref_field = (uz_raw / (ref_params['E'] ** e_power)) * t_scale_pinn
    u_ref_grid = u_ref_field.reshape(nx_plot, 1, nz_plot).squeeze(1).T # (nz, nx)
    
    # Now Apply Composite Scaling to this Field
    # U_composite approx = U_ref * Scaling_Factors
    u_comp_approx = u_ref_grid * scaling_E * scaling_H
    
    # Coordinate Mapping for Plotting
    # The U_comp_approx is defined on z in [0, 0.1] (ref thickness).
    # But physical reality is [0, 0.3].
    # We simply map the Plot Extent (dextents) to [0, 0.3].
    # This stretches the *image* of the field, effectively mapping the mode shape.
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. FEA
    im1 = axes[0].imshow(uz_fea_xz, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='jet', aspect='auto', interpolation='bilinear')
    axes[0].set_title(f"FEA Ground Truth\nPeak: {fea_peak_disp:.4f}")
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Surrogate (Scaled + Stretched)
    # Note: imshow extent stretches the [0.1] height data to [0.3] visual
    im2 = axes[1].imshow(u_comp_approx, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='jet', aspect='auto', interpolation='bilinear')
    axes[1].set_title(f"Surrogate (Hybrid Scaled)\nPred: {predicted_disp:.4f}")
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Absolute Difference
    # We need to interpolate FEA onto the Surrogate grid (or vice versa) to compute diff.
    # FEA grid is 30x30 roughly. Surrogate is 50x50.
    # Quick assumption: interpolate Surrogate to FEA resolution is easier?
    # Actually, visual comparison is main goal.
    from scipy.interpolate import RegularGridInterpolator
    
    # Interpolate Surrogate (defined on 50x50) to match FEA grid (nx, nz)
    z_surr_norm = np.linspace(0, total_thickness, nz_plot)
    x_surr_norm = np.linspace(0, 1.0, nx_plot)
    # RGI expects (z, x) for u_comp_approx which is (nz, nx)
    interp_func = RegularGridInterpolator((z_surr_norm, x_surr_norm), u_comp_approx, bounds_error=False, fill_value=0.0)
    
    # Create query points at FEA node locations
    z_fea_nodes = np.linspace(0, total_thickness, u_grid.shape[2]) # nz
    x_fea_nodes = np.linspace(0, 1.0, u_grid.shape[0]) # nx
    X_f, Z_f = np.meshgrid(x_fea_nodes, z_fea_nodes, indexing='xy') 
    # indexing='xy' -> X_f is (nz, nx)
    
    pts_fea = np.stack([Z_f.flatten(), X_f.flatten()], axis=1)
    u_surr_interp = interp_func(pts_fea).reshape(Z_f.shape)
    
    diff = np.abs(uz_fea_xz - u_surr_interp)
    im3 = axes[2].imshow(diff, extent=[0, 1.0, 0, total_thickness], origin='lower', cmap='magma', aspect='auto')
    axes[2].set_title(f"Absolute Error\nMax Diff: {np.max(diff):.4f}")
    plt.colorbar(im3, ax=axes[2])
    
    # 4. Layer Profile (Line Plot) at center x=0.5
    center_col_idx = u_grid.shape[0] // 2
    z_line = np.linspace(0, total_thickness, u_grid.shape[2])
    u_line_fea = uz_fea_xz[:, center_col_idx] # z-profile at center
    u_line_surr = u_surr_interp[:, center_col_idx]
    
    axes[3].plot(u_line_fea, z_line, 'b-', label='FEA', linewidth=2)
    axes[3].plot(u_line_surr, z_line, 'r--', label='Surrogate', linewidth=2)
    
    # Draw layer boundaries
    current_z = 0
    colors = ['gray', 'silver', 'whitesmoke']
    for i, l in enumerate(layers):
        rect = plt.Rectangle((0, current_z), np.max(u_line_fea)*1.1, l['thickness'], 
                             facecolor=colors[i%3], alpha=0.1)
        axes[3].add_patch(rect)
        current_z += l['thickness']
        axes[3].axhline(current_z, color='k', linestyle=':', alpha=0.3)
        
    axes[3].set_title("Centerline Profile (x=0.5)")
    axes[3].set_xlabel("Uz Displacement")
    axes[3].set_ylabel("Z Position")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Global labels
    for ax in axes[:3]:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Z Position")
        # Draw layer lines
        current_z = 0
        for l in layers:
            current_z += l['thickness']
            ax.axhline(current_z, color='white', linestyle='--', linewidth=1)

    plt.suptitle(f"3-Layer Composite Verification (Hybrid Scaling)\nModel: {pinn_model.__class__.__name__} | Error: {rel_error:.2f}%", fontsize=16)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), "visualization", "3_layer_field_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"\nComparing Plot saved to {out_path}")

if __name__ == "__main__":
    run_3layer_verification()
