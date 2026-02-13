
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
import data
import physics

def _get_e_sweep_values():
    if hasattr(config, "VERIFY_E_SWEEP_VALUES"):
        return [float(v) for v in config.VERIFY_E_SWEEP_VALUES]
    e_min, e_max = config.E_RANGE
    steps = int(os.getenv("PINN_VERIFY_E_STEPS", "10"))
    steps = max(2, steps)
    return np.linspace(float(e_min), float(e_max), steps).tolist()

def _u_from_v(v_pinn_flat, E_val, thickness):
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    if alpha == 0.0:
        t_scale = 1.0
    else:
        t_scale = (float(getattr(config, "H", 1.0)) / max(1e-8, float(thickness))) ** alpha
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    return (v_pinn_flat / (float(E_val) ** e_pow)) * t_scale

def verify_e_sweep(pinn, device, thickness_values, viz_dir, fea_refs=None):
    """Verify E-parametrization by comparing peak top-surface u_z vs E.

    For linear elasticity here, u scales approximately as 1/E. We compute one
    FEA solution per thickness at E=1, then scale the whole field by 1/E for
    the sweep curve.
    """
    e_values = _get_e_sweep_values()
    e_ref = 1.0

    print("\n=== E Sweep Verification (Peak u_z vs E) ===")
    for thickness in thickness_values:
        if fea_refs is not None and thickness in fea_refs:
            x_nodes, y_nodes, z_nodes, u_fea_ref = fea_refs[thickness]
        else:
            x_nodes, y_nodes, z_nodes, u_fea_ref = run_fea(e_ref, thickness)

        u_z_fea_top_ref = np.array(u_fea_ref, dtype=float)[:, :, -1, 2].T  # (ny, nx)

        X, Y = np.meshgrid(x_nodes, y_nodes)
        nx, ny = len(x_nodes), len(y_nodes)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = np.ones_like(X_flat) * thickness
        T_flat = np.ones_like(X_flat) * thickness

        fea_peaks = []
        pinn_peaks = []
        rel_peak_errs = []

        for E_val in e_values:
            u_z_fea_top = u_z_fea_top_ref * (e_ref / float(E_val))
            peak_fea = float(np.min(u_z_fea_top))

            E_flat = np.ones_like(X_flat) * E_val
            input_pts = np.stack([X_flat, Y_flat, Z_flat, E_flat, T_flat], axis=1)
            input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
            with torch.no_grad():
                v_pinn_flat = pinn(input_tensor).cpu().numpy()
            u_pinn_flat = _u_from_v(v_pinn_flat, E_val, thickness)
            u_z_pinn_top = u_pinn_flat[:, 2].reshape(ny, nx)
            peak_pinn = float(np.min(u_z_pinn_top))

            fea_peaks.append(peak_fea)
            pinn_peaks.append(peak_pinn)
            rel_peak_errs.append(abs((peak_pinn - peak_fea) / peak_fea) if peak_fea != 0 else np.nan)

        e_arr = np.array(e_values, dtype=float)
        fea_amp = np.clip(-np.array(fea_peaks, dtype=float), 1e-12, None)
        pinn_amp = np.clip(-np.array(pinn_peaks, dtype=float), 1e-12, None)
        k_fea = -np.polyfit(np.log(e_arr), np.log(fea_amp), 1)[0]
        k_pinn = -np.polyfit(np.log(e_arr), np.log(pinn_amp), 1)[0]

        print(f"\nThickness t={thickness}:")
        print(f"  Expected scaling ~ 1/E (k≈1.0). FEA fit k={k_fea:.3f}, PINN fit k={k_pinn:.3f}")
        print(f"  Peak rel error: mean={float(np.nanmean(rel_peak_errs)):.3f}, max={float(np.nanmax(rel_peak_errs)):.3f}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(e_values, fea_peaks, "o-", label="FEA peak u_z (scaled)", linewidth=1.5)
        ax.plot(e_values, pinn_peaks, "s-", label="PINN peak u_z", linewidth=1.5)
        ax.set_xlabel("E")
        ax.set_ylabel("Peak u_z (top surface)")
        ax.set_title(f"E Sweep | t={thickness} | FEA k={k_fea:.2f}, PINN k={k_pinn:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(viz_dir, f"e_sweep_peaks_t{thickness:.3f}.png")
        plt.savefig(out_path)
        print(f"  Saved {out_path}")
        plt.close()

def plot_training_data_distribution():
    print("Generating Training Data Distribution Plot...")
    
    # Generate a sample batch of data
    # Note: This is a fresh sample based on the distribution logic, not the exact points from training history.
    sample_data = data.get_data()
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Interior Points (3D Scatter)
    ax1 = fig.add_subplot(231, projection='3d')
    pts = sample_data['interior'][0].numpy()
    # decimate for speed/clarity
    if len(pts) > 1000:
        indices = np.random.choice(len(pts), 1000, replace=False)
        pts = pts[indices]
    
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], cmap='viridis', s=2)
    ax1.set_title(f"Interior (Sample N={len(pts)})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.colorbar(sc1, ax=ax1, label='E')

    # 2. BC Sides (3D Scatter)
    ax2 = fig.add_subplot(232, projection='3d')
    pts = sample_data['sides'][0].numpy()
    if len(pts) > 1000:
        indices = np.random.choice(len(pts), 1000, replace=False)
        pts = pts[indices]
    
    sc2 = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], cmap='coolwarm', s=2)
    ax2.set_title(f"BC Sides (Sample N={len(pts)})")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.colorbar(sc2, ax=ax2, label='E')

    # 3. Top Load (2D Projection x-y)
    ax3 = fig.add_subplot(233)
    pts = sample_data['top_load'].numpy()
    sc3 = ax3.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap='plasma', s=5)
    ax3.set_title(f"Top Load (z=H) (N={len(pts)})")
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(0, config.Lx)
    ax3.set_ylim(0, config.Ly)
    # Draw load patch box
    rect = plt.Rectangle((config.LOAD_PATCH_X[0], config.LOAD_PATCH_Y[0]), 
                         config.LOAD_PATCH_X[1]-config.LOAD_PATCH_X[0], 
                         config.LOAD_PATCH_Y[1]-config.LOAD_PATCH_Y[0], 
                         linewidth=1, edgecolor='r', facecolor='none')
    ax3.add_patch(rect)
    plt.colorbar(sc3, ax=ax3, label='E')

    # 4. Top Free (2D Projection x-y)
    ax4 = fig.add_subplot(234)
    pts = sample_data['top_free'].numpy()
    if len(pts) > 1000:
        indices = np.random.choice(len(pts), 1000, replace=False)
        pts = pts[indices]
    sc4 = ax4.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap='plasma', s=5)
    ax4.set_title(f"Top Free (z=H) (Sample N={len(pts)})")
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_xlim(0, config.Lx)
    ax4.set_ylim(0, config.Ly)
    plt.colorbar(sc4, ax=ax4, label='E')

    # 5. Bottom (2D Projection x-y)
    ax5 = fig.add_subplot(235)
    pts = sample_data['bottom'].numpy()
    if len(pts) > 1000:
        indices = np.random.choice(len(pts), 1000, replace=False)
        pts = pts[indices]
    sc5 = ax5.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap='plasma', s=5)
    ax5.set_title(f"Bottom (z=0) (Sample N={len(pts)})")
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_xlim(0, config.Lx)
    ax5.set_ylim(0, config.Ly)
    plt.colorbar(sc5, ax=ax5, label='E')

    plt.tight_layout()
    save_path = os.path.join(REPO_ROOT, "pinn-workflow", "visualization", "training_data_distribution.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_training_history():
    print("Generating Training History Plot...")
    history_path = os.path.join(PINN_WORKFLOW_DIR, "loss_history.npy")
    if not os.path.exists(history_path):
        history_path = os.path.join(REPO_ROOT, "loss_history.npy")
        
    if not os.path.exists(history_path):
        print(f"Error: Could not find loss_history.npy at {history_path}")
        return

    hist = np.load(history_path, allow_pickle=True).item()
    adam = hist.get('adam', {})
    lbfgs = hist.get('lbfgs', {})
    
    total_loss = adam.get('total', []) + lbfgs.get('total', [])
    n_adam = len(adam.get('total', []))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    def concat_losses(key):
        return adam.get(key, []) + lbfgs.get(key, [])

    # Plot total loss
    axes[0].plot(total_loss, linewidth=1.5, color='black')
    axes[0].axvline(x=n_adam, color='r', linestyle='--', label='Switch to L-BFGS')
    axes[0].set_xlabel("Training Step", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Total Loss", fontsize=14)
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PDE loss
    pde_loss = concat_losses('pde')
    axes[1].plot(pde_loss, linewidth=1.5, color='green')
    axes[1].axvline(x=n_adam, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("PDE Loss", fontsize=12)
    axes[1].set_title("PDE Loss", fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    # Boundary condition losses
    bc_sides_loss = concat_losses('bc_sides')
    free_top_loss = concat_losses('free_top')
    free_bot_loss = concat_losses('free_bot')
    axes[2].plot(bc_sides_loss, linewidth=1.5, label='BC Sides', color='blue')
    axes[2].plot(free_top_loss, linewidth=1.5, label='Free Top', color='cyan')
    axes[2].plot(free_bot_loss, linewidth=1.5, label='Free Bot', color='purple')
    axes[2].axvline(x=n_adam, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Training Step", fontsize=12)
    axes[2].set_ylabel("BC Loss", fontsize=12)
    axes[2].set_title("Boundary Condition Losses", fontsize=14)
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Load loss
    load_loss = concat_losses('load')
    axes[3].plot(load_loss, linewidth=1.5, color='red')
    axes[3].axvline(x=n_adam, color='r', linestyle='--', alpha=0.5)
    axes[3].set_xlabel("Training Step", fontsize=12)
    axes[3].set_ylabel("Load Loss", fontsize=12)
    axes[3].set_title("Load BC Loss", fontsize=14)
    axes[3].set_yscale('log')
    axes[3].grid(True, alpha=0.3)

    # FEM error plots (if available)
    has_fem = len(adam.get('fem_mae', [])) > 0
    if has_fem:
        # FEM MAE
        adam_fem_epochs = adam['epochs']
        adam_fem_mae = adam['fem_mae']
        lbfgs_fem_steps = [n_adam + s for s in lbfgs.get('steps', [])]
        lbfgs_fem_mae = lbfgs.get('fem_mae', [])
        
        axes[4].plot(adam_fem_epochs, adam_fem_mae, 'o-', linewidth=1.5, label='Adam', color='blue')
        if len(lbfgs_fem_mae) > 0:
            axes[4].plot(lbfgs_fem_steps, lbfgs_fem_mae, 's-', linewidth=1.5, label='L-BFGS', color='orange')
        axes[4].axvline(x=n_adam, color='r', linestyle='--', alpha=0.5)
        axes[4].set_xlabel("Training Step", fontsize=12)
        axes[4].set_ylabel("FEM MAE", fontsize=12)
        axes[4].set_title("FEM Mean Absolute Error", fontsize=14)
        axes[4].set_yscale('log')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # FEM Max Error
        adam_fem_max = adam['fem_max_err']
        lbfgs_fem_max = lbfgs.get('fem_max_err', [])
        
        axes[5].plot(adam_fem_epochs, adam_fem_max, 'o-', linewidth=1.5, label='Adam', color='blue')
        if len(lbfgs_fem_max) > 0:
            axes[5].plot(lbfgs_fem_steps, lbfgs_fem_max, 's-', linewidth=1.5, label='L-BFGS', color='orange')
        axes[5].axvline(x=n_adam, color='r', linestyle='--', alpha=0.5)
        axes[5].set_xlabel("Training Step", fontsize=12)
        axes[5].set_ylabel("FEM Max Error", fontsize=12)
        axes[5].set_title("FEM Maximum Error", fontsize=14)
        axes[5].set_yscale('log')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(REPO_ROOT, "pinn-workflow", "visualization", "training_history.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_pde_residual_xz(pinn, device):
    print("Generating PDE Residual X-Z Plot...")
    
    # Define grid in x-z plane at y = 0.5 * Ly
    nx, nz = 100, 100
    x = np.linspace(0, config.Lx, nx)
    t_min, t_max = config.THICKNESS_RANGE
    t_mean = 0.5 * (t_min + t_max)
    z = np.linspace(0, t_mean, nz)
    X_res, Z_res = np.meshgrid(x, z)
    Y_res = np.ones_like(X_res) * (config.Ly * 0.5)
    
    # E and thickness values? Use mean values for this diagnostic
    E_mean = np.mean(config.E_vals)
    E_grid = np.ones_like(X_res) * E_mean
    T_grid = np.ones_like(X_res) * t_mean
    
    # Prepare input
    # Need to flatten
    pts = np.stack([X_res.flatten(), Y_res.flatten(), Z_res.flatten(), E_grid.flatten(), T_grid.flatten()], axis=1)
    pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
    
    # Reuse compute_residuals logic but just for this grid
    # We need a data dict structure for physics.compute_residuals OR just reimplement the check here.
    # Reimplementing simplified version to get spatial field:
    
    pts_tensor.requires_grad = True
    
    # Forward pass
    v_pred = pinn(pts_tensor, 0) # Layer index 0 (doesn't usually matter for single-net if monolithic, but model expects something)
    
    # Material properties
    E_local = pts_tensor[:, 3:4]
    nu = config.nu_vals[0]
    lm = (E_local * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    # u = v / E (PDE/traction definition used in training)
    u = v_pred / E_local
    
    # Gradients
    grad_u = physics.gradient(u, pts_tensor)
    eps = physics.strain(grad_u)
    sig = physics.stress(eps, lm, mu)
    div_sigma = physics.divergence(sig, pts_tensor)
    
    # Residual: -div(sigma) -- should be 0
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1)).detach().cpu().numpy()
    
    residual_mag_grid = residual_mag.reshape(nz, nx)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    c = ax.contourf(X_res, Z_res, residual_mag_grid, levels=50, cmap='viridis')
    ax.set_title(f"PDE Residual Magnitude |−∇·σ| (x-z plane at y={config.Ly*0.5:.2f}, E={E_mean}, t={t_mean})", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("Residual Magnitude")
    
    plt.tight_layout()
    save_path = os.path.join(REPO_ROOT, "pinn-workflow", "visualization", "pde_residual_xz.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def run_fea(E_val, thickness):
    print(f"Running FEA for E={E_val}, thickness={thickness}...")
    # Mock config for FEA solver
    cfg = {
        'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': thickness},
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
    t_min, t_max = config.THICKNESS_RANGE
    thickness_values = [t_min, config.H, t_max]
    results = {}

    # Load PINN
    requested_device = os.getenv("PINN_DEVICE")
    if requested_device:
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(REPO_ROOT, "pinn-workflow", "pinn_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(REPO_ROOT, "pinn_model.pth")
        
    print(f"Loading model from: {model_path}")
    pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    pinn.eval()

    # Create visualization dir if not exists
    viz_dir = os.path.join(REPO_ROOT, "pinn-workflow", "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    # --- 1. Diagnostic Plots ---
    if os.getenv("PINN_VERIFY_SKIP_DIAGNOSTICS", "0") != "1":
        plot_training_data_distribution()
        plot_training_history()
        plot_pde_residual_xz(pinn, device)

    # --- 2. Parametric Verification Plot ---
    # Create one figure per thickness to preserve FEA/PINN/Error rows
    print("\nGenerating Parametric Comparison Plots...")
    fea_refs = {}
    for thickness in thickness_values:
        # Linear elastic response: compute FEA once at E=1 and scale by 1/E.
        x_nodes_ref, y_nodes_ref, z_nodes_ref, u_fea_ref = run_fea(1.0, thickness)
        u_fea_ref = np.array(u_fea_ref, dtype=float)
        fea_refs[thickness] = (x_nodes_ref, y_nodes_ref, z_nodes_ref, u_fea_ref)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        # Row 0: FEA, Row 1: PINN, Row 2: Error
        # Cols: E=1, E=5, E=10

        for idx, E_val in enumerate(E_test_values):
            # 1. FEA via scaling from E=1 reference
            x_nodes, y_nodes, z_nodes = x_nodes_ref, y_nodes_ref, z_nodes_ref
            u_fea_grid = u_fea_ref * (1.0 / float(E_val))
            
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
            Z_flat = np.ones_like(X_flat) * thickness
            E_flat = np.ones_like(X_flat) * E_val
            T_flat = np.ones_like(X_flat) * thickness
            
            # Prepare input (N, 5)
            input_pts = np.stack([X_flat, Y_flat, Z_flat, E_flat, T_flat], axis=1)
            input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                v_pinn_flat = pinn(input_tensor).cpu().numpy()
                
            u_pinn_flat = _u_from_v(v_pinn_flat, E_val, thickness)
                
            u_z_pinn_top = u_pinn_flat[:, 2].reshape(ny, nx)
            
            # 3. Compute Error
            abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
            mae = np.mean(abs_diff)
            max_err = np.max(abs_diff)
            peak_fea = np.min(u_z_fea_top)
            peak_pinn = np.min(u_z_pinn_top)
            
            print(f"\n--- Results for E={E_val}, thickness={thickness} ---")
            print(f"Peak Deflection FEA: {peak_fea:.6f}")
            print(f"Peak Deflection PINN: {peak_pinn:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"Max Error: {max_err:.6f}")
            
            # 4. Plot
            # Row 0: FEA
            ax_fea = axes[0, idx]
            c1 = ax_fea.contourf(X, Y, u_z_fea_top, levels=50, cmap="jet")
            ax_fea.set_title(f"FEA (E={E_val}, t={thickness})\nPeak: {peak_fea:.5f}")
            plt.colorbar(c1, ax=ax_fea)
            
            # Row 1: PINN
            ax_pinn = axes[1, idx]
            c2 = ax_pinn.contourf(X, Y, u_z_pinn_top, levels=50, cmap="jet")
            ax_pinn.set_title(f"PINN (E={E_val}, t={thickness})\nPeak: {peak_pinn:.5f}")
            plt.colorbar(c2, ax=ax_pinn)
            
            # Row 2: Error
            ax_err = axes[2, idx]
            c3 = ax_err.contourf(X, Y, abs_diff, levels=50, cmap="magma")
            ax_err.set_title(f"Error (MAE: {mae:.5f})")
            plt.colorbar(c3, ax=ax_err)

        plt.tight_layout()
        result_path = os.path.join(viz_dir, f"parametric_verification_t{thickness:.3f}.png")
        plt.savefig(result_path)
        print(f"\nVerification plot saved to: {result_path}")
    # plt.show()

    # --- 3. E Sweep Verification ---
    verify_e_sweep(pinn, device, thickness_values, viz_dir, fea_refs=fea_refs)

if __name__ == "__main__":
    main()
