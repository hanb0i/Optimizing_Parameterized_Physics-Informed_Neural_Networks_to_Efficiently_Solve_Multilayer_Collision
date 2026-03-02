import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import torch

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
import physics


def _load_model(model_path, device):
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device, weights_only=True)

        # Backward-compat: replicate old single-net checkpoints into each layer net.
        if any(k.startswith("layer.") for k in sd.keys()):
            replicated = {}
            for k, v in sd.items():
                if k.startswith("layer."):
                    suffix = k[len("layer.") :]
                    for li in range(int(getattr(config, "NUM_LAYERS", 2))):
                        replicated[f"layers.{li}.{suffix}"] = v
                else:
                    replicated[k] = v
            sd = replicated

        try:
            pinn.load_state_dict(sd, strict=False)
            print(f"Model loaded from {model_path}")
        except RuntimeError as e:
            print(f"Warning: could not load checkpoint strictly due to shape mismatch: {e}")
            print("Proceeding with randomly initialized weights for verification script.")
    else:
        print(f"Warning: {model_path} not found.")
    pinn.eval()
    return pinn


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def run_fea(E_val, thickness):
    cfg = {
        'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': thickness},
        'material': {'E': E_val, 'nu': config.nu_vals[0]},
        'load_patch': {
            'pressure': config.p0,
            'x_start': config.LOAD_PATCH_X[0] / config.Lx,
            'x_end': config.LOAD_PATCH_X[1] / config.Lx,
            'y_start': config.LOAD_PATCH_Y[0] / config.Ly,
            'y_end': config.LOAD_PATCH_Y[1] / config.Ly,
        }
    }
    return fem_solver.solve_fem(cfg)


def _pinn_predict_top(pinn, device, X_flat, Y_flat, thickness, E_val):
    r_ref, mu_ref, v0_ref = _ref_params()
    Z_flat = np.ones_like(X_flat) * thickness
    L = int(getattr(config, "NUM_LAYERS", 2))
    t_list = [float(thickness) / float(L)] * L
    E_list = [float(E_val)] * L
    R_flat = np.ones_like(X_flat) * r_ref
    MU_flat = np.ones_like(X_flat) * mu_ref
    V0_flat = np.ones_like(X_flat) * v0_ref
    cols = [X_flat, Y_flat, Z_flat]
    for Ei, ti in zip(E_list, t_list):
        cols.append(np.ones_like(X_flat) * Ei)
        cols.append(np.ones_like(X_flat) * ti)
    cols.extend([R_flat, MU_flat, V0_flat])
    pts = np.stack(cols, axis=1).astype(np.float32, copy=False)
    with torch.no_grad():
        x_t = torch.tensor(pts, dtype=torch.float32).to(device)
        v = pinn(x_t)
        u = physics.decode_u(v, x_t).cpu().numpy()
    return u


def verify_parametric(pinn, device, viz_dir):
    os.makedirs(viz_dir, exist_ok=True)

    t_targets = [0.05, 0.1, 0.15]
    E_targets = [1.0, 5.0, 10.0]

    nx, ny = 101, 101
    x_range = np.linspace(0, config.Lx, nx)
    y_range = np.linspace(0, config.Ly, ny)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    fea_cache = {}  # {t_val: (x_n, y_n, z_n, u_arr)}
    for t_val in t_targets:
        x_n, y_n, z_n, u_fea = run_fea(1.0, t_val)
        fea_cache[t_val] = (np.array(x_n), np.array(y_n), np.array(z_n), np.array(u_fea, dtype=float))

    print("\n=== Parametric Verification ===")

    for t_val in t_targets:
        x_n, y_n, z_n, u_fea_ref = fea_cache[t_val]
        print(f"\n--- Thickness t={t_val} ---")

        fig_grid, axes_grid = plt.subplots(3, 3, figsize=(18, 15))

        for e_idx, E_val in enumerate(E_targets):
            # FEA: scale from E=1 reference
            scale = 1.0 / float(E_val)
            u_z_fea_top = (u_fea_ref * scale)[:, :, -1, 2].T  # (ny, nx)

            # PINN prediction
            u_pinn = _pinn_predict_top(pinn, device, X_flat, Y_flat, t_val, E_val)
            UZ_pinn = u_pinn[:, 2].reshape(ny, nx)

            peak_fea = float(np.min(u_z_fea_top))
            peak_pinn = float(np.min(UZ_pinn))
            rel_err = abs((peak_pinn - peak_fea) / peak_fea) if peak_fea != 0 else float('nan')
            print(f"  E={E_val}: FEA peak={peak_fea:.6f}, PINN peak={peak_pinn:.6f}, rel_err={rel_err:.3f}")

            # Interpolate FEA onto same grid for comparison
            u_fea_top_2d = (u_fea_ref * scale)[:, :, -1, 2]  # (nx_fea, ny_fea)
            interp = RegularGridInterpolator((x_n, y_n), u_fea_top_2d, method='linear', bounds_error=False, fill_value=None)
            UZ_fea_interp = interp(np.stack([X_flat, Y_flat], axis=1)).reshape(ny, nx)

            # --- 3x3 Grid: FEA / PINN / Error ---
            vmin = min(float(np.nanmin(UZ_fea_interp)), float(np.min(UZ_pinn)))
            vmax = max(float(np.nanmax(UZ_fea_interp)), float(np.max(UZ_pinn)))

            im0 = axes_grid[0, e_idx].contourf(X, Y, UZ_fea_interp, 50, cmap='jet', vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=axes_grid[0, e_idx])
            axes_grid[0, e_idx].set_title(f"FEA (E={E_val})\nPeak: {peak_fea:.4f}")

            im1 = axes_grid[1, e_idx].contourf(X, Y, UZ_pinn, 50, cmap='jet', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=axes_grid[1, e_idx])
            axes_grid[1, e_idx].set_title(f"PINN (E={E_val})\nPeak: {peak_pinn:.4f}")

            error = np.abs(UZ_pinn - UZ_fea_interp)
            im2 = axes_grid[2, e_idx].contourf(X, Y, error, 50, cmap='magma')
            plt.colorbar(im2, ax=axes_grid[2, e_idx])
            axes_grid[2, e_idx].set_title(f"Abs Error\nMAE: {np.nanmean(error):.5f}")

            # --- 3D Surface Comparison ---
            fig_3d = plt.figure(figsize=(16, 8))
            ax1 = fig_3d.add_subplot(121, projection='3d')
            ax2 = fig_3d.add_subplot(122, projection='3d')
            v_min_3d = min(float(np.min(UZ_pinn)), float(np.nanmin(UZ_fea_interp)))
            v_max_3d = max(float(np.max(UZ_pinn)), float(np.nanmax(UZ_fea_interp)))

            surf1 = ax1.plot_surface(X, Y, UZ_pinn, cmap='jet', edgecolor='none', vmin=v_min_3d, vmax=v_max_3d)
            ax1.set_title(f"PINN Uz (E={E_val}, t={t_val})")
            ax1.set_zlim(v_min_3d, v_max_3d)
            fig_3d.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            surf2 = ax2.plot_surface(X, Y, UZ_fea_interp, cmap='jet', edgecolor='none', vmin=v_min_3d, vmax=v_max_3d)
            ax2.set_title(f"FEA Uz (E={E_val}, t={t_val})")
            ax2.set_zlim(v_min_3d, v_max_3d)
            fig_3d.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

            fig_3d.tight_layout()
            fig_3d.savefig(os.path.join(viz_dir, f"3d_view_E{E_val:.1f}_t{t_val:.2f}.png"))
            plt.close(fig_3d)

            # --- Cross-Section at y=0.5 ---
            nz_c = 51
            z_c = np.linspace(0, t_val, nz_c)
            X_c, Z_c = np.meshgrid(x_range, z_c)
            r_ref, mu_ref, v0_ref = _ref_params()

            x_in = X_c.flatten()
            y_in = np.ones_like(x_in) * 0.5
            z_in = Z_c.flatten()
            L = int(getattr(config, "NUM_LAYERS", 2))
            t_list = [float(t_val) / float(L)] * L
            E_list = [float(E_val)] * L
            R_in = np.ones_like(x_in) * r_ref
            MU_in = np.ones_like(x_in) * mu_ref
            V0_in = np.ones_like(x_in) * v0_ref
            cols = [x_in, y_in, z_in]
            for Ei, ti in zip(E_list, t_list):
                cols.append(np.ones_like(x_in) * Ei)
                cols.append(np.ones_like(x_in) * ti)
            cols.extend([R_in, MU_in, V0_in])
            pts_c = np.stack(cols, axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                x_t = torch.tensor(pts_c, dtype=torch.float32).to(device)
                v_c = pinn(x_t)
                u_c = physics.decode_u(v_c, x_t).cpu().numpy()
            UZ_pinn_c = u_c[:, 2].reshape(nz_c, nx)

            # FEA cross-section: interpolate from 3D FEA data
            u_fea_3d = u_fea_ref * scale  # (nx_fea, ny_fea, nz_fea, 3)
            y_idx = np.argmin(np.abs(y_n - 0.5))
            uz_fea_xz = u_fea_3d[:, y_idx, :, 2]  # (nx_fea, nz_fea)
            interp_xz = RegularGridInterpolator((x_n, z_n), uz_fea_xz, method='linear', bounds_error=False, fill_value=None)
            UZ_fea_c = interp_xz(np.stack([X_c.flatten(), Z_c.flatten()], axis=1)).reshape(nz_c, nx)

            fig_cs, axes_cs = plt.subplots(1, 3, figsize=(18, 5))
            v_min_c = min(float(np.min(UZ_pinn_c)), float(np.nanmin(UZ_fea_c)))
            v_max_c = max(float(np.max(UZ_pinn_c)), float(np.nanmax(UZ_fea_c)))

            im_fea_c = axes_cs[0].contourf(X_c, Z_c, UZ_fea_c, 50, cmap='jet', vmin=v_min_c, vmax=v_max_c)
            plt.colorbar(im_fea_c, ax=axes_cs[0])
            axes_cs[0].set_title(f"FEA Cross-Section\nPeak: {np.nanmin(UZ_fea_c):.4f}")

            im_pinn_c = axes_cs[1].contourf(X_c, Z_c, UZ_pinn_c, 50, cmap='jet', vmin=v_min_c, vmax=v_max_c)
            plt.colorbar(im_pinn_c, ax=axes_cs[1])
            axes_cs[1].set_title(f"PINN Cross-Section\nPeak: {UZ_pinn_c.min():.4f}")

            err_c = np.abs(UZ_pinn_c - UZ_fea_c)
            im_err_c = axes_cs[2].contourf(X_c, Z_c, err_c, 50, cmap='magma')
            plt.colorbar(im_err_c, ax=axes_cs[2])
            axes_cs[2].set_title(f"Abs Error\nMAE: {np.nanmean(err_c):.5f}")

            for ax in axes_cs:
                ax.set_xlabel('x')
                ax.set_ylabel('z')

            fig_cs.tight_layout()
            fig_cs.savefig(os.path.join(viz_dir, f"cross_section_E{E_val:.1f}_t{t_val:.2f}.png"))
            plt.close(fig_cs)

        fig_grid.suptitle(f"Top-Surface Uz Comparison | t={t_val}", fontsize=16)
        fig_grid.tight_layout()
        fig_grid.savefig(os.path.join(viz_dir, f"top_view_t{t_val:.2f}.png"))
        plt.close(fig_grid)

    print("\nVerification Complete.")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    viz_dir = os.path.join(PINN_WORKFLOW_DIR, "visualization")

    pinn = _load_model(model_path, device)
    verify_parametric(pinn, device, viz_dir)
