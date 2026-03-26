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


def _load_model(model_path, device):
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device, weights_only=True)
        sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
        pinn.load_state_dict(sd, strict=False)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found.")
    pinn.eval()
    return pinn


def _u_from_v(v, E1_val, E2_val, t1_val, t2_val):
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    e_scale = 0.5 * (float(E1_val) + float(E2_val))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    t_total = max(float(t1_val) + float(t2_val), 1e-8)
    return scale * v / (e_scale ** e_pow) * (h_ref / t_total) ** alpha


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def run_fea(E1_val, E2_val, t1_val, t2_val):
    thickness = float(t1_val) + float(t2_val)
    cfg = {
        'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': thickness},
        'material': {
            'E_layers': [float(E1_val), float(E2_val)],
            't_layers': [float(t1_val), float(t2_val)],
            'nu': config.nu_vals[0],
        },
        'load_patch': {
            'pressure': config.p0,
            'x_start': config.LOAD_PATCH_X[0] / config.Lx,
            'x_end': config.LOAD_PATCH_X[1] / config.Lx,
            'y_start': config.LOAD_PATCH_Y[0] / config.Ly,
            'y_end': config.LOAD_PATCH_Y[1] / config.Ly,
        }
    }
    return fem_solver.solve_two_layer_fem(cfg)


def _pinn_predict_top(pinn, device, X_flat, Y_flat, t1_val, t2_val, E1_val, E2_val):
    r_ref, mu_ref, v0_ref = _ref_params()
    thickness = float(t1_val) + float(t2_val)
    Z_flat = np.ones_like(X_flat) * thickness
    E1_flat = np.ones_like(X_flat) * E1_val
    E2_flat = np.ones_like(X_flat) * E2_val
    R_flat = np.ones_like(X_flat) * r_ref
    MU_flat = np.ones_like(X_flat) * mu_ref
    V0_flat = np.ones_like(X_flat) * v0_ref
    T1_flat = np.ones_like(X_flat) * t1_val
    T2_flat = np.ones_like(X_flat) * t2_val
    pts = np.stack([X_flat, Y_flat, Z_flat, E1_flat, T1_flat, E2_flat, T2_flat, R_flat, MU_flat, V0_flat], axis=1)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
    return _u_from_v(v, E1_val, E2_val, t1_val, t2_val)


def verify_parametric(pinn, device, viz_dir):
    os.makedirs(viz_dir, exist_ok=True)

    t1_targets = [float(val) for val in getattr(config, "DATA_T1_VALUES", [float(config.H) * 0.5])]
    t2_targets = [float(val) for val in getattr(config, "DATA_T2_VALUES", [float(config.H) * 0.5])]
    E_targets = [1.0, 5.0, 10.0]

    nx, ny = 101, 101
    x_range = np.linspace(0, config.Lx, nx)
    y_range = np.linspace(0, config.Ly, ny)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    fea_cache = {}  # {(t1, t2): (x_n, y_n, z_n, u_arr)}
    for t1_val in t1_targets:
        for t2_val in t2_targets:
            x_n, y_n, z_n, u_fea = run_fea(1.0, 1.0, t1_val, t2_val)
            fea_cache[(t1_val, t2_val)] = (np.array(x_n), np.array(y_n), np.array(z_n), np.array(u_fea, dtype=float))

    print("\n=== Parametric Verification ===")

    for t1_val in t1_targets:
        for t2_val in t2_targets:
            x_n, y_n, z_n, u_fea_ref = fea_cache[(t1_val, t2_val)]
            print(f"\n--- Thickness t1={t1_val}, t2={t2_val} ---")

            fig_grid, axes_grid = plt.subplots(3, 3, figsize=(18, 15))

            for e_idx, E_val in enumerate(E_targets):
                scale = 1.0 / float(E_val)
                u_z_fea_top = (u_fea_ref * scale)[:, :, -1, 2].T

                u_pinn = _pinn_predict_top(pinn, device, X_flat, Y_flat, t1_val, t2_val, E_val, E_val)
                UZ_pinn = u_pinn[:, 2].reshape(ny, nx)

                peak_fea = float(np.min(u_z_fea_top))
                peak_pinn = float(np.min(UZ_pinn))
                rel_err = abs((peak_pinn - peak_fea) / peak_fea) if peak_fea != 0 else float('nan')
                print(f"  E={E_val}: FEA peak={peak_fea:.6f}, PINN peak={peak_pinn:.6f}, rel_err={rel_err:.3f}")

                u_fea_top_2d = (u_fea_ref * scale)[:, :, -1, 2]
                interp = RegularGridInterpolator((x_n, y_n), u_fea_top_2d, method='linear', bounds_error=False, fill_value=None)
                UZ_fea_interp = interp(np.stack([X_flat, Y_flat], axis=1)).reshape(ny, nx)

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

                fig_3d = plt.figure(figsize=(16, 8))
                ax1 = fig_3d.add_subplot(121, projection='3d')
                ax2 = fig_3d.add_subplot(122, projection='3d')
                v_min_3d = min(float(np.min(UZ_pinn)), float(np.nanmin(UZ_fea_interp)))
                v_max_3d = max(float(np.max(UZ_pinn)), float(np.nanmax(UZ_fea_interp)))

                surf1 = ax1.plot_surface(X, Y, UZ_pinn, cmap='jet', edgecolor='none', vmin=v_min_3d, vmax=v_max_3d)
                ax1.set_title(f"PINN Uz (E={E_val}, t1={t1_val}, t2={t2_val})")
                ax1.set_zlim(v_min_3d, v_max_3d)
                fig_3d.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

                surf2 = ax2.plot_surface(X, Y, UZ_fea_interp, cmap='jet', edgecolor='none', vmin=v_min_3d, vmax=v_max_3d)
                ax2.set_title(f"FEA Uz (E={E_val}, t1={t1_val}, t2={t2_val})")
                ax2.set_zlim(v_min_3d, v_max_3d)
                fig_3d.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

                fig_3d.tight_layout()
                fig_3d.savefig(os.path.join(viz_dir, f"3d_view_E{E_val:.1f}_t1{t1_val:.2f}_t2{t2_val:.2f}.png"))
                plt.close(fig_3d)

                nz_c = 51
                z_c = np.linspace(0, float(t1_val) + float(t2_val), nz_c)
                X_c, Z_c = np.meshgrid(x_range, z_c)
                r_ref, mu_ref, v0_ref = _ref_params()

                x_in = X_c.flatten()
                y_in = np.ones_like(x_in) * 0.5
                z_in = Z_c.flatten()
                E_in = np.ones_like(x_in) * E_val
                E2_in = np.ones_like(x_in) * E_val
                R_in = np.ones_like(x_in) * r_ref
                MU_in = np.ones_like(x_in) * mu_ref
                V0_in = np.ones_like(x_in) * v0_ref
                T1_in = np.ones_like(x_in) * t1_val
                T2_in = np.ones_like(x_in) * t2_val
                pts_c = np.stack([x_in, y_in, z_in, E_in, T1_in, E2_in, T2_in, R_in, MU_in, V0_in], axis=1)

                with torch.no_grad():
                    v_c = pinn(torch.tensor(pts_c, dtype=torch.float32).to(device)).cpu().numpy()
                u_c = _u_from_v(v_c, E_val, E_val, t1_val, t2_val)
                UZ_pinn_c = u_c[:, 2].reshape(nz_c, nx)

                u_fea_3d = u_fea_ref * scale
                y_idx = np.argmin(np.abs(y_n - 0.5))
                uz_fea_xz = u_fea_3d[:, y_idx, :, 2]
                interp_xz = RegularGridInterpolator((x_n, z_n), uz_fea_xz, method='linear', bounds_error=False, fill_value=None)
                UZ_fea_c = interp_xz(np.stack([X_c.flatten(), Z_c.flatten()], axis=1)).reshape(nz_c, nx)

                fig_cs, axes_cs = plt.subplots(1, 3, figsize=(18, 5))
                v_min_c = min(float(np.min(UZ_pinn_c)), float(np.nanmin(UZ_fea_c)))
                v_max_c = max(float(np.max(UZ_pinn_c)), float(np.nanmax(UZ_fea_c)))

                im_fea_c = axes_cs[0].contourf(X_c, Z_c, UZ_fea_c, 50, cmap='jet', vmin=v_min_c, vmax=v_max_c)
                plt.colorbar(im_fea_c, ax=axes_cs[0])
                axes_cs[0].set_title(f"FEA Cross-Section\nPeak: {np.nanmin(UZ_fea_c):.4f}")
                axes_cs[0].axhline(float(t1_val), color='white', linestyle='--', linewidth=1.2, alpha=0.9)

                im_pinn_c = axes_cs[1].contourf(X_c, Z_c, UZ_pinn_c, 50, cmap='jet', vmin=v_min_c, vmax=v_max_c)
                plt.colorbar(im_pinn_c, ax=axes_cs[1])
                axes_cs[1].set_title(f"PINN Cross-Section\nPeak: {UZ_pinn_c.min():.4f}")
                axes_cs[1].axhline(float(t1_val), color='white', linestyle='--', linewidth=1.2, alpha=0.9)

                err_c = np.abs(UZ_pinn_c - UZ_fea_c)
                im_err_c = axes_cs[2].contourf(X_c, Z_c, err_c, 50, cmap='magma')
                plt.colorbar(im_err_c, ax=axes_cs[2])
                axes_cs[2].set_title(f"Abs Error\nMAE: {np.nanmean(err_c):.5f}")
                axes_cs[2].axhline(float(t1_val), color='white', linestyle='--', linewidth=1.2, alpha=0.9)

                for ax in axes_cs:
                    ax.set_xlabel('x')
                    ax.set_ylabel('z')

                fig_cs.tight_layout()
                fig_cs.savefig(os.path.join(viz_dir, f"cross_section_E{E_val:.1f}_t1{t1_val:.2f}_t2{t2_val:.2f}.png"))
                plt.close(fig_cs)

            fig_grid.suptitle(f"Top-Surface Uz Comparison | t1={t1_val}, t2={t2_val}", fontsize=16)
            fig_grid.tight_layout()
            fig_grid.savefig(os.path.join(viz_dir, f"top_view_t1{t1_val:.2f}_t2{t2_val:.2f}.png"))
            plt.close(fig_grid)

    print("\nVerification Complete.")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    viz_dir = os.path.join(PINN_WORKFLOW_DIR, "visualization")

    pinn = _load_model(model_path, device)
    verify_parametric(pinn, device, viz_dir)
