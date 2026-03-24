import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver


def _u_from_v(v, pts):
    e_scale = 0.5 * (pts[:, 3:4] + pts[:, 5:6])
    t_scale = pts[:, 4:5] + pts[:, 6:7]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _load_pinn(device):
    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn


def _run_two_layer_fea(e1, e2, t1, t2, ne_x, ne_y, ne_z):
    thickness = float(t1) + float(t2)
    cfg = {
        "geometry": {
            "Lx": config.Lx,
            "Ly": config.Ly,
            "H": thickness,
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {
            "E_layers": [float(e1), float(e2)],
            "t_layers": [float(t1), float(t2)],
            "nu": config.nu_vals[0],
        },
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }
    return fem_solver.solve_two_layer_fem(cfg)


def _predict_pinn(pinn, device, x_flat, y_flat, z_flat, e1, e2, t1, t2):
    r_ref, mu_ref, v0_ref = _ref_params()
    pts = np.stack(
        [
            x_flat,
            y_flat,
            z_flat,
            np.full_like(x_flat, float(e1)),
            np.full_like(x_flat, float(t1)),
            np.full_like(x_flat, float(e2)),
            np.full_like(x_flat, float(t2)),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return _u_from_v(v, pts)


def _plot_case(case_name, output_dir, pinn, device, e1, e2, t1, t2):
    thickness = float(t1) + float(t2)
    x_nodes, y_nodes, z_nodes, u_fea = _run_two_layer_fea(e1, e2, t1, t2, ne_x=10, ne_y=10, ne_z=4)
    x_nodes = np.array(x_nodes)
    y_nodes = np.array(y_nodes)
    z_nodes = np.array(z_nodes)

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    top_z = np.full(x_grid.size, thickness)
    u_pinn_top = _predict_pinn(
        pinn,
        device,
        x_grid.ravel(),
        y_grid.ravel(),
        top_z,
        e1,
        e2,
        t1,
        t2,
    ).reshape(len(x_nodes), len(y_nodes), 3)

    u_z_fea_top = u_fea[:, :, -1, 2]
    u_z_pinn_top = u_pinn_top[:, :, 2]
    abs_err_top = np.abs(u_z_pinn_top - u_z_fea_top)

    mae = float(np.mean(abs_err_top))
    denom = float(np.max(np.abs(u_z_fea_top)))
    mae_pct = 100.0 * mae / denom if denom > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(min(np.min(u_z_fea_top), np.min(u_z_pinn_top)))
    vmax = float(max(np.max(u_z_fea_top), np.max(u_z_pinn_top)))

    c0 = axes[0].contourf(x_grid, y_grid, u_z_fea_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c0, ax=axes[0])
    axes[0].set_title(f"Two-Layer FEA\nE1={e1}, E2={e2}, t={thickness}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    c1 = axes[1].contourf(x_grid, y_grid, u_z_pinn_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=axes[1])
    axes[1].set_title("Two-Layer PINN")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    c2 = axes[2].contourf(x_grid, y_grid, abs_err_top, levels=50, cmap="magma")
    plt.colorbar(c2, ax=axes[2])
    axes[2].set_title(f"Abs Error\nMAE={mae_pct:.2f}%")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_name}_top.png"), dpi=160)
    plt.close(fig)

    mid_y_idx = len(y_nodes) // 2
    x_cross, z_cross = np.meshgrid(x_nodes, z_nodes, indexing="ij")
    y_cross = np.full(x_cross.size, y_nodes[mid_y_idx])
    u_pinn_cross = _predict_pinn(
        pinn,
        device,
        x_cross.ravel(),
        y_cross,
        z_cross.ravel(),
        e1,
        e2,
        t1,
        t2,
    ).reshape(len(x_nodes), len(z_nodes), 3)

    u_z_fea_cross = u_fea[:, mid_y_idx, :, 2]
    u_z_pinn_cross = u_pinn_cross[:, :, 2]
    abs_err_cross = np.abs(u_z_pinn_cross - u_z_fea_cross)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    vmin2 = float(min(np.min(u_z_fea_cross), np.min(u_z_pinn_cross)))
    vmax2 = float(max(np.max(u_z_fea_cross), np.max(u_z_pinn_cross)))

    c3 = axes2[0].contourf(x_cross, z_cross, u_z_fea_cross, levels=50, cmap="jet", vmin=vmin2, vmax=vmax2)
    plt.colorbar(c3, ax=axes2[0])
    axes2[0].set_title("Two-Layer FEA Cross-Section")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("z")

    c4 = axes2[1].contourf(x_cross, z_cross, u_z_pinn_cross, levels=50, cmap="jet", vmin=vmin2, vmax=vmax2)
    plt.colorbar(c4, ax=axes2[1])
    axes2[1].set_title("Two-Layer PINN Cross-Section")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("z")

    c5 = axes2[2].contourf(x_cross, z_cross, abs_err_cross, levels=50, cmap="magma")
    plt.colorbar(c5, ax=axes2[2])
    axes2[2].set_title(f"Abs Error Cross-Section\nMAE={np.mean(abs_err_cross):.5f}")
    axes2[2].set_xlabel("x")
    axes2[2].set_ylabel("z")

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{case_name}_cross_section.png"), dpi=160)
    plt.close(fig2)

    return mae_pct


def _plot_sweep(output_dir, pinn, device, t1, t2):
    e_values = [float(v) for v in getattr(config, "DATA_E_VALUES", [1.0, 5.0, 10.0])]
    mae_pct_grid = np.zeros((len(e_values), len(e_values)))

    fea_cache = {}
    for e1 in e_values:
        for e2 in e_values:
            x_nodes, y_nodes, z_nodes, u_fea = _run_two_layer_fea(e1, e2, t1, t2, ne_x=8, ne_y=8, ne_z=4)
            fea_cache[(e1, e2)] = (np.array(x_nodes), np.array(y_nodes), np.array(z_nodes), np.array(u_fea))

    for row_idx, e1 in enumerate(e_values):
        for col_idx, e2 in enumerate(e_values):
            x_nodes, y_nodes, _, u_fea = fea_cache[(e1, e2)]
            x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
            u_pinn_top = _predict_pinn(
                pinn,
                device,
                x_grid.ravel(),
                y_grid.ravel(),
                np.full(x_grid.size, float(t1) + float(t2)),
                e1,
                e2,
                t1,
                t2,
            ).reshape(len(x_nodes), len(y_nodes), 3)
            u_z_fea_top = u_fea[:, :, -1, 2]
            u_z_pinn_top = u_pinn_top[:, :, 2]
            mae = float(np.mean(np.abs(u_z_pinn_top - u_z_fea_top)))
            denom = float(np.max(np.abs(u_z_fea_top)))
            mae_pct_grid[row_idx, col_idx] = 100.0 * mae / denom if denom > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(mae_pct_grid, cmap="magma", origin="lower")
    plt.colorbar(image, ax=ax, label="Top-Surface MAE (%)")
    ax.set_xticks(range(len(e_values)))
    ax.set_yticks(range(len(e_values)))
    ax.set_xticklabels([f"{val:g}" for val in e_values])
    ax.set_yticklabels([f"{val:g}" for val in e_values])
    ax.set_xlabel("E2")
    ax.set_ylabel("E1")
    ax.set_title(f"Two-Layer PINN vs Two-Layer FEA\nTop-Surface MAE at t1={t1}, t2={t2}")

    for row_idx in range(len(e_values)):
        for col_idx in range(len(e_values)):
            ax.text(
                col_idx,
                row_idx,
                f"{mae_pct_grid[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"two_layer_sweep_t1{t1:.2f}_t2{t2:.2f}.png"), dpi=160)
    plt.close(fig)
    return e_values, mae_pct_grid


def main():
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_dir = os.path.join(PINN_WORKFLOW_DIR, "visualization_two_layer")
    os.makedirs(output_dir, exist_ok=True)

    pinn = _load_pinn(device)

    t1_values = [float(v) for v in getattr(config, "DATA_T1_VALUES", [float(getattr(config, "H", 0.1)) * 0.5])]
    t2_values = [float(v) for v in getattr(config, "DATA_T2_VALUES", [float(getattr(config, "H", 0.1)) * 0.5])]
    t1_min, t1_max = min(t1_values), max(t1_values)
    t2_min, t2_max = min(t2_values), max(t2_values)

    representative_cases = [
        ("two_layer_soft_bottom", 1.0, 10.0, t1_min, t2_max),
        ("two_layer_soft_top", 10.0, 1.0, t1_max, t2_min),
    ]
    for case_name, e1, e2, t1_val, t2_val in representative_cases:
        mae_pct = _plot_case(case_name, output_dir, pinn, device, e1, e2, t1_val, t2_val)
        print(f"{case_name}: top-surface MAE={mae_pct:.2f}%")

    _, mae_pct_grid = _plot_sweep(output_dir, pinn, device, t1_min, t2_min)
    print(f"Two-layer sweep mean MAE={np.mean(mae_pct_grid):.2f}%")
    print(f"Two-layer sweep worst MAE={np.max(mae_pct_grid):.2f}%")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
