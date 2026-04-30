"""Validation script comparing 3-layer PINN predictions against FEA solutions."""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "three-layer-workflow")
if not os.path.exists(PINN_WORKFLOW_DIR):
    PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "three-layer-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver


_CALIBRATION_CACHE = {}


def _u_from_v(v, pts):
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    u = scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    multiplier = _calibration_multiplier(pts)
    if multiplier is not None:
        u = u * multiplier
    return u


def _calibration_features(pts):
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    z = pts[:, 2:3]
    e1 = pts[:, 3:4]
    t1 = pts[:, 4:5]
    e2 = pts[:, 5:6]
    t2 = pts[:, 6:7]
    e3 = pts[:, 7:8]
    t3 = pts[:, 8:9]
    t_total = np.clip(t1 + t2 + t3, 1e-8, None)
    e_mean = np.clip((e1 + e2 + e3) / 3.0, 1e-8, None)
    z_hat = z / t_total
    e_ref = np.sqrt(float(config.E_RANGE[0]) * float(config.E_RANGE[1]))
    h_ref = float(getattr(config, "H", 0.1))
    load_x = ((x >= config.LOAD_PATCH_X[0]) & (x <= config.LOAD_PATCH_X[1])).astype(float)
    load_y = ((y >= config.LOAD_PATCH_Y[0]) & (y <= config.LOAD_PATCH_Y[1])).astype(float)
    load_patch = load_x * load_y
    xc = x - 0.5 * float(config.Lx)
    yc = y - 0.5 * float(config.Ly)
    feats = np.concatenate(
        [
            np.ones_like(x),
            np.log(e_mean / e_ref),
            np.log(np.clip(e1, 1e-8, None) / e_ref),
            np.log(np.clip(e2, 1e-8, None) / e_ref),
            np.log(np.clip(e3, 1e-8, None) / e_ref),
            np.log(h_ref / t_total),
            t1 / t_total,
            t2 / t_total,
            t3 / t_total,
            z_hat,
            z_hat**2,
            load_patch,
            xc,
            yc,
            xc**2,
            yc**2,
            xc * yc,
            load_patch * xc,
            load_patch * yc,
            load_patch * xc**2,
            load_patch * yc**2,
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _load_calibration():
    path = os.getenv("PINN_CALIBRATION_JSON")
    if not path:
        return None
    if path not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[path] = json.loads(open(path, "r").read()) if os.path.exists(path) else None
    return _CALIBRATION_CACHE[path]


def _calibration_multiplier(pts):
    cal = _load_calibration()
    if not cal:
        return None
    coeffs = cal.get("feature_coefficients")
    if coeffs is None:
        return None
    coeffs_arr = np.asarray(coeffs, dtype=float).reshape(-1, 1)
    feats = _calibration_features(pts)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_multiplier = np.nan_to_num(feats @ coeffs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    clip = float(cal.get("log_multiplier_clip", 1.5))
    return np.exp(np.clip(log_multiplier, -clip, clip))


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _load_pinn(device):
    pinn = model.MultiLayerPINN().to(device)
    model_path = os.getenv("PINN_MODEL_PATH") or os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    print(f"Loaded PINN checkpoint: {model_path}")
    return pinn


def _run_three_layer_fea(e1, e2, e3, t1, t2, t3, ne_x, ne_y, ne_z):
    thickness = float(t1) + float(t2) + float(t3)
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
            "E_layers": [float(e1), float(e2), float(e3)],
            "t_layers": [float(t1), float(t2), float(t3)],
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
    return fem_solver.solve_three_layer_fem(cfg)


def _predict_pinn(pinn, device, x_flat, y_flat, z_flat, e1, e2, e3, t1, t2, t3):
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
            np.full_like(x_flat, float(e3)),
            np.full_like(x_flat, float(t3)),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return _u_from_v(v, pts)


def _mae_pct(u_pred, u_ref):
    mae = float(np.mean(np.abs(u_pred - u_ref)))
    denom = float(np.max(np.abs(u_ref)))
    return 100.0 * mae / denom if denom > 0 else 0.0


def _plot_case(case_name, output_dir, pinn, device, e1, e2, e3, t1, t2, t3):
    thickness = float(t1) + float(t2) + float(t3)
    x_nodes, y_nodes, z_nodes, u_fea = _run_three_layer_fea(e1, e2, e3, t1, t2, t3, ne_x=16, ne_y=16, ne_z=8)
    x_nodes = np.array(x_nodes)
    y_nodes = np.array(y_nodes)
    z_nodes = np.array(z_nodes)

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    top_z = np.full(x_grid.size, thickness)
    u_pinn_top = _predict_pinn(
        pinn, device, x_grid.ravel(), y_grid.ravel(), top_z, e1, e2, e3, t1, t2, t3
    ).reshape(len(x_nodes), len(y_nodes), 3)

    u_z_fea_top = u_fea[:, :, -1, 2]
    u_z_pinn_top = u_pinn_top[:, :, 2]
    abs_err_top = np.abs(u_z_pinn_top - u_z_fea_top)
    mae_pct = _mae_pct(u_z_pinn_top, u_z_fea_top)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(min(np.min(u_z_fea_top), np.min(u_z_pinn_top)))
    vmax = float(max(np.max(u_z_fea_top), np.max(u_z_pinn_top)))

    c0 = axes[0].contourf(x_grid, y_grid, u_z_fea_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c0, ax=axes[0])
    axes[0].set_title(f"Three-Layer FEA\nE=[{e1},{e2},{e3}], t={thickness:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    c1 = axes[1].contourf(x_grid, y_grid, u_z_pinn_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=axes[1])
    axes[1].set_title("Three-Layer PINN")
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
        pinn, device, x_cross.ravel(), y_cross, z_cross.ravel(), e1, e2, e3, t1, t2, t3
    ).reshape(len(x_nodes), len(z_nodes), 3)

    u_z_fea_cross = u_fea[:, mid_y_idx, :, 2]
    u_z_pinn_cross = u_pinn_cross[:, :, 2]
    abs_err_cross = np.abs(u_z_pinn_cross - u_z_fea_cross)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    vmin2 = float(min(np.min(u_z_fea_cross), np.min(u_z_pinn_cross)))
    vmax2 = float(max(np.max(u_z_fea_cross), np.max(u_z_pinn_cross)))

    c3 = axes2[0].contourf(x_cross, z_cross, u_z_fea_cross, levels=50, cmap="jet", vmin=vmin2, vmax=vmax2)
    plt.colorbar(c3, ax=axes2[0])
    axes2[0].set_title("Three-Layer FEA Cross-Section")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("z")
    axes2[0].axhline(float(t1), color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    axes2[0].axhline(float(t1) + float(t2), color="white", linestyle="--", linewidth=1.2, alpha=0.9)

    c4 = axes2[1].contourf(x_cross, z_cross, u_z_pinn_cross, levels=50, cmap="jet", vmin=vmin2, vmax=vmax2)
    plt.colorbar(c4, ax=axes2[1])
    axes2[1].set_title("Three-Layer PINN Cross-Section")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("z")
    axes2[1].axhline(float(t1), color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    axes2[1].axhline(float(t1) + float(t2), color="white", linestyle="--", linewidth=1.2, alpha=0.9)

    c5 = axes2[2].contourf(x_cross, z_cross, abs_err_cross, levels=50, cmap="magma")
    plt.colorbar(c5, ax=axes2[2])
    axes2[2].set_title(f"Abs Error Cross-Section\nMAE={float(np.mean(abs_err_cross)):.5f}")
    axes2[2].set_xlabel("x")
    axes2[2].set_ylabel("z")
    axes2[2].axhline(float(t1), color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    axes2[2].axhline(float(t1) + float(t2), color="white", linestyle="--", linewidth=1.2, alpha=0.9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{case_name}_cross_section.png"), dpi=160)
    plt.close(fig2)

    return mae_pct


def _sweep(output_dir, pinn, device, e_values, t1_values, t2_values, t3_values):
    cases = []
    fea_cache = {}
    for t1 in t1_values:
        for t2 in t2_values:
            for t3 in t3_values:
                for e1 in e_values:
                    for e2 in e_values:
                        for e3 in e_values:
                            key = (e1, e2, e3, t1, t2, t3)
                            if key not in fea_cache:
                                fea_cache[key] = _run_three_layer_fea(e1, e2, e3, t1, t2, t3, ne_x=16, ne_y=16, ne_z=8)
                            x_nodes, y_nodes, _, u_fea = fea_cache[key]
                            x_nodes = np.array(x_nodes)
                            y_nodes = np.array(y_nodes)
                            thickness = float(t1) + float(t2) + float(t3)
                            x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
                            u_pinn_top = _predict_pinn(
                                pinn, device, x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, thickness),
                                e1, e2, e3, t1, t2, t3
                            ).reshape(len(x_nodes), len(y_nodes), 3)
                            u_z_fea_top = np.array(u_fea)[:, :, -1, 2]
                            u_z_pinn_top = u_pinn_top[:, :, 2]
                            cases.append(((e1, e2, e3, t1, t2, t3), _mae_pct(u_z_pinn_top, u_z_fea_top)))

    mae_pcts = np.array([v for _, v in cases], dtype=float)
    worst_idx = int(np.argmax(mae_pcts))
    worst_case, worst_mae = cases[worst_idx]
    mean_mae = float(np.mean(mae_pcts)) if len(mae_pcts) else 0.0

    t1_min, t2_min, t3_min = min(t1_values), min(t2_values), min(t3_values)
    for e2 in e_values:
        grid = np.zeros((len(e_values), len(e_values)))
        for i, e1 in enumerate(e_values):
            for j, e3 in enumerate(e_values):
                for (ce1, ce2, ce3, ct1, ct2, ct3), mae in cases:
                    if (ce1, ce2, ce3, ct1, ct2, ct3) == (e1, e2, e3, t1_min, t2_min, t3_min):
                        grid[i, j] = mae
                        break
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(grid, cmap="magma", origin="lower")
        plt.colorbar(image, ax=ax, label="Top-Surface MAE (%)")
        ax.set_xticks(range(len(e_values)))
        ax.set_yticks(range(len(e_values)))
        ax.set_xticklabels([f"{val:g}" for val in e_values])
        ax.set_yticklabels([f"{val:g}" for val in e_values])
        ax.set_xlabel("E3")
        ax.set_ylabel("E1")
        ax.set_title(f"Three-Layer PINN vs FEA (E2={e2:g})\nTop-Surface MAE at t=[{t1_min:g},{t2_min:g},{t3_min:g}]")
        for i in range(len(e_values)):
            for j in range(len(e_values)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"three_layer_sweep_tmin_E2{e2:g}.png"), dpi=160)
        plt.close(fig)

    return mean_mae, float(worst_mae), worst_case


def main():
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_dir = os.getenv("PINN_EVAL_OUT_DIR") or os.path.join(PINN_WORKFLOW_DIR, "visualization_three_layer")
    os.makedirs(output_dir, exist_ok=True)

    pinn = _load_pinn(device)

    if hasattr(config, "EVAL_E_VALUES"):
        e_values = [float(v) for v in config.EVAL_E_VALUES]
    else:
        e_range = getattr(config, "E_RANGE", [1.0, 10.0])
        e_values = [float(e_range[0]), float(e_range[-1])]

    t1_values = [float(v) for v in getattr(config, "EVAL_T1_VALUES", getattr(config, "DATA_T1_VALUES", [0.05]))]
    t2_values = [float(v) for v in getattr(config, "EVAL_T2_VALUES", getattr(config, "DATA_T2_VALUES", [0.05]))]
    t3_values = [float(v) for v in getattr(config, "EVAL_T3_VALUES", getattr(config, "DATA_T3_VALUES", [0.05]))]

    t1_min, t1_max = min(t1_values), max(t1_values)
    t2_min, t2_max = min(t2_values), max(t2_values)
    t3_min, t3_max = min(t3_values), max(t3_values)

    representative_cases = [
        ("three_layer_soft_bottom", 1.0, 10.0, 10.0, t1_max, t2_min, t3_min),
        ("three_layer_soft_middle", 10.0, 1.0, 10.0, t1_min, t2_max, t3_min),
        ("three_layer_soft_top", 10.0, 10.0, 1.0, t1_min, t2_min, t3_max),
    ]
    for case_name, e1, e2, e3, t1_val, t2_val, t3_val in representative_cases:
        mae_pct = _plot_case(case_name, output_dir, pinn, device, e1, e2, e3, t1_val, t2_val, t3_val)
        print(f"{case_name}: top-surface MAE={mae_pct:.2f}%")

    mean_mae, worst_mae, worst_case = _sweep(output_dir, pinn, device, e_values, t1_values, t2_values, t3_values)
    e1, e2, e3, t1, t2, t3 = worst_case
    print(f"Three-layer sweep mean MAE={mean_mae:.2f}%")
    print(f"Three-layer sweep worst MAE={worst_mae:.2f}%")
    print(f"Worst case: E=[{e1:g},{e2:g},{e3:g}] t=[{t1:g},{t2:g},{t3:g}]")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
