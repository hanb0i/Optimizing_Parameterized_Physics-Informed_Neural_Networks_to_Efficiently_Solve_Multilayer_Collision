#!/usr/bin/env python3
"""Generate averaged cross-section MAE heatmaps across multiple random cases."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))


def _one_layer_averaged(n_cases: int = 15, ne_x: int = 16, ne_y: int = 16, ne_z: int = 8):
    one_layer_dir = REPO_ROOT / "one-layer-workflow"
    sys.path.insert(0, str(one_layer_dir))
    for mod in ["model", "pinn_config", "fem_solver"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import pinn_config as config
    import model
    import fem_solver

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    ckpt = one_layer_dir / "pinn_model.pth"
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    target_sd = pinn.state_dict()
    w_key = "layer.net.0.weight"
    if w_key in sd and w_key in target_sd:
        src_w = sd[w_key]
        tgt_w = target_sd[w_key]
        if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0]:
            if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
                adapted = torch.zeros_like(tgt_w)
                adapted[:, 0:5] = src_w[:, 0:5]
                adapted[:, 8:11] = src_w[:, 5:8]
                sd[w_key] = adapted
            elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
                adapted = torch.zeros_like(tgt_w)
                adapted[:, 0:7] = src_w[:, 0:7]
                adapted[:, 8:11] = src_w[:, 7:10]
                sd[w_key] = adapted
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    # Read random cases from CSV
    csv_path = REPO_ROOT / "graphs" / "generalized_study" / "one_layer_random_50.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows = rows[:n_cases]

    # Common grid in normalized coordinates
    nx_common = 51
    nz_common = 51
    x_common = np.linspace(0, config.Lx, nx_common)
    z_hat_common = np.linspace(0, 1, nz_common)
    X_c, ZH_c = np.meshgrid(x_common, z_hat_common, indexing="ij")

    error_fields = []
    for r in rows:
        E_val = float(r["E"])
        t_val = float(r["thickness"])

        cfg = {
            "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": t_val, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
            "material": {"E": E_val, "nu": config.nu_vals[0]},
            "load_patch": {
                "pressure": config.p0,
                "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
            },
        }
        x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_fem(cfg)
        x_nodes = np.array(x_nodes)
        y_nodes = np.array(y_nodes)
        z_nodes = np.array(z_nodes)

        mid_y_idx = len(y_nodes) // 2
        x_cross, z_cross = np.meshgrid(x_nodes, z_nodes, indexing="ij")

        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

        pts = np.stack([
            x_cross.ravel(),
            np.full(x_cross.size, y_nodes[mid_y_idx]),
            z_cross.ravel(),
            np.full(x_cross.size, E_val),
            np.full(x_cross.size, t_val),
            np.full(x_cross.size, r_ref),
            np.full(x_cross.size, mu_ref),
            np.full(x_cross.size, v0_ref),
        ], axis=1)

        with torch.no_grad():
            v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()

        alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
        e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
        t_scale = 1.0 if alpha == 0.0 else (float(config.H) / max(1e-8, float(t_val))) ** alpha
        u_pinn = (v / (float(E_val) ** e_pow)) * t_scale
        u_pinn_cross = u_pinn.reshape(len(x_nodes), len(z_nodes), 3)

        u_fea_cross = np.array(u_fea)[:, mid_y_idx, :, 2]
        abs_err_cross = np.abs(u_pinn_cross[:, :, 2] - u_fea_cross)

        # Interpolate to common normalized grid
        z_hat = z_nodes / t_val
        interp = RegularGridInterpolator((x_nodes, z_hat), abs_err_cross, method="linear", bounds_error=False, fill_value=None)
        err_common = interp(np.stack([X_c.ravel(), ZH_c.ravel()], axis=1)).reshape(nx_common, nz_common)
        error_fields.append(err_common)
        print(f"  One-layer case {r['case_id']}: E={E_val:.2f}, t={t_val:.3f}, MAE={float(r['top_uz_mae_pct']):.2f}%")

    mean_error = np.mean(np.stack(error_fields, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.contourf(X_c, ZH_c, mean_error, levels=50, cmap="magma")
    plt.colorbar(im, ax=ax, label="Mean |u_z PINN − u_z FEA|")
    ax.set_xlabel("x")
    ax.set_ylabel("z / H (normalized thickness)")
    ax.set_title(f"One-Layer Averaged Cross-Section Error (y=0.5)\nAveraged over {n_cases} random cases")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "one_layer_averaged_cross_section.pdf", format="pdf", dpi=200)
    fig.savefig(OUT_DIR / "one_layer_averaged_cross_section.png", format="png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'one_layer_averaged_cross_section.pdf'}")
    print(f"Saved: {OUT_DIR / 'one_layer_averaged_cross_section.png'}")


def _three_layer_averaged(n_cases: int = 15, ne_x: int = 16, ne_y: int = 16, ne_z: int = 8):
    three_layer_dir = REPO_ROOT / "three-layer-workflow"
    sys.path.insert(0, str(three_layer_dir))
    for mod in ["model", "pinn_config", "fem_solver"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import pinn_config as config
    import model
    import fem_solver

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    ckpt = three_layer_dir / "pinn_model.pth"
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    csv_path = REPO_ROOT / "graphs" / "generalized_study" / "three_layer_random_50.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows = rows[:n_cases]

    nx_common = 51
    nz_common = 51
    x_common = np.linspace(0, config.Lx, nx_common)
    z_hat_common = np.linspace(0, 1, nz_common)
    X_c, ZH_c = np.meshgrid(x_common, z_hat_common, indexing="ij")

    error_fields = []
    interface_positions = []

    for r in rows:
        e1 = float(r["e1"])
        e2 = float(r["e2"])
        e3 = float(r["e3"])
        t1 = float(r["t1"])
        t2 = float(r["t2"])
        t3 = float(r["t3"])
        thickness = t1 + t2 + t3

        cfg = {
            "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
            "material": {"E_layers": [e1, e2, e3], "t_layers": [t1, t2, t3], "nu": config.nu_vals[0]},
            "load_patch": {
                "pressure": config.p0,
                "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
            },
        }
        x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_three_layer_fem(cfg)
        x_nodes = np.array(x_nodes)
        y_nodes = np.array(y_nodes)
        z_nodes = np.array(z_nodes)

        mid_y_idx = len(y_nodes) // 2
        x_cross, z_cross = np.meshgrid(x_nodes, z_nodes, indexing="ij")

        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        pts = np.stack([
            x_cross.ravel(), np.full(x_cross.size, y_nodes[mid_y_idx]), z_cross.ravel(),
            np.full(x_cross.size, e1), np.full(x_cross.size, t1),
            np.full(x_cross.size, e2), np.full(x_cross.size, t2),
            np.full(x_cross.size, e3), np.full(x_cross.size, t3),
            np.full(x_cross.size, r_ref), np.full(x_cross.size, mu_ref), np.full(x_cross.size, v0_ref),
        ], axis=1)

        e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
        alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
        scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
        h_ref = float(getattr(config, "H", 1.0))

        with torch.no_grad():
            v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()

        e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
        t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
        u_pinn = scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
        u_pinn_cross = u_pinn.reshape(len(x_nodes), len(z_nodes), 3)

        u_fea_cross = np.array(u_fea)[:, mid_y_idx, :, 2]
        abs_err_cross = np.abs(u_pinn_cross[:, :, 2] - u_fea_cross)

        z_hat = z_nodes / thickness
        interp = RegularGridInterpolator((x_nodes, z_hat), abs_err_cross, method="linear", bounds_error=False, fill_value=None)
        err_common = interp(np.stack([X_c.ravel(), ZH_c.ravel()], axis=1)).reshape(nx_common, nz_common)
        error_fields.append(err_common)
        interface_positions.append((t1 / thickness, (t1 + t2) / thickness))
        print(f"  Three-layer case {r['case_id']}: MAE={float(r['top_uz_mae_pct']):.2f}%")

    mean_error = np.mean(np.stack(error_fields, axis=0), axis=0)
    mean_iface1 = float(np.mean([p[0] for p in interface_positions]))
    mean_iface2 = float(np.mean([p[1] for p in interface_positions]))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.contourf(X_c, ZH_c, mean_error, levels=50, cmap="magma")
    plt.colorbar(im, ax=ax, label="Mean |u_z PINN − u_z FEA|")
    ax.axhline(mean_iface1, color="white", linestyle="--", linewidth=1.5)
    ax.axhline(mean_iface2, color="white", linestyle="--", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("z / H (normalized thickness)")
    ax.set_title(f"Three-Layer Averaged Cross-Section Error (y=0.5)\nAveraged over {n_cases} random cases\nMean interfaces at z/H = {mean_iface1:.3f}, {mean_iface2:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "three_layer_averaged_cross_section.pdf", format="pdf", dpi=200)
    fig.savefig(OUT_DIR / "three_layer_averaged_cross_section.png", format="png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'three_layer_averaged_cross_section.pdf'}")
    print(f"Saved: {OUT_DIR / 'three_layer_averaged_cross_section.png'}")


if __name__ == "__main__":
    _one_layer_averaged(n_cases=15, ne_x=16, ne_y=16, ne_z=8)
    _three_layer_averaged(n_cases=15, ne_x=16, ne_y=16, ne_z=8)
