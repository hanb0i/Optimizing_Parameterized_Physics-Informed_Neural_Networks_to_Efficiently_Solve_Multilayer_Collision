import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_fea_solution(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if {"x_nodes", "y_nodes", "z_nodes", "u_grid"}.issubset(data.files):
            x_nodes = data["x_nodes"]
            y_nodes = data["y_nodes"]
            z_nodes = data["z_nodes"]
            u_grid = data["u_grid"]
        elif {"x", "y", "z", "u"}.issubset(data.files):
            x_nodes, y_nodes, z_nodes, u_grid = _coerce_fea_fields(
                data["x"], data["y"], data["z"], data["u"]
            )
        else:
            raise KeyError(f"Missing expected keys in npz file: {data.files}")
        return x_nodes, y_nodes, z_nodes, u_grid

    if hasattr(data, "item"):
        data = data.item()
        if {"x_nodes", "y_nodes", "z_nodes", "u_grid"}.issubset(data):
            return data["x_nodes"], data["y_nodes"], data["z_nodes"], data["u_grid"]
        if {"x", "y", "z", "u"}.issubset(data):
            return _coerce_fea_fields(data["x"], data["y"], data["z"], data["u"])
        raise KeyError(f"Missing expected keys in dict: {sorted(data.keys())}")

    raise ValueError(f"Unrecognized FEA solution format at {path}")


def _coerce_fea_fields(x, y, z, u):
    if x.ndim == y.ndim == z.ndim == 1:
        return x, y, z, u
    if x.ndim == y.ndim == z.ndim == 3:
        x_nodes = x[:, 0, 0]
        y_nodes = y[0, :, 0]
        z_nodes = z[0, 0, :]
        return x_nodes, y_nodes, z_nodes, u
    raise ValueError(
        "Expected x/y/z as 1D node arrays or 3D grids matching u."
    )


def find_mid_index(arr):
    return int(np.argmin(np.abs(arr - 0.5 * (arr[0] + arr[-1]))))


def save_contour(x_nodes, y_nodes, field, title, path):
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, field, levels=60, cmap="jet")
    plt.colorbar(cs)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_line(x_vals, y_vals, title, path, xlabel="x", ylabel="u_z"):
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, "k-")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def diagnostics(x_nodes, y_nodes, z_nodes, u_grid):
    # Basic stats
    u_min = u_grid.min(axis=(0, 1, 2))
    u_max = u_grid.max(axis=(0, 1, 2))

    # Clamped side check (x=0, x=Lx, y=0, y=Ly)
    x0 = u_grid[0, :, :, :]
    xL = u_grid[-1, :, :, :]
    y0 = u_grid[:, 0, :, :]
    yL = u_grid[:, -1, :, :]
    clamp_rms = np.sqrt(np.mean(np.concatenate([x0, xL, y0, yL], axis=0) ** 2))

    # Symmetry checks on top surface (expected symmetry for centered patch)
    k_top = len(z_nodes) - 1
    uz_top = u_grid[:, :, k_top, 2]
    uz_flip_x = np.flip(uz_top, axis=0)
    uz_flip_y = np.flip(uz_top, axis=1)
    symm_rms_x = np.sqrt(np.mean((uz_top - uz_flip_x) ** 2))
    symm_rms_y = np.sqrt(np.mean((uz_top - uz_flip_y) ** 2))

    return {
        "u_min": u_min,
        "u_max": u_max,
        "clamp_rms": clamp_rms,
        "symm_rms_x": symm_rms_x,
        "symm_rms_y": symm_rms_y,
    }

def _get_pinn_config():
    repo_root = Path(__file__).resolve().parents[2]
    pinn_path = repo_root / "pinn-workflow"
    sys.path.insert(0, str(pinn_path))
    import pinn_config as config  # noqa: E402
    sys.path.pop(0)
    return config


def _lame_params_from_config(config):
    return config.Lame_Params, config.Layer_Interfaces, config.Lx, config.Ly, config.H, config.p0


def _finite_gradients(u_grid, dx, dy, dz):
    # u_grid: (nx, ny, nz, 3)
    grads = []
    for i in range(3):
        du_dx, du_dy, du_dz = np.gradient(
            u_grid[:, :, :, i], dx, dy, dz, edge_order=2
        )
        grads.append((du_dx, du_dy, du_dz))
    return grads


def _strain_tensor(grads):
    # grads: list of (du_dx, du_dy, du_dz) for ux, uy, uz
    nx, ny, nz = grads[0][0].shape
    eps = np.zeros((nx, ny, nz, 3, 3))
    for i in range(3):
        for j in range(3):
            if i == 0:
                di = grads[0][j]
            elif i == 1:
                di = grads[1][j]
            else:
                di = grads[2][j]
            if j == 0:
                dj = grads[0][i]
            elif j == 1:
                dj = grads[1][i]
            else:
                dj = grads[2][i]
            eps[:, :, :, i, j] = 0.5 * (di + dj)
    return eps


def _stress_tensor(eps, lam, mu):
    trace = eps[:, :, :, 0, 0] + eps[:, :, :, 1, 1] + eps[:, :, :, 2, 2]
    nx, ny, nz = trace.shape
    sigma = np.zeros((nx, ny, nz, 3, 3))
    for i in range(3):
        for j in range(3):
            sigma[:, :, :, i, j] = 2.0 * mu * eps[:, :, :, i, j]
    for i in range(3):
        sigma[:, :, :, i, i] += lam * trace
    return sigma


def _divergence_sigma(sigma, dx, dy, dz):
    nx, ny, nz = sigma.shape[:3]
    div = np.zeros((nx, ny, nz, 3))
    for i in range(3):
        d0_dx = np.gradient(sigma[:, :, :, i, 0], dx, axis=0, edge_order=2)
        d1_dy = np.gradient(sigma[:, :, :, i, 1], dy, axis=1, edge_order=2)
        d2_dz = np.gradient(sigma[:, :, :, i, 2], dz, axis=2, edge_order=2)
        div[:, :, :, i] = d0_dx + d1_dy + d2_dz
    return div


def pinn_style_loss(x_nodes, y_nodes, z_nodes, u_grid):
    config = _get_pinn_config()
    lame_params, interfaces, Lx, Ly, H, p0 = _lame_params_from_config(config)

    dx = x_nodes[1] - x_nodes[0] if len(x_nodes) > 1 else 1.0
    dy = y_nodes[1] - y_nodes[0] if len(y_nodes) > 1 else 1.0
    dz = z_nodes[1] - z_nodes[0] if len(z_nodes) > 1 else 1.0

    grads = _finite_gradients(u_grid, dx, dy, dz)
    eps = _strain_tensor(grads)

    # Layered material parameters by z
    z_vals = z_nodes
    layer_masks = [
        (z_vals >= interfaces[0]) & (z_vals <= interfaces[1]),
        (z_vals >= interfaces[1]) & (z_vals <= interfaces[2]),
        (z_vals >= interfaces[2]) & (z_vals <= interfaces[3]),
    ]

    pde_loss = 0.0
    for layer_idx, mask in enumerate(layer_masks):
        if not np.any(mask):
            continue
        lam, mu = lame_params[layer_idx]
        sigma = _stress_tensor(eps[:, :, mask, :, :], lam, mu)
        div = _divergence_sigma(sigma, dx, dy, dz)
        residual = -div
        pde_loss += np.mean(residual ** 2)

    # BC loss: clamped sides
    sides = np.concatenate(
        [
            u_grid[0, :, :, :],
            u_grid[-1, :, :, :],
            u_grid[:, 0, :, :],
            u_grid[:, -1, :, :],
        ],
        axis=0,
    )
    bc_loss = np.mean(sides ** 2)

    # Traction on top
    top_idx = len(z_nodes) - 1
    lam3, mu3 = lame_params[2]
    sigma_top = _stress_tensor(eps[:, :, top_idx:top_idx + 1, :, :], lam3, mu3)
    T_top = sigma_top[:, :, 0, :, 2]

    # Patch mask
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    in_patch = (
        (X >= Lx / 3.0)
        & (X <= 2.0 * Lx / 3.0)
        & (Y >= Ly / 3.0)
        & (Y <= 2.0 * Ly / 3.0)
    )
    target = np.zeros_like(T_top)
    target[:, :, 2] = -p0

    load_loss = np.mean((T_top[in_patch] - target[in_patch]) ** 2)
    free_loss = np.mean((T_top[~in_patch]) ** 2)

    # Interface displacement continuity
    if_loss = 0.0
    for z_if in interfaces[1:-1]:
        k = int(np.argmin(np.abs(z_nodes - z_if)))
        u_below = u_grid[:, :, k - 1, :]
        u_above = u_grid[:, :, k, :]
        if_loss += np.mean((u_below - u_above) ** 2)

    return {
        "pde": pde_loss,
        "bc_sides": bc_loss,
        "load": load_loss,
        "free_top": free_loss,
        "interface": if_loss,
    }

def main():
    parser = argparse.ArgumentParser(description="FEA diagnostics and plots.")
    parser.add_argument(
        "--solution",
        default="fea_solution.npy",
        help="Path to fea_solution.npy saved by the solver.",
    )
    parser.add_argument(
        "--out",
        default="fea-workflow/postprocessing/outputs",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    x_nodes, y_nodes, z_nodes, u_grid = load_fea_solution(args.solution)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = diagnostics(x_nodes, y_nodes, z_nodes, u_grid)
    pinn_loss = pinn_style_loss(x_nodes, y_nodes, z_nodes, u_grid)

    # Plots
    k_top = len(z_nodes) - 1
    k_mid = find_mid_index(z_nodes)
    j_mid = find_mid_index(y_nodes)

    uz_top = u_grid[:, :, k_top, 2]
    uz_mid = u_grid[:, :, k_mid, 2]
    uz_centerline = u_grid[:, j_mid, k_top, 2]

    save_contour(x_nodes, y_nodes, uz_top, "Top Surface u_z", out_dir / "uz_top.png")
    save_contour(x_nodes, y_nodes, uz_mid, "Mid-Plane u_z", out_dir / "uz_mid.png")
    save_line(
        x_nodes,
        uz_centerline,
        "Top Centerline u_z (y=Ly/2)",
        out_dir / "uz_centerline.png",
    )

    # Diagnostic report
    report_path = out_dir / "fea_diagnostics.txt"
    with open(report_path, "w", encoding="ascii") as f:
        f.write("FEA Diagnostics\n")
        f.write("----------------\n")
        f.write(f"u_min (ux, uy, uz): {stats['u_min']}\n")
        f.write(f"u_max (ux, uy, uz): {stats['u_max']}\n")
        f.write(f"clamp_rms: {stats['clamp_rms']:.6e}\n")
        f.write(f"symm_rms_x (top uz): {stats['symm_rms_x']:.6e}\n")
        f.write(f"symm_rms_y (top uz): {stats['symm_rms_y']:.6e}\n")
        f.write("\nPINN-Style Loss (FEA Field)\n")
        f.write("---------------------------\n")
        f.write(f"pde: {pinn_loss['pde']:.6e}\n")
        f.write(f"bc_sides: {pinn_loss['bc_sides']:.6e}\n")
        f.write(f"load: {pinn_loss['load']:.6e}\n")
        f.write(f"free_top: {pinn_loss['free_top']:.6e}\n")
        f.write(f"interface: {pinn_loss['interface']:.6e}\n")


    print(f"Saved plots and report to {out_dir}")


if __name__ == "__main__":
    main()
