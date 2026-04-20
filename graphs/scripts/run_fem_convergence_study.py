#!/usr/bin/env python3
"""FEM mesh convergence study for one-layer and three-layer models."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
DATA_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))

import fem_solver


def _run_one_layer_fem(E_val: float, thickness: float, ne_x: int, ne_y: int, ne_z: int):
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": E_val, "nu": 0.3},
        "load_patch": {
            "pressure": 1.0,
            "x_start": 1.0 / 3.0,
            "x_end": 2.0 / 3.0,
            "y_start": 1.0 / 3.0,
            "y_end": 2.0 / 3.0,
        },
    }
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
    u_z_top = np.array(u_grid, dtype=float)[:, :, -1, 2]
    peak_uz = float(np.min(u_z_top))
    return peak_uz


def _run_three_layer_fem(e1, e2, e3, t1, t2, t3, ne_x: int, ne_y: int, ne_z: int):
    thickness = t1 + t2 + t3
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {
            "pressure": 1.0,
            "x_start": 1.0 / 3.0,
            "x_end": 2.0 / 3.0,
            "y_start": 1.0 / 3.0,
            "y_end": 2.0 / 3.0,
        },
    }
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_three_layer_fem(cfg)
    u_z_top = np.array(u_grid, dtype=float)[:, :, -1, 2]
    peak_uz = float(np.min(u_z_top))
    return peak_uz


def _convergence_study(name: str, run_fn, mesh_sequence: list[tuple[int, int, int]]):
    results = []
    print(f"\n=== {name} ===")
    for ne_x, ne_y, ne_z in mesh_sequence:
        print(f"  Running mesh {ne_x}x{ne_y}x{ne_z} ...", end=" ", flush=True)
        peak_uz = run_fn(ne_x, ne_y, ne_z)
        h = 1.0 / ne_x  # characteristic mesh size
        n_elements = ne_x * ne_y * ne_z
        results.append({"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z, "h": h, "n_elements": n_elements, "peak_uz": peak_uz})
        print(f"peak_uz={peak_uz:.6f}")

    # Compute relative error vs finest mesh
    finest_peak = results[-1]["peak_uz"]
    for r in results:
        r["rel_err"] = abs((r["peak_uz"] - finest_peak) / finest_peak) if finest_peak != 0 else 0.0

    # Save CSV
    csv_path = DATA_DIR / f"{name}_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ne_x", "ne_y", "ne_z", "h", "n_elements", "peak_uz", "rel_err"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    h_vals = np.array([r["h"] for r in results])
    peak_vals = np.array([r["peak_uz"] for r in results])
    rel_err_vals = np.array([r["rel_err"] for r in results])

    # Peak displacement vs h
    axes[0].plot(h_vals, np.abs(peak_vals), "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Mesh size h = 1/ne_x")
    axes[0].set_ylabel("|Peak u_z|")
    axes[0].set_title(f"{name}: Peak Displacement vs Mesh Size")
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)

    # Relative error vs h (log-log)
    axes[1].loglog(h_vals[:-1], rel_err_vals[:-1], "o-", linewidth=2, markersize=8, label="FEM convergence")
    # Reference slope lines
    h_ref = h_vals[:-1]
    axes[1].loglog(h_ref, h_ref**2 * rel_err_vals[0] / h_vals[0]**2, "k--", alpha=0.5, label="O(h²) reference")
    axes[1].set_xlabel("Mesh size h")
    axes[1].set_ylabel("Relative error vs finest mesh")
    axes[1].set_title(f"{name}: Convergence Rate")
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig_path = FIG_DIR / f"{name}_convergence.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return results


def main():
    # Mesh sequence: coarse → fine (64 mesh omitted due to memory constraints)
    mesh_sequence = [
        (4, 4, 2),
        (8, 8, 4),
        (16, 16, 8),
        (32, 32, 16),
    ]

    # One-layer: representative stiff-thin case
    _convergence_study(
        "one_layer",
        lambda nx, ny, nz: _run_one_layer_fem(E_val=10.0, thickness=0.05, ne_x=nx, ne_y=ny, ne_z=nz),
        mesh_sequence,
    )

    # Three-layer: representative worst-case from grid sweep
    _convergence_study(
        "three_layer",
        lambda nx, ny, nz: _run_three_layer_fem(
            e1=10.0, e2=10.0, e3=10.0,
            t1=0.02, t2=0.10, t3=0.02,
            ne_x=nx, ne_y=ny, ne_z=nz,
        ),
        mesh_sequence,
    )

    print("\nConvergence study complete.")


if __name__ == "__main__":
    main()
