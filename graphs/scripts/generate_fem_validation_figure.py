#!/usr/bin/env python3
"""Generate publication-ready FEM solver validation figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication font setup (Times New Roman via matplotlib's built-in serif)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",  # STIX fonts look like Times for math
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    one_data = _load_csv(OUT_DIR / "one_layer_convergence.csv")
    three_data = _load_csv(OUT_DIR / "three_layer_convergence.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Row 0: One-Layer ---
    h_1 = np.array([float(r["h"]) for r in one_data])
    peak_1 = np.array([abs(float(r["peak_uz"])) for r in one_data])
    rel_err_1 = np.array([float(r["rel_err"]) for r in one_data])
    n_elem_1 = np.array([int(r["n_elements"]) for r in one_data])

    # Peak displacement
    ax = axes[0, 0]
    ax.plot(h_1, peak_1, "o-", linewidth=2, markersize=10, color="#1f77b4", label="FEM solution")
    ax.axvline(0.125, color="red", linestyle="--", alpha=0.7, label="Benchmark mesh (8×8×4)")
    ax.set_xlabel(r"Mesh size $h = 1/n_{e,x}$")
    ax.set_ylabel(r"$|\max u_z|$")
    ax.set_title(r"(a) One-Layer: Peak Displacement vs Mesh Size" + "\n" + r"$E=10$, $t=0.05$")
    ax.invert_xaxis()
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    # Annotate benchmark point — text placed outside plot area
    bench_idx = 1  # 8x8x4
    ax.annotate(
        rf"Benchmark: $|u_z|={peak_1[bench_idx]:.3f}$",
        xy=(h_1[bench_idx], peak_1[bench_idx]),
        xytext=(0.02, 0.75),
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7, lw=1.2),
        fontsize=10,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9),
    )

    # Convergence rate
    ax = axes[0, 1]
    ax.loglog(h_1[:-1], rel_err_1[:-1], "o-", linewidth=2, markersize=10, color="#1f77b4", label="FEM convergence")
    h_ref = h_1[:-1]
    ax.loglog(h_ref, h_ref**2 * rel_err_1[0] / h_1[0]**2, "k--", alpha=0.5, label="O(h²) reference")
    ax.axvline(0.125, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Mesh size $h$")
    ax.set_ylabel(r"Relative error vs finest mesh")
    ax.set_title(r"(b) One-Layer: Convergence Rate")
    ax.invert_xaxis()
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # --- Row 1: Three-Layer ---
    h_3 = np.array([float(r["h"]) for r in three_data])
    peak_3 = np.array([abs(float(r["peak_uz"])) for r in three_data])
    rel_err_3 = np.array([float(r["rel_err"]) for r in three_data])
    n_elem_3 = np.array([int(r["n_elements"]) for r in three_data])

    # Peak displacement
    ax = axes[1, 0]
    ax.plot(h_3, peak_3, "o-", linewidth=2, markersize=10, color="#2ca02c", label="FEM solution")
    ax.axvline(0.125, color="red", linestyle="--", alpha=0.7, label="Benchmark mesh (8×8×4)")
    ax.set_xlabel(r"Mesh size $h = 1/n_{e,x}$")
    ax.set_ylabel(r"$|\max u_z|$")
    ax.set_title(r"(c) Three-Layer: Peak Displacement vs Mesh Size" + "\n" + r"$E=[10,10,10]$, $t=[0.02,0.10,0.02]$")
    ax.invert_xaxis()
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    bench_idx = 1
    ax.annotate(
        rf"Benchmark: $|u_z|={peak_3[bench_idx]:.4f}$",
        xy=(h_3[bench_idx], peak_3[bench_idx]),
        xytext=(0.02, 0.75),
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7, lw=1.2),
        fontsize=10,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9),
    )

    # Convergence rate
    ax = axes[1, 1]
    ax.loglog(h_3[:-1], rel_err_3[:-1], "o-", linewidth=2, markersize=10, color="#2ca02c", label="FEM convergence")
    h_ref = h_3[:-1]
    ax.loglog(h_ref, h_ref**2 * rel_err_3[0] / h_3[0]**2, "k--", alpha=0.5, label="O(h²) reference")
    ax.axvline(0.125, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Mesh size $h$")
    ax.set_ylabel(r"Relative error vs finest mesh")
    ax.set_title(r"(d) Three-Layer: Convergence Rate")
    ax.invert_xaxis()
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "FEM Solver Validation: Mesh Convergence Study\n"
        r"Benchmark mesh ($8 \times 8 \times 4$) indicated by red dashed line",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(OUT_DIR / "fem_validation_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fem_validation_combined.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'fem_validation_combined.png'}")
    print(f"Saved: {OUT_DIR / 'fem_validation_combined.pdf'}")

    # Also generate a clean summary table
    table_lines = [
        "# FEM Solver Validation Summary",
        "",
        "## One-Layer (E=10, t=0.05)",
        "",
        "| Mesh | Elements | Peak |u_z| | Δ vs 32³ |",
        "|------|----------|-------------|----------|",
    ]
    for r in one_data:
        peak = abs(float(r["peak_uz"]))
        finest_peak = abs(float(one_data[-1]["peak_uz"]))
        delta = peak - finest_peak
        table_lines.append(f"| {r['ne_x']}×{r['ne_y']}×{r['ne_z']} | {r['n_elements']} | {peak:.6f} | {delta:+.6f} |")

    table_lines.extend([
        "",
        "## Three-Layer (E=[10,10,10], t=[0.02,0.10,0.02])",
        "",
        "| Mesh | Elements | Peak |u_z| | Δ vs 32³ |",
        "|------|----------|-------------|----------|",
    ])
    for r in three_data:
        peak = abs(float(r["peak_uz"]))
        finest_peak = abs(float(three_data[-1]["peak_uz"]))
        delta = peak - finest_peak
        table_lines.append(f"| {r['ne_x']}×{r['ne_y']}×{r['ne_z']} | {r['n_elements']} | {peak:.6f} | {delta:+.6f} |")

    table_lines.extend([
        "",
        "**Note:** Benchmark mesh (8×8×4) used for all PINN-vs-FEM comparisons. "
        "Absolute differences in physical units are small; PINN and FEM use identical meshes for consistency.",
    ])

    md_path = OUT_DIR / "fem_validation_summary.md"
    md_path.write_text("\n".join(table_lines))
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
