from __future__ import annotations

from _common import apply_ieee_style, save_figure

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    apply_ieee_style()

    # Single-column friendly.
    fig, ax = plt.subplots(figsize=(3.6, 3.2))

    # Domain in the top surface (x-y) plane.
    domain = Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, linewidth=1.4, edgecolor="black")
    ax.add_patch(domain)

    # Load patch: x,y in [1/3, 2/3].
    patch = Rectangle(
        (1.0 / 3.0, 1.0 / 3.0),
        1.0 / 3.0,
        1.0 / 3.0,
        facecolor="0.85",
        edgecolor="0.2",
        linewidth=1.0,
    )
    ax.add_patch(patch)

    # Simple traction arrows (schematic).
    for dx in (-0.12, 0.0, 0.12):
        ax.annotate(
            "",
            xy=(0.5 + dx, 0.5 - 0.10),
            xytext=(0.5 + dx, 0.5 + 0.10),
            arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.0),
        )

    ax.text(0.5, 0.70, "Top traction\n(load patch)", ha="center", va="bottom", fontsize=8, color="0.15")

    # Clamped side boundaries (schematic): thicker outline and label.
    for (x0, y0, x1, y1) in [(0, 0, 1, 0), (0, 1, 1, 1), (0, 0, 0, 1), (1, 0, 1, 1)]:
        ax.plot([x0, x1], [y0, y1], color="black", lw=2.2)
    ax.text(1.02, 0.5, "Clamped\nsides", rotation=90, va="center", ha="left", fontsize=8, color="0.15")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Geometry and Boundary Conditions (Top View)")

    out_paths = save_figure(fig, "fig_geometry_bc")
    plt.close(fig)

    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
