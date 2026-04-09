from __future__ import annotations

import csv
from pathlib import Path

from _common import DATA_DIR, apply_ieee_style, save_figure

import matplotlib.pyplot as plt


VARIANTS = [
    "Base parametric PINN",
    "+ Compliance-aware scaling",
    "+ Layerwise PDE decomposition",
    "+ Interface continuity enforcement",
    "+ Sparse FEM supervision",
    "Full framework",
]


def _load_results(csv_path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = (row.get("variant") or "").strip()
            if not variant:
                continue
            try:
                mean_mae = float(row["mean_mae"])
                worst_mae = float(row["worst_mae"])
            except Exception:
                continue
            out[variant] = (mean_mae, worst_mae)
    return out


def main() -> None:
    apply_ieee_style()

    results_path = DATA_DIR / "ablation_results.csv"
    results = _load_results(results_path) if results_path.exists() else {}

    cell_text = []
    for variant in VARIANTS:
        if variant in results:
            mean_mae, worst_mae = results[variant]
            mean_s = f"{mean_mae:.2f}"
            worst_s = f"{worst_mae:.2f}"
        else:
            mean_s = "TBD"
            worst_s = "TBD"
        cell_text.append([variant, mean_s, worst_s])

    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=["Variant", "Mean MAE (%)", "Worst MAE (%)"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.62, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    # Header styling.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.95")

    ax.set_title("Ablation Study of Proposed Framework Components", pad=10)

    out_paths = save_figure(fig, "fig_ablation_table")
    plt.close(fig)

    if results:
        print(f"Loaded results from: {results_path}")
    else:
        print("No ablation results found; rendered a TBD placeholder table.")
        print(f"Expected CSV: {results_path}")
        print("Schema: variant,mean_mae,worst_mae (values in percent)")

    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
