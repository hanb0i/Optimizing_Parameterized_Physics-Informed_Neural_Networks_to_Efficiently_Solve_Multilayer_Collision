from __future__ import annotations

import os
import sys
from pathlib import Path

from _common import REPO_ROOT, apply_ieee_style, save_figure, watermark_placeholder, print_inputs_used

import numpy as np

import matplotlib.pyplot as plt
import torch


def _try_import_surrogate_modules():
    sw_dir = REPO_ROOT / "pinn-workflow"
    if str(sw_dir) not in sys.path:
        sys.path.insert(0, str(sw_dir))
    from surrogate_workflow import config, data as data_utils, surrogate, validate  # noqa: WPS433
    return config, data_utils, surrogate, validate


def main() -> None:
    apply_ieee_style()

    outputs_dir = REPO_ROOT / "pinn-workflow" / "surrogate_workflow" / "outputs"
    dataset_path = outputs_dir / "phase1_dataset.npz"
    model_path = outputs_dir / "surrogate_model.pt"

    fig, ax = plt.subplots(figsize=(3.6, 3.2))

    if not dataset_path.exists() or not model_path.exists():
        ax.set_title("Surrogate Verification (Predicted vs Reference)")
        ax.set_xlabel("Reference (PINN) peak deflection")
        ax.set_ylabel("Surrogate predicted peak deflection")
        watermark_placeholder(ax, "PLACEHOLDER\n(run surrogate workflow to generate outputs)")
        ax.text(
            0.02,
            0.02,
            f"Missing:\n- {dataset_path}\n- {model_path}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
        out_paths = save_figure(fig, "fig_surrogate_scatter_placeholder")
        plt.close(fig)
        print("Wrote placeholder surrogate verification figure.")
        for p in out_paths:
            print(f"  - {p}")
        return

    config, data_utils, surrogate, validate = _try_import_surrogate_modules()
    print_inputs_used([dataset_path, model_path])

    dataset = data_utils.load_dataset(str(dataset_path))
    try:
        payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(str(model_path), map_location="cpu", weights_only=False)

    model = surrogate.MLPRegressor(
        input_dim=len(payload["param_names"]),
        output_dim=1,
        hidden_layers=int(payload["config"]["hidden_layers"]),
        hidden_units=int(payload["config"]["hidden_units"]),
        activation=str(payload["config"]["activation"]),
        fourier_dim=int(payload["config"].get("fourier_dim", 0)),
        fourier_scale=float(payload["config"].get("fourier_scale", 1.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    n_samples = int(dataset["x_norm"].shape[0])
    train_idx, val_idx, test_idx = data_utils.split_indices(
        n_samples,
        float(getattr(config, "TRAIN_FRACTION", 0.8)),
        float(getattr(config, "VAL_FRACTION", 0.1)),
        int(getattr(config, "SEED", 7)),
        n_anchors=int(dataset.get("n_anchors", 0)),
    )

    with torch.no_grad():
        y_norm = surrogate.predict(model, dataset["x_norm"], torch.device("cpu"))

    y_pred = validate.denormalize_y(
        y_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", payload.get("y_transform", "identity")),
        dataset.get("y_eps", payload.get("y_eps", 1e-12)),
    )
    y_true = np.asarray(dataset["y_raw"], dtype=float)

    y_true_test = y_true[test_idx]
    y_pred_test = np.asarray(y_pred, dtype=float)[test_idx]

    # Scatter + 45-degree line.
    ax.scatter(y_true_test, y_pred_test, s=10, alpha=0.75, color="tab:blue", edgecolors="none")
    lo = float(min(np.min(y_true_test), np.min(y_pred_test)))
    hi = float(max(np.max(y_true_test), np.max(y_pred_test)))
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, linestyle="--", label="y=x")

    rel_err = np.abs(y_true_test - y_pred_test) / (np.abs(y_true_test) + 1e-12)
    worst = float(np.max(rel_err) * 100.0) if rel_err.size else float("nan")
    ax.text(0.02, 0.98, f"Test worst rel err: {worst:.2f}%", transform=ax.transAxes, ha="left", va="top")

    ax.set_title("Surrogate Verification (Predicted vs Reference)")
    ax.set_xlabel("Reference (PINN) peak deflection")
    ax.set_ylabel("Surrogate predicted peak deflection")
    ax.legend(frameon=False, loc="lower right")

    out_paths = save_figure(fig, "fig_surrogate_scatter")
    plt.close(fig)

    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
