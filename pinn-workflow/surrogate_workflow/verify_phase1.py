import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from surrogate_workflow import config
from surrogate_workflow import baseline
from surrogate_workflow import data as data_utils
from surrogate_workflow import surrogate
from surrogate_workflow import validate


def _load_trained_model(device: torch.device):
    try:
        payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model = surrogate.MLPRegressor(
        input_dim=len(payload["param_names"]),
        output_dim=1,
        hidden_layers=int(payload["config"]["hidden_layers"]),
        hidden_units=int(payload["config"]["hidden_units"]),
        activation=str(payload["config"]["activation"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)


def _plot_rel_error_hist(rel_err: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 4))
    plt.hist(rel_err * 100.0, bins=40, alpha=0.9, color="tab:blue")
    plt.xlabel("Relative error (%)")
    plt.ylabel("Count")
    plt.title("Surrogate Relative Error (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_e1_e2_grid(model, dataset: dict, device: torch.device, out_path: str, n: int):
    e1_low, e1_high = config.DESIGN_RANGES["E1"]
    e2_low, e2_high = config.DESIGN_RANGES["E2"]
    e1_vals = np.linspace(e1_low, e1_high, n)
    e2_vals = np.linspace(e2_low, e2_high, n)

    mu_mid = config.mid_design()
    t1_mid = float(mu_mid[config.DESIGN_PARAMS.index("t1")])
    t2_mid = float(mu_mid[config.DESIGN_PARAMS.index("t2")])

    y_true = np.zeros((n, n), dtype=float)
    y_pred = np.zeros((n, n), dtype=float)
    for i, e1 in enumerate(e1_vals):
        for j, e2 in enumerate(e2_vals):
            mu = np.array([e1, t1_mid, e2, t2_mid], dtype=float)
            y_true[i, j] = baseline.compute_response(mu)
            x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), config.DESIGN_RANGES)
            y_norm = surrogate.predict(model, x_norm, device)[0]
            y_pred[i, j] = validate.denormalize_y(y_norm, dataset["y_min"], dataset["y_max"])

    abs_err = np.abs(y_pred - y_true)
    rel_err = _relative_error(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis")
    axes[0].set_title("Baseline (PINN) y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis")
    axes[1].set_title("Surrogate y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(rel_err * 100.0, origin="lower", cmap="magma")
    axes[2].set_title("Rel error (%)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        ax.set_xticklabels([f"{e2_low:g}", f"{e2_high:g}"])
        ax.set_yticklabels([f"{e1_low:g}", f"{e1_high:g}"])
        ax.set_xlabel("E2")
        ax.set_ylabel("E1")

    fig.suptitle(f"Mid-thickness grid: t1={t1_mid:.3f}, t2={t2_mid:.3f}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 surrogate outputs")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--grid", type=int, default=11, help="E1/E2 grid resolution for verification plot")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    dataset = data_utils.load_dataset(config.DATASET_PATH)
    model, _payload = _load_trained_model(device)

    n_samples = dataset["x_norm"].shape[0]
    train_idx, val_idx, test_idx = data_utils.split_indices(
        n_samples, float(config.TRAIN_FRACTION), float(config.VAL_FRACTION), int(config.SEED)
    )

    y_norm = surrogate.predict(model, dataset["x_norm"], device)
    y_pred = validate.denormalize_y(y_norm, dataset["y_min"], dataset["y_max"])
    y_true = dataset["y_raw"]

    rel_err = _relative_error(y_true[test_idx], y_pred[test_idx])
    mae = float(np.mean(np.abs(y_true[test_idx] - y_pred[test_idx])))
    p50 = float(np.percentile(rel_err, 50) * 100.0)
    p95 = float(np.percentile(rel_err, 95) * 100.0)
    worst = float(np.max(rel_err) * 100.0)
    print(f"Test MAE: {mae:.6e}")
    print(f"Test rel error p50/p95/worst: {p50:.2f}% / {p95:.2f}% / {worst:.2f}%")

    _plot_rel_error_hist(rel_err, os.path.join(config.PLOTS_DIR, "test_rel_error_hist.png"))
    _plot_e1_e2_grid(
        model,
        dataset,
        device,
        os.path.join(config.PLOTS_DIR, "grid_e1_e2_mid_thickness.png"),
        n=int(args.grid),
    )

    print(f"Wrote plots to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
