import argparse
import os
import sys

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from surrogate_workflow import config
from surrogate_workflow import data as data_utils
from surrogate_workflow import surrogate
from surrogate_workflow import validate


def prepare_outputs():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def load_or_generate_dataset(regenerate: bool):
    if os.path.exists(config.DATASET_PATH) and not regenerate:
        print(f"Loading dataset from {config.DATASET_PATH}")
        return data_utils.load_dataset(config.DATASET_PATH)

    print("Generating dataset...")
    dataset = data_utils.generate_dataset()
    data_utils.save_dataset(config.DATASET_PATH, dataset)
    print(f"Saved dataset to {config.DATASET_PATH}")
    return dataset


def build_loaders(dataset, seed: int):
    n_samples = dataset["x_norm"].shape[0]
    train_idx, val_idx, test_idx = data_utils.split_indices(
        n_samples,
        float(config.TRAIN_FRACTION),
        float(config.VAL_FRACTION),
        seed,
        n_anchors=int(dataset.get("n_anchors", 0)),
    )

    x = dataset["x_norm"].astype(np.float32)
    y = dataset["y_norm"].astype(np.float32).reshape(-1, 1)
    y_raw = dataset["y_raw"].astype(np.float32).reshape(-1, 1)
    rel_eps = float(getattr(config, "RELATIVE_LOSS_EPS", 1e-3))
    weights = (1.0 / (y_raw ** 2 + rel_eps ** 2)).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]), torch.from_numpy(weights[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]), torch.from_numpy(weights[val_idx]))
    test_ds = TensorDataset(torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx]), torch.from_numpy(weights[test_idx]))

    train_loader = DataLoader(train_ds, batch_size=int(config.BATCH_SIZE), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config.BATCH_SIZE), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(config.BATCH_SIZE), shuffle=False)
    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)


def evaluate_splits(model, dataset, splits, device):
    x_norm = dataset["x_norm"]
    y_raw = dataset["y_raw"]
    y_norm = surrogate.predict(model, x_norm, device)
    y_pred = validate.denormalize_y(
        y_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", "identity"),
        dataset.get("y_eps", 1e-12),
    )

    train_idx, val_idx, test_idx = splits
    return {
        "train_mse": validate.mse(y_raw[train_idx], y_pred[train_idx]),
        "val_mse": validate.mse(y_raw[val_idx], y_pred[val_idx]),
        "test_mse": validate.mse(y_raw[test_idx], y_pred[test_idx]),
        "y_pred": y_pred,
    }


def save_model(model, dataset):
    payload = {
        "state_dict": model.state_dict(),
        "x_min": dataset["x_min"],
        "x_max": dataset["x_max"],
        "y_min": dataset["y_min"],
        "y_max": dataset["y_max"],
        "y_transform": dataset.get("y_transform", "identity"),
        "y_eps": float(dataset.get("y_eps", 1e-12)),
        "param_names": dataset["param_names"],
        "config": {
            "hidden_layers": config.HIDDEN_LAYERS,
            "hidden_units": config.HIDDEN_UNITS,
            "activation": config.ACTIVATION,
            "fourier_dim": int(getattr(config, "FOURIER_DIM", 0)),
            "fourier_scale": float(getattr(config, "FOURIER_SCALE", 1.0)),
        },
    }
    torch.save(payload, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 surrogate pipeline (PINN baseline)")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate baseline dataset")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--n-samples", type=int, default=None, help="Override number of LHS samples")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max training epochs")
    args = parser.parse_args()

    if args.n_samples is not None:
        config.N_SAMPLES = int(args.n_samples)
    if args.max_epochs is not None:
        config.MAX_EPOCHS = int(args.max_epochs)

    prepare_outputs()
    dataset = load_or_generate_dataset(args.regenerate)

    seed = int(config.SEED)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, splits = build_loaders(dataset, seed)
    model = surrogate.MLPRegressor(
        input_dim=len(dataset["param_names"]),
        output_dim=1,
        hidden_layers=int(config.HIDDEN_LAYERS),
        hidden_units=int(config.HIDDEN_UNITS),
        activation=str(config.ACTIVATION),
        fourier_dim=int(getattr(config, "FOURIER_DIM", 0)),
        fourier_scale=float(getattr(config, "FOURIER_SCALE", 1.0)),
    ).to(device)

    model, _history = surrogate.train_model(model, train_loader, val_loader, config, device)
    save_model(model, dataset)

    metrics = evaluate_splits(model, dataset, splits, device)
    pred_plot = os.path.join(config.PLOTS_DIR, "pred_vs_truth.png")
    validate.plot_pred_vs_true(
        dataset["y_raw"][splits[2]],
        metrics["y_pred"][splits[2]],
        pred_plot,
    )
    trend_plot = os.path.join(config.PLOTS_DIR, "trend_fidelity.png")
    validate.plot_trend(config.TREND_SWEEP_PARAM, model, dataset, device, trend_plot)
    safety = validate.optimization_safety_check(model, dataset, device)
    validate.write_summary(config.SUMMARY_PATH, dataset["param_names"], metrics, safety)

    print("Phase 1 complete.")
    print(f"Plots: {config.PLOTS_DIR}")
    print(f"Summary: {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
