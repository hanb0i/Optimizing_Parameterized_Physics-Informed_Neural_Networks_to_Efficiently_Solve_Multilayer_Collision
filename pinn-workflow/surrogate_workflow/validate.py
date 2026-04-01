import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogate_workflow import config
from surrogate_workflow import baseline
from surrogate_workflow import data as data_utils
from surrogate_workflow import surrogate


def denormalize_y(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def plot_pred_vs_true(y_true, y_pred, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    plt.xlabel("Baseline y")
    plt.ylabel("Surrogate y")
    plt.title("Prediction vs Truth")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_trend(param_name, model, dataset, device, path):
    param_names = dataset["param_names"]
    if param_name not in param_names:
        raise ValueError(f"{param_name} not found in design params.")

    ranges = config.DESIGN_RANGES
    sweep = np.linspace(ranges[param_name][0], ranges[param_name][1], int(config.TREND_SWEEP_POINTS))
    mu_mid = config.mid_design()
    idx = param_names.index(param_name)

    baseline_vals = []
    for val in sweep:
        mu = mu_mid.copy()
        mu[idx] = val
        baseline_vals.append(baseline.compute_response(mu))

    x_raw = np.column_stack(
        [sweep if name == param_name else np.full_like(sweep, mu_mid[i]) for i, name in enumerate(param_names)]
    )
    x_norm, _, _ = data_utils.normalize_inputs(x_raw, ranges)
    y_norm = surrogate.predict(model, x_norm, device)
    surrogate_vals = denormalize_y(y_norm, dataset["y_min"], dataset["y_max"])

    plt.figure(figsize=(7, 4))
    plt.plot(sweep, baseline_vals, "o-", label="Baseline")
    plt.plot(sweep, surrogate_vals, "s--", label="Surrogate")
    plt.xlabel(param_name)
    plt.ylabel("y (peak top displacement)")
    plt.title(f"Trend Fidelity Sweep: {param_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def optimization_safety_check(model, dataset, device):
    ranges = config.DESIGN_RANGES
    params = dataset["param_names"]
    candidates = data_utils.sample_designs(int(config.OPT_CANDIDATES), ranges, int(config.SEED) + 13)
    x_norm, _, _ = data_utils.normalize_inputs(candidates, ranges)
    y_norm = surrogate.predict(model, x_norm, device)
    y_pred = denormalize_y(y_norm, dataset["y_min"], dataset["y_max"])

    best_idx = int(np.argmin(y_pred))
    mu_star = candidates[best_idx]
    pred_star = float(y_pred[best_idx])
    true_star = baseline.compute_response(mu_star)

    mu_mid = config.mid_design()
    baseline_mid = baseline.compute_response(mu_mid)

    return {
        "mu_star": mu_star,
        "pred_star": pred_star,
        "true_star": true_star,
        "baseline_mid": baseline_mid,
    }


def write_summary(path, param_names, metrics, safety_check):
    lines = [
        "Phase 1 Surrogate Summary",
        "",
        f"Design parameters: {', '.join(param_names)}",
        f"Train MSE: {metrics['train_mse']:.6e}",
        f"Val MSE: {metrics['val_mse']:.6e}",
        f"Test MSE: {metrics['test_mse']:.6e}",
        "",
        "Optimization safety check:",
        f"  mu*: {np.array2string(safety_check['mu_star'], precision=3)}",
        f"  surrogate y(mu*): {safety_check['pred_star']:.6e}",
        f"  baseline y(mu*): {safety_check['true_star']:.6e}",
        f"  baseline y(mu_mid): {safety_check['baseline_mid']:.6e}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
