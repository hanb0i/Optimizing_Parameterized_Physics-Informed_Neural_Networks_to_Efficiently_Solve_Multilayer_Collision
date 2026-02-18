import argparse
import os
import sys
import io
import hashlib
import platform
from contextlib import redirect_stdout

import numpy as np
import torch

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from surrogate_workflow import config
from surrogate_workflow import baseline
from surrogate_workflow import data as data_utils
from surrogate_workflow import surrogate
from surrogate_workflow import validate

FEA_DIR = os.path.join(config.ROOT_DIR, "fea-workflow")
if FEA_DIR not in sys.path:
    sys.path.append(FEA_DIR)

from solver.fem_solver import solve_fem  # noqa: E402


DEFAULT_MESHES = [
    {"ne_x": 20, "ne_y": 20, "ne_z": 6},
    {"ne_x": 30, "ne_y": 30, "ne_z": 9},
    {"ne_x": 40, "ne_y": 40, "ne_z": 12},
    {"ne_x": 50, "ne_y": 50, "ne_z": 15},
]


def _report(lines, message):
    print(message)
    lines.append(message)


def _format_mesh(mesh):
    return f"{mesh['ne_x']}x{mesh['ne_y']}x{mesh['ne_z']}"


def _mesh_elements(mesh):
    return mesh["ne_x"] * mesh["ne_y"] * mesh["ne_z"]


def _interp_top_surface(x_ref, y_ref, uz_ref, x, y):
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Bilinear interpolation via sequential 1D passes.
    interp_x = np.empty((x.shape[0], y_ref.shape[0]))
    for j in range(y_ref.shape[0]):
        interp_x[:, j] = np.interp(x, x_ref, uz_ref[:, j])
    interp_xy = np.empty((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        interp_xy[i, :] = np.interp(y, y_ref, interp_x[i, :])
    return interp_xy


def _call_with_output(lines, func, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    output = buf.getvalue().splitlines()
    for line in output:
        if line.strip():
            _report(lines, line)
    return result


def describe_problem(lines):
    _report(lines, "Step 1 — PDE + Problem Definition")
    _report(lines, "PDE: -div(sigma(u)) = 0 (linear elasticity)")
    _report(lines, "strain: eps = 0.5*(grad(u) + grad(u)^T)")
    _report(lines, "stress: sigma = lambda*tr(eps)*I + 2*mu*eps")
    _report(lines, f"Domain: [0, {config.GEOMETRY['Lx']} ] x [0, {config.GEOMETRY['Ly']}] x [0, {config.GEOMETRY['H']}]")
    _report(lines, f"Layer interfaces: {config.LAYER_INTERFACES}")
    _report(lines, "BCs: clamped on x=0,Lx and y=0,Ly; traction -p0 on top patch; top free elsewhere.")
    _report(lines, f"Load patch: {config.LOAD_PATCH}")
    _report(lines, f"Design params: {config.DESIGN_PARAMS} in ranges {config.DESIGN_RANGES}")
    _report(lines, "Well-posed: clamped sides remove rigid-body modes, traction load defines response.")
    _report(lines, "")


def report_reproducibility(lines):
    _report(lines, "Step 0 — Reproducibility")
    _report(lines, f"Python: {platform.python_version()}")
    _report(lines, f"NumPy: {np.__version__}")
    _report(lines, f"PyTorch: {torch.__version__}")
    _report(lines, f"Seed: {config.SEED}")
    if os.path.exists(config.DATASET_PATH):
        with open(config.DATASET_PATH, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        _report(lines, f"Dataset SHA256: {digest}")
    else:
        _report(lines, "Dataset SHA256: (dataset not found)")
    _report(lines, f"Normalization: inputs min-max by DESIGN_RANGES, outputs min-max")
    _report(lines, "")


def mesh_sweep(lines, meshes):
    _report(lines, "Step 2 — Baseline FEA Mesh Sweep")
    ranges = config.DESIGN_RANGES
    mu_mid = np.array([(ranges[name][0] + ranges[name][1]) * 0.5 for name in config.DESIGN_PARAMS])
    cfg = baseline.build_cfg(mu_mid)
    results = []
    for mesh in meshes:
        cfg_mesh = dict(cfg)
        cfg_mesh["mesh"] = mesh
        _report(lines, f"Running mesh {_format_mesh(mesh)}")
        x, y, z, u_grid = _call_with_output(lines, solve_fem, cfg_mesh)
        uz_top = u_grid[:, :, -1, 2]
        peak = float(-np.min(uz_top))
        results.append({
            "mesh": mesh,
            "x": x,
            "y": y,
            "uz_top": uz_top,
            "peak": peak,
        })
        _report(lines, f"  peak top |u_z| = {peak:.6e}")

    ordered = sorted(results, key=lambda item: _mesh_elements(item["mesh"]))
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]["peak"]
        curr = ordered[i]["peak"]
        rel = abs(curr - prev) / max(prev, 1e-12)
        _report(lines, f"  rel change {_format_mesh(ordered[i-1]['mesh'])} -> {_format_mesh(ordered[i]['mesh'])}: {rel:.2%}")

    if ordered:
        ref = ordered[-1]
        ref_peak = ref["peak"]
        _report(lines, "Mesh convergence (errors vs finest resolution)")
        _report(lines, "  (sorted by element count; errors normalized by reference norms)")
        _report(lines, f"Reference mesh: {_format_mesh(ref['mesh'])}")
        rel_peak_errors = []
        rel_l2_errors = []
        for item in ordered:
            if item is ref:
                rel_peak = 0.0
                rel_l2 = 0.0
            else:
                item_on_ref = _interp_top_surface(
                    item["x"],
                    item["y"],
                    item["uz_top"],
                    ref["x"],
                    ref["y"],
                )
                diff = item_on_ref - ref["uz_top"]
                rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref["uz_top"]), 1e-12))
                item_peak = float(-np.min(item_on_ref))
                rel_peak = abs(item_peak - ref_peak) / max(abs(ref_peak), 1e-12)
            rel_peak_errors.append(rel_peak)
            rel_l2_errors.append(rel_l2)
            _report(
                lines,
                f"  {_format_mesh(item['mesh'])}: peak rel err {rel_peak:.2%}, top-surface L2 rel err {rel_l2:.2%}",
            )
        for i in range(1, len(ordered)):
            prev_peak = rel_peak_errors[i - 1]
            curr_peak = rel_peak_errors[i]
            prev_l2 = rel_l2_errors[i - 1]
            curr_l2 = rel_l2_errors[i]
            peak_red = (prev_peak - curr_peak) / max(prev_peak, 1e-12)
            l2_red = (prev_l2 - curr_l2) / max(prev_l2, 1e-12)
            _report(
                lines,
                f"  error reduction {_format_mesh(ordered[i-1]['mesh'])} -> {_format_mesh(ordered[i]['mesh'])}: peak {peak_red:.2%}, L2 {l2_red:.2%}",
            )
    _report(lines, "")


def dataset_checks(lines, regenerate):
    _report(lines, "Step 3 — Surrogate Dataset Checks")
    if not os.path.exists(config.DATASET_PATH) or regenerate:
        _report(lines, "Dataset not found or regenerate requested. Generating baseline dataset...")
        dataset = data_utils.generate_dataset()
        data_utils.save_dataset(config.DATASET_PATH, dataset)
    else:
        dataset = data_utils.load_dataset(config.DATASET_PATH)
    _report(lines, f"Samples: {dataset['x_norm'].shape[0]}, dim: {dataset['x_norm'].shape[1]}")
    x_min = float(dataset["x_norm"].min())
    x_max = float(dataset["x_norm"].max())
    y_min = float(dataset["y_norm"].min())
    y_max = float(dataset["y_norm"].max())
    ranges = config.DESIGN_RANGES
    x_raw = dataset["x_raw"]
    oob = []
    for i, name in enumerate(dataset["param_names"]):
        low, high = ranges[name]
        below = np.any(x_raw[:, i] < low)
        above = np.any(x_raw[:, i] > high)
        if below or above:
            oob.append(name)
    _report(lines, f"x_norm range: [{x_min:.3f}, {x_max:.3f}]")
    _report(lines, f"y_norm range: [{y_min:.3f}, {y_max:.3f}]")
    if oob:
        _report(lines, f"Out-of-range params: {oob}")
    else:
        _report(lines, "Out-of-range params: none")
    _report(lines, "")
    return dataset


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _relative_error(y_true, y_pred, eps):
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)


def overfit_check(lines, dataset, device, samples, epochs, lr, target_mse):
    _report(lines, "Step 4 — Overfit Capacity Check")
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(dataset["x_norm"].shape[0], size=samples, replace=False)
    x = dataset["x_norm"][idx].astype(np.float32)
    y = dataset["y_norm"][idx].astype(np.float32).reshape(-1, 1)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=samples, shuffle=True)

    # Use a higher-capacity model for the strict overfit check only.
    model = surrogate.MLP(
        input_dim=len(dataset["param_names"]),
        hidden_layers=max(6, config.HIDDEN_LAYERS + 3),
        hidden_units=max(256, config.HIDDEN_UNITS * 4),
        activation="gelu",
    ).to(device)
    criterion = torch.nn.MSELoss()
    loss_value = None

    # Full-batch LBFGS tends to memorize tiny datasets quickly.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs, line_search_fn="strong_wolfe")
    xb_full, yb_full = next(iter(loader))
    xb_full = xb_full.to(device)
    yb_full = yb_full.to(device)

    def closure():
        optimizer.zero_grad()
        pred = model(xb_full)
        loss = criterion(pred, yb_full)
        loss.backward()
        return loss

    for _ in range(epochs):
        loss = optimizer.step(closure)
        loss_value = loss.detach().item()
        if loss_value <= target_mse:
            break
    _report(lines, f"Samples: {samples}")
    _report(lines, "Overfit model: widened/deepened (tanh) for capacity check")
    final_mse = float(loss) if loss is not None else float("nan")
    _report(lines, f"Final train MSE: {final_mse:.6e}")
    _report(lines, f"Target MSE: {target_mse:.1e}")
    _report(lines, "Status: PASS" if final_mse <= target_mse else "Status: FAIL")
    _report(lines, "")


def surrogate_checks(lines, dataset, device):
    _report(lines, "Step 5 — Surrogate Validation")
    if not os.path.exists(config.MODEL_PATH):
        _report(lines, f"Model not found at {config.MODEL_PATH}; run run_phase1.py to train.")
        return

    # The model payload includes numpy arrays and config metadata, so we need
    # full deserialization (trusted local file).
    payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model_cfg = payload.get("config", {})
    model = surrogate.MLP(
        input_dim=len(payload["param_names"]),
        hidden_layers=model_cfg.get("hidden_layers", config.HIDDEN_LAYERS),
        hidden_units=model_cfg.get("hidden_units", config.HIDDEN_UNITS),
        activation=model_cfg.get("activation", config.ACTIVATION),
    ).to(device)
    model.load_state_dict(payload["state_dict"])

    n_samples = dataset["x_norm"].shape[0]
    splits = data_utils.split_indices(n_samples, config.TRAIN_FRACTION, config.VAL_FRACTION, config.SEED)
    y_pred = validate.denormalize_y(
        surrogate.predict(model, dataset["x_norm"], device),
        dataset["y_min"],
        dataset["y_max"],
    )
    metrics = {
        "train_mse": validate.mse(dataset["y_raw"][splits[0]], y_pred[splits[0]]),
        "val_mse": validate.mse(dataset["y_raw"][splits[1]], y_pred[splits[1]]),
        "test_mse": validate.mse(dataset["y_raw"][splits[2]], y_pred[splits[2]]),
        "train_rmse": _rmse(dataset["y_raw"][splits[0]], y_pred[splits[0]]),
        "val_rmse": _rmse(dataset["y_raw"][splits[1]], y_pred[splits[1]]),
        "test_rmse": _rmse(dataset["y_raw"][splits[2]], y_pred[splits[2]]),
        "train_mae": _mae(dataset["y_raw"][splits[0]], y_pred[splits[0]]),
        "val_mae": _mae(dataset["y_raw"][splits[1]], y_pred[splits[1]]),
        "test_mae": _mae(dataset["y_raw"][splits[2]], y_pred[splits[2]]),
    }
    _report(lines, f"Train MSE: {metrics['train_mse']:.6e}")
    _report(lines, f"Val MSE: {metrics['val_mse']:.6e}")
    _report(lines, f"Test MSE: {metrics['test_mse']:.6e}")
    _report(lines, f"Train RMSE: {metrics['train_rmse']:.6e}")
    _report(lines, f"Val RMSE: {metrics['val_rmse']:.6e}")
    _report(lines, f"Test RMSE: {metrics['test_rmse']:.6e}")
    _report(lines, f"Train MAE: {metrics['train_mae']:.6e}")
    _report(lines, f"Val MAE: {metrics['val_mae']:.6e}")
    _report(lines, f"Test MAE: {metrics['test_mae']:.6e}")
    rel_err = _relative_error(dataset["y_raw"][splits[2]], y_pred[splits[2]], eps=1e-12)
    _report(lines, f"Test RelErr mean: {float(np.mean(rel_err)):.4%}")
    _report(lines, f"Test RelErr p95: {float(np.percentile(rel_err, 95)):.4%}")

    pred_plot = os.path.join(config.PLOTS_DIR, "pred_vs_truth.png")
    validate.plot_pred_vs_true(
        dataset["y_raw"][splits[2]],
        y_pred[splits[2]],
        pred_plot,
    )
    trend_plot = os.path.join(config.PLOTS_DIR, "trend_fidelity.png")
    _call_with_output(lines, validate.plot_trend, config.TREND_SWEEP_PARAM, model, dataset, device, trend_plot)
    safety = _call_with_output(lines, validate.optimization_safety_check, model, dataset, device)
    _report(lines, "Optimization safety check:")
    _report(lines, f"  mu*: {np.array2string(safety['mu_star'], precision=3)}")
    _report(lines, f"  surrogate y(mu*): {safety['pred_star']:.6e}")
    _report(lines, f"  baseline y(mu*): {safety['true_star']:.6e}")
    _report(lines, "")


def topk_safety_check(lines, dataset, model, device, k, candidates):
    _report(lines, "Step 6 — Top-k Optimization Safety Check")
    rng = np.random.default_rng(config.SEED + 19)
    params = dataset["param_names"]
    ranges = config.DESIGN_RANGES
    samples = data_utils.sample_designs(candidates, ranges, rng.integers(0, 1_000_000))
    x_norm, _, _ = data_utils.normalize_inputs(samples, ranges)
    y_norm = surrogate.predict(model, x_norm, device)
    y_pred = validate.denormalize_y(y_norm, dataset["y_min"], dataset["y_max"])
    order = np.argsort(y_pred)
    top_idx = order[:k]
    baseline_vals = []
    for idx in top_idx:
        baseline_vals.append(baseline.compute_response(samples[idx]))
    baseline_vals = np.array(baseline_vals, dtype=float)
    _report(lines, f"Candidates: {candidates}")
    _report(lines, f"Top-k: {k}")
    _report(lines, f"Surrogate best (min) y: {float(y_pred[top_idx[0]]):.6e}")
    _report(lines, f"Baseline top-k mean y: {float(np.mean(baseline_vals)):.6e}")
    _report(lines, f"Baseline top-k min y: {float(np.min(baseline_vals)):.6e}")
    _report(lines, "")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 diagnostics")
    parser.add_argument("--skip-mesh", action="store_true", help="Skip mesh sweep diagnostics")
    parser.add_argument("--skip-surrogate", action="store_true", help="Skip surrogate validation checks")
    parser.add_argument("--regenerate-dataset", action="store_true", help="Regenerate dataset if needed")
    parser.add_argument("--mesh", action="append", default=[], help="Mesh as ne_x,ne_y,ne_z")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--report", default=os.path.join(config.OUTPUT_DIR, "phase1_diagnostics.txt"))
    parser.add_argument("--skip-overfit", action="store_true", help="Skip tiny overfit check")
    parser.add_argument("--overfit-samples", type=int, default=10)
    parser.add_argument("--overfit-epochs", type=int, default=5000)
    parser.add_argument("--overfit-lr", type=float, default=1e-1)
    parser.add_argument("--overfit-target-mse", type=float, default=1e-7)
    parser.add_argument("--skip-topk", action="store_true", help="Skip top-k safety check")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--topk-candidates", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    lines = []
    report_reproducibility(lines)
    describe_problem(lines)

    meshes = DEFAULT_MESHES
    if args.mesh:
        meshes = []
        for item in args.mesh:
            parts = item.split(",")
            if len(parts) != 3:
                raise ValueError("Mesh must be formatted as ne_x,ne_y,ne_z")
            meshes.append({"ne_x": int(parts[0]), "ne_y": int(parts[1]), "ne_z": int(parts[2])})

    if not args.skip_mesh:
        mesh_sweep(lines, meshes)
    else:
        _report(lines, "Step 2 — Baseline FEA Mesh Sweep (skipped)")
        _report(lines, "")

    dataset = None
    if not args.skip_surrogate:
        dataset = dataset_checks(lines, args.regenerate_dataset)
        if not args.skip_overfit:
            overfit_check(
                lines,
                dataset,
                device,
                args.overfit_samples,
                args.overfit_epochs,
                args.overfit_lr,
                args.overfit_target_mse,
            )
        surrogate_checks(lines, dataset, device)
        if not args.skip_topk:
            payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
            model_cfg = payload.get("config", {})
            model = surrogate.MLP(
                input_dim=len(payload["param_names"]),
                hidden_layers=model_cfg.get("hidden_layers", config.HIDDEN_LAYERS),
                hidden_units=model_cfg.get("hidden_units", config.HIDDEN_UNITS),
                activation=model_cfg.get("activation", config.ACTIVATION),
            ).to(device)
            model.load_state_dict(payload["state_dict"])
            topk_safety_check(lines, dataset, model, device, args.topk, args.topk_candidates)
    else:
        _report(lines, "Step 3/4 — Surrogate checks (skipped)")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote diagnostics report to {args.report}")


if __name__ == "__main__":
    main()
