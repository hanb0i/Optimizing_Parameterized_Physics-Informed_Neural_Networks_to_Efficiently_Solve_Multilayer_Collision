"""Fit a transparent compliance calibration for the three-layer PINN.

The PINN remains the solver: this script loads the trained PINN, evaluates it
directly, and fits a small multiplicative compliance map to PINN displacements.
FEM is used only as the reference target for the same parameter cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import torch

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ThreeLayerCase,
    calibration_features,
    config,
    ensure_output_dirs,
    load_pinn,
    make_points,
    max_pct,
    random_interior_cases,
    select_device,
    solve_fem_case,
    u_from_v,
    write_json,
)


def _base_u_from_v(v: np.ndarray, pts: np.ndarray, scale: float, e_pow: float, alpha: float) -> np.ndarray:
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _prepare_case(pinn, device, case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    x_nodes, y_nodes, z_nodes, u_fem, _ = solve_fem_case(case, ne_x, ne_y, ne_z)
    xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return {
        "case": case,
        "pts": pts,
        "v": v,
        "u_fem": u_fem.reshape(-1, 3),
        "top_mask": np.isclose(pts[:, 2], case.thickness),
    }


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def _predict(blob: dict, params: np.ndarray, coeffs: np.ndarray | None = None) -> np.ndarray:
    scale, e_pow, alpha = [float(v) for v in params]
    u = _base_u_from_v(blob["v"], blob["pts"], scale, e_pow, alpha)
    if coeffs is not None:
        feats = calibration_features(blob["pts"])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_mult = np.nan_to_num(feats @ coeffs.reshape(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
        mult = np.exp(np.clip(log_mult, -1.5, 1.5))
        u = u * mult
    return u


def _metrics(prepared: list[dict], params: np.ndarray, coeffs: np.ndarray | None = None) -> dict:
    top_maes = []
    volume_maes = []
    top_maxes = []
    for blob in prepared:
        u_pred = _predict(blob, params, coeffs)
        u_ref = blob["u_fem"]
        top = blob["top_mask"]
        top_maes.append(_mae_pct(u_pred[top, 2], u_ref[top, 2]))
        volume_maes.append(_mae_pct(u_pred, u_ref))
        top_maxes.append(max_pct(u_pred[top, 2], u_ref[top, 2]))
    return {
        "top_mae_pct_mean": float(np.mean(top_maes)),
        "top_mae_pct_worst": float(np.max(top_maes)),
        "volume_mae_pct_mean": float(np.mean(volume_maes)),
        "volume_mae_pct_worst": float(np.max(volume_maes)),
        "top_max_pct_worst": float(np.max(top_maxes)),
        "all_top_cases_below_5_pct": bool(np.all(np.array(top_maes) < 5.0)),
    }


def _fit_feature_coefficients(prepared: list[dict], params: np.ndarray, ridge: float, fit_scope: str) -> np.ndarray:
    feat_rows = []
    targets = []
    weights = []
    for blob in prepared:
        u_base = _predict(blob, params, None)
        u_ref = blob["u_fem"]
        if fit_scope == "top":
            mask = blob["top_mask"]
        elif fit_scope == "uz":
            mask = np.ones(len(blob["pts"]), dtype=bool)
        else:
            raise ValueError(f"Unknown fit scope: {fit_scope}")
        pred = u_base[mask, 2]
        ref = u_ref[mask, 2]
        good = (np.abs(pred) > 1e-10) & (np.abs(ref) > 1e-10) & (np.sign(pred) == np.sign(ref))
        if not np.any(good):
            continue
        feats = calibration_features(blob["pts"][mask])[good]
        ratio = np.clip(ref[good] / pred[good], 0.2, 5.0)
        feat_rows.append(feats)
        targets.append(np.log(ratio).reshape(-1, 1))
        z_hat = blob["pts"][mask][good, 2:3] / np.clip(
            blob["pts"][mask][good, 4:5] + blob["pts"][mask][good, 6:7] + blob["pts"][mask][good, 8:9],
            1e-8,
            None,
        )
        top_boost = 1.0 + 2.0 * (z_hat > 0.9).astype(float)
        weights.append(((np.abs(ref[good]).reshape(-1, 1) / max(float(np.max(np.abs(ref))), 1e-12)) + 0.1) * top_boost)

    x_mat = np.vstack(feat_rows)
    y_vec = np.vstack(targets)
    w_vec = np.sqrt(np.vstack(weights))
    xw = x_mat * w_vec
    yw = y_vec * w_vec
    reg = ridge * np.eye(x_mat.shape[1])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lhs = np.nan_to_num(xw.T @ xw, nan=0.0, posinf=0.0, neginf=0.0) + reg
        rhs = np.nan_to_num(xw.T @ yw, nan=0.0, posinf=0.0, neginf=0.0)
    coeffs = np.linalg.solve(lhs, rhs).reshape(-1)
    return coeffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--n-calibration", type=int, default=8)
    parser.add_argument("--n-holdout", type=int, default=8)
    parser.add_argument("--ne-x", type=int, default=8)
    parser.add_argument("--ne-y", type=int, default=8)
    parser.add_argument("--ne-z", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--fit-scope", choices=["top", "uz"], default="uz")
    parser.add_argument("--optimize-coefficients", action="store_true")
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--out", default=str(GRAPHS_DATA_DIR / "three_layer_compliance_calibration.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    random_cases = random_interior_cases(args.n_calibration + args.n_holdout, args.seed)
    calibration_cases = [
        ThreeLayerCase("supervised_soft_bottom", 1.0, 10.0, 10.0, 0.10, 0.02, 0.02),
        ThreeLayerCase("supervised_soft_bottom_repeat1", 1.0, 10.0, 10.0, 0.10, 0.02, 0.02),
        ThreeLayerCase("supervised_soft_bottom_repeat2", 1.0, 10.0, 10.0, 0.10, 0.02, 0.02),
        ThreeLayerCase("supervised_soft_middle", 10.0, 1.0, 10.0, 0.02, 0.10, 0.02),
        ThreeLayerCase("supervised_soft_top", 10.0, 10.0, 1.0, 0.02, 0.02, 0.10),
        *random_cases[: args.n_calibration],
    ]
    holdout_cases = random_cases[args.n_calibration :]

    calibration = [_prepare_case(pinn, device, case, args.ne_x, args.ne_y, args.ne_z) for case in calibration_cases]
    holdout = [_prepare_case(pinn, device, case, args.ne_x, args.ne_y, args.ne_z) for case in holdout_cases]

    params = np.array(
        [
            float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0)),
            float(getattr(config, "E_COMPLIANCE_POWER", 0.95)),
            float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 3.0)),
        ],
        dtype=float,
    )
    coeffs = _fit_feature_coefficients(calibration, params, args.ridge, args.fit_scope)
    if args.optimize_coefficients:
        def objective(c: np.ndarray) -> float:
            m = _metrics(calibration, params, c)
            return m["top_mae_pct_worst"] + 0.35 * m["top_mae_pct_mean"] + 0.05 * m["volume_mae_pct_mean"]

        result = opt.minimize(
            objective,
            coeffs,
            method="Powell",
            bounds=[(-2.0, 2.0)] * len(coeffs),
            options={"maxiter": args.maxiter, "xtol": 1e-3, "ftol": 1e-3},
        )
        if result.success or np.isfinite(result.fun):
            coeffs = np.asarray(result.x, dtype=float)

    payload = {
        "model_path": str(model_path),
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "base_params": {
            "PINN_DISPLACEMENT_COMPLIANCE_SCALE": float(params[0]),
            "PINN_E_COMPLIANCE_POWER": float(params[1]),
            "PINN_THICKNESS_COMPLIANCE_ALPHA": float(params[2]),
        },
        "feature_names": [
            "bias",
            "log_e_mean",
            "log_e1",
            "log_e2",
            "log_e3",
            "log_h_over_total_t",
            "t1_fraction",
            "t2_fraction",
            "t3_fraction",
            "z_hat",
            "z_hat_squared",
            "load_patch_indicator",
            "x_centered",
            "y_centered",
            "x_centered_squared",
            "y_centered_squared",
            "x_y_centered",
            "load_patch_x_centered",
            "load_patch_y_centered",
            "load_patch_x_centered_squared",
            "load_patch_y_centered_squared",
        ],
        "feature_coefficients": [float(v) for v in coeffs],
        "log_multiplier_clip": 1.5,
        "fit_scope": args.fit_scope,
        "calibration_metrics_uncalibrated": _metrics(calibration, params, None),
        "calibration_metrics_tuned": _metrics(calibration, params, coeffs),
        "holdout_metrics_uncalibrated": _metrics(holdout, params, None) if holdout else None,
        "holdout_metrics_tuned": _metrics(holdout, params, coeffs) if holdout else None,
        "note": "The PINN remains the solver. This file defines a transparent multiplicative compliance calibration applied to PINN displacement outputs.",
    }
    write_json(Path(args.out), payload)
    print(f"Wrote {args.out}")
    print(f"Holdout tuned top worst: {payload['holdout_metrics_tuned']['top_mae_pct_worst']:.2f}%")


if __name__ == "__main__":
    main()
