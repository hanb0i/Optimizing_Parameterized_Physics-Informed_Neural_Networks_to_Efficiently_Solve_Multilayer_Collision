#!/usr/bin/env python3
"""
Reproducible grid-sweep verification for the one-layer PINN.

This script evaluates the one-layer PINN against FEM on a structured grid of
parameter combinations (E × thickness), matching the benchmark protocol used
in the paper.  It reports top-surface z-displacement MAE% for every case and
summarizes mean / worst / best.

Usage:
    cd /path/to/repo
    python scripts/verify_one_layer_grid_sweep.py

Environment variables (all optional):
    PINN_MODEL_PATH      – path to the one-layer PINN checkpoint
                           (default: one-layer-workflow/pinn_model.pth)
    PINN_DEVICE          – torch device string, e.g. "cpu", "cuda", "mps"
                           (default: auto-detect)
    PINN_EVAL_E_VALUES   – comma-separated E values to sweep
                           (default: from one-layer-workflow/pinn_config.py E_RANGE)
    PINN_EVAL_T_VALUES   – comma-separated thickness values to sweep
                           (default: from one-layer-workflow/pinn_config.py DATA_THICKNESS_VALUES)
    PINN_EVAL_NE_X       – FEM mesh elements in x (default: 16)
    PINN_EVAL_NE_Y       – FEM mesh elements in y (default: 16)
    PINN_EVAL_NE_Z       – FEM mesh elements in z (default: 8)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"

for _path in (ONE_LAYER_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (mirrored from one_layer_experiment_utils.py to keep script standalone)
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    requested = os.getenv("PINN_DEVICE")
    if requested:
        return torch.device(requested)
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _adapt_state_dict(state: dict, target: dict) -> dict:
    state = dict(state)
    w_key = "layer.net.0.weight"
    if w_key not in state or w_key not in target:
        return state
    src_w = state[w_key]
    tgt_w = target[w_key]
    if src_w.shape == tgt_w.shape or src_w.shape[0] != tgt_w.shape[0]:
        return state
    if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:5] = src_w[:, 0:5]
        adapted[:, 8:11] = src_w[:, 5:8]
        state[w_key] = adapted
    elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:7] = src_w[:, 0:7]
        adapted[:, 8:11] = src_w[:, 7:10]
        state[w_key] = adapted
    return state


def _load_pinn(device: torch.device) -> torch.nn.Module:
    model_path = Path(os.getenv("PINN_MODEL_PATH") or ONE_LAYER_DIR / "pinn_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {model_path}")
    pinn = model.MultiLayerPINN().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    state = _adapt_state_dict(state, pinn.state_dict())
    pinn.load_state_dict(state, strict=False)
    pinn.eval()
    print(f"Loaded PINN checkpoint: {model_path}")
    return pinn


def _u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    e_vals = pts[:, 3:4]
    t_vals = pts[:, 4:5]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_vals**e_pow) * (h_ref / np.clip(t_vals, 1e-8, None)) ** alpha


def _make_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, E: float, thickness: float) -> np.ndarray:
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    return np.stack(
        [
            x,
            y,
            z,
            np.full_like(x, float(E), dtype=float),
            np.full_like(x, float(thickness), dtype=float),
            np.full_like(x, float(r_ref), dtype=float),
            np.full_like(x, float(mu_ref), dtype=float),
            np.full_like(x, float(v0_ref), dtype=float),
        ],
        axis=1,
    )


def _predict_displacement(
    pinn: torch.nn.Module, device: torch.device, pts: np.ndarray, batch_size: int = 32768
) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            batch_pts = pts[start : start + batch_size]
            v = pinn(torch.tensor(batch_pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
            out.append(_u_from_v(v, batch_pts))
    return np.concatenate(out, axis=0)


def _fem_cfg(E: float, thickness: float, ne_x: int, ne_y: int, ne_z: int) -> dict:
    return {
        "geometry": {
            "Lx": float(config.Lx),
            "Ly": float(config.Ly),
            "H": float(thickness),
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {"E": float(E), "nu": float(config.nu_vals[0])},
        "load_patch": {
            "pressure": float(config.p0),
            "x_start": float(config.LOAD_PATCH_X[0]) / float(config.Lx),
            "x_end": float(config.LOAD_PATCH_X[1]) / float(config.Lx),
            "y_start": float(config.LOAD_PATCH_Y[0]) / float(config.Ly),
            "y_end": float(config.LOAD_PATCH_Y[1]) / float(config.Ly),
        },
    }


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    device = _select_device()
    print(f"Using device: {device}")

    # Mesh resolution
    ne_x = int(os.getenv("PINN_EVAL_NE_X", "16"))
    ne_y = int(os.getenv("PINN_EVAL_NE_Y", "16"))
    ne_z = int(os.getenv("PINN_EVAL_NE_Z", "8"))
    print(f"FEM mesh: {ne_x} × {ne_y} × {ne_z}")

    # Parameter grid
    e_env = os.getenv("PINN_EVAL_E_VALUES")
    if e_env:
        e_values = [float(v.strip()) for v in e_env.split(",")]
    else:
        e_values = [float(v) for v in getattr(config, "DATA_E_VALUES", [1.0, 5.0, 10.0])]

    t_env = os.getenv("PINN_EVAL_T_VALUES")
    if t_env:
        t_values = [float(v.strip()) for v in t_env.split(",")]
    else:
        t_values = [float(v) for v in getattr(config, "DATA_THICKNESS_VALUES", [0.05, 0.10, 0.15])]

    print(f"E values: {e_values}")
    print(f"Thickness values: {t_values}")
    print(f"Total cases: {len(e_values)} × {len(t_values)} = {len(e_values) * len(t_values)}")

    pinn = _load_pinn(device)

    results = []
    for E in e_values:
        for thickness in t_values:
            # FEM solve
            x_nodes, y_nodes, z_nodes, u_fem = fem_solver.solve_fem(_fem_cfg(E, thickness, ne_x, ne_y, ne_z))
            x_nodes = np.asarray(x_nodes)
            y_nodes = np.asarray(y_nodes)
            z_nodes = np.asarray(z_nodes)
            u_fem = np.asarray(u_fem)

            # PINN predict (full volume)
            xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
            pts = _make_points(xg.ravel(), yg.ravel(), zg.ravel(), E, thickness)
            u_pinn = _predict_displacement(pinn, device, pts).reshape(u_fem.shape)

            # Top-surface z-displacement MAE%
            top_pred = u_pinn[:, :, -1, 2]
            top_ref = u_fem[:, :, -1, 2]
            mae = _mae_pct(top_pred, top_ref)

            case_id = f"E{E:g}_t{thickness:g}"
            results.append({
                "case_id": case_id,
                "E": float(E),
                "thickness": float(thickness),
                "top_uz_mae_pct": float(mae),
            })
            print(f"  {case_id}: top MAE = {mae:.2f}%")

    maes = [r["top_uz_mae_pct"] for r in results]
    mean_mae = float(np.mean(maes))
    worst_mae = float(np.max(maes))
    best_mae = float(np.min(maes))

    summary = {
        "model": "one-layer",
        "model_path": str(os.getenv("PINN_MODEL_PATH") or ONE_LAYER_DIR / "pinn_model.pth"),
        "device": str(device),
        "mesh": {"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "e_values": e_values,
        "t_values": t_values,
        "n_cases": len(results),
        "mean_top_uz_mae_pct": mean_mae,
        "worst_top_uz_mae_pct": worst_mae,
        "best_top_uz_mae_pct": best_mae,
        "cases": results,
    }

    out_dir = REPO_ROOT / "graphs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "one_layer_grid_sweep_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("ONE-LAYER GRID SWEEP VERIFICATION")
    print("=" * 60)
    print(f"  Mean top MAE: {mean_mae:.2f}%")
    print(f"  Worst top MAE: {worst_mae:.2f}%")
    print(f"  Best top MAE: {best_mae:.2f}%")
    print(f"  Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
