#!/usr/bin/env python3
"""
Reproducible grid-sweep verification for the three-layer PINN.

This script evaluates the three-layer PINN against FEM on an exhaustive grid of
parameter combinations (E₁ × E₂ × E₃ × t₁ × t₂ × t₃), matching the benchmark
protocol used in the paper.  It reports top-surface z-displacement MAE% for
every case and summarizes mean / worst / best.

Usage:
    cd /path/to/repo
    python scripts/verify_three_layer_grid_sweep.py

Environment variables (all optional):
    PINN_MODEL_PATH      – path to the three-layer PINN checkpoint
                           (default: pinn-workflow/pinn_model.pth)
    PINN_DEVICE          – torch device string, e.g. "cpu", "cuda", "mps"
                           (default: auto-detect)
    PINN_EVAL_E_VALUES   – comma-separated E values to sweep
                           (default: from three-layer-workflow/pinn_config.py E_RANGE)
    PINN_EVAL_T1_VALUES  – comma-separated t₁ values to sweep
                           (default: from three-layer-workflow/pinn_config.py DATA_T1_VALUES)
    PINN_EVAL_T2_VALUES  – comma-separated t₂ values to sweep
                           (default: from three-layer-workflow/pinn_config.py DATA_T2_VALUES)
    PINN_EVAL_T3_VALUES  – comma-separated t₃ values to sweep
                           (default: from three-layer-workflow/pinn_config.py DATA_T3_VALUES)
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
# Code lives in three-layer-workflow/; the best checkpoint is in pinn-workflow/
CODE_DIR = REPO_ROOT / "three-layer-workflow"
PINN_WORKFLOW_DIR = REPO_ROOT / "pinn-workflow"
if not PINN_WORKFLOW_DIR.exists():
    PINN_WORKFLOW_DIR = CODE_DIR
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"

for _path in (CODE_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (mirrored from compare_three_layer_pinn_fem.py to keep standalone)
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


def _load_pinn(device: torch.device) -> torch.nn.Module:
    model_path = Path(os.getenv("PINN_MODEL_PATH") or PINN_WORKFLOW_DIR / "pinn_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {model_path}")
    pinn = model.MultiLayerPINN().to(device)
    sd = torch.load(str(model_path), map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    print(f"Loaded PINN checkpoint: {model_path}")
    return pinn


def _ref_params() -> tuple[float, float, float]:
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply compliance scaling to raw network output v to get displacement u."""
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    u = scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    return u


def _predict_pinn(
    pinn: torch.nn.Module,
    device: torch.device,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    z_flat: np.ndarray,
    e1: float,
    e2: float,
    e3: float,
    t1: float,
    t2: float,
    t3: float,
) -> np.ndarray:
    r_ref, mu_ref, v0_ref = _ref_params()
    pts = np.stack(
        [
            x_flat,
            y_flat,
            z_flat,
            np.full_like(x_flat, float(e1)),
            np.full_like(x_flat, float(t1)),
            np.full_like(x_flat, float(e2)),
            np.full_like(x_flat, float(t2)),
            np.full_like(x_flat, float(e3)),
            np.full_like(x_flat, float(t3)),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return _u_from_v(v, pts)


def _run_fem(e1: float, e2: float, e3: float, t1: float, t2: float, t3: float, ne_x: int, ne_y: int, ne_z: int):
    thickness = float(t1) + float(t2) + float(t3)
    cfg = {
        "geometry": {
            "Lx": float(config.Lx),
            "Ly": float(config.Ly),
            "H": thickness,
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {
            "E_layers": [float(e1), float(e2), float(e3)],
            "t_layers": [float(t1), float(t2), float(t3)],
            "nu": float(config.nu_vals[0]),
        },
        "load_patch": {
            "pressure": float(config.p0),
            "x_start": float(config.LOAD_PATCH_X[0]) / float(config.Lx),
            "x_end": float(config.LOAD_PATCH_X[1]) / float(config.Lx),
            "y_start": float(config.LOAD_PATCH_Y[0]) / float(config.Ly),
            "y_end": float(config.LOAD_PATCH_Y[1]) / float(config.Ly),
        },
    }
    return fem_solver.solve_three_layer_fem(cfg)


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    mae = float(np.mean(np.abs(pred - ref)))
    denom = float(np.max(np.abs(ref)))
    return 100.0 * mae / denom if denom > 0 else 0.0


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
        e_range = getattr(config, "E_RANGE", [1.0, 10.0])
        e_values = [float(e_range[0]), float(e_range[1])]

    t1_env = os.getenv("PINN_EVAL_T1_VALUES")
    if t1_env:
        t1_values = [float(v.strip()) for v in t1_env.split(",")]
    else:
        t1_values = [float(v) for v in getattr(config, "DATA_T1_VALUES", [0.02, 0.10])]

    t2_env = os.getenv("PINN_EVAL_T2_VALUES")
    if t2_env:
        t2_values = [float(v.strip()) for v in t2_env.split(",")]
    else:
        t2_values = [float(v) for v in getattr(config, "DATA_T2_VALUES", [0.02, 0.10])]

    t3_env = os.getenv("PINN_EVAL_T3_VALUES")
    if t3_env:
        t3_values = [float(v.strip()) for v in t3_env.split(",")]
    else:
        t3_values = [float(v) for v in getattr(config, "DATA_T3_VALUES", [0.02, 0.10])]

    n_cases = len(e_values) ** 3 * len(t1_values) * len(t2_values) * len(t3_values)
    print(f"E values: {e_values}")
    print(f"t1 values: {t1_values}")
    print(f"t2 values: {t2_values}")
    print(f"t3 values: {t3_values}")
    print(f"Total cases: {len(e_values)}³ × {len(t1_values)} × {len(t2_values)} × {len(t3_values)} = {n_cases}")

    pinn = _load_pinn(device)

    results = []
    fea_cache: dict[tuple, tuple] = {}

    for t1 in t1_values:
        for t2 in t2_values:
            for t3 in t3_values:
                for e1 in e_values:
                    for e2 in e_values:
                        for e3 in e_values:
                            key = (e1, e2, e3, t1, t2, t3)
                            if key not in fea_cache:
                                fea_cache[key] = _run_fem(e1, e2, e3, t1, t2, t3, ne_x, ne_y, ne_z)
                            x_nodes, y_nodes, _, u_fea = fea_cache[key]
                            x_nodes = np.asarray(x_nodes)
                            y_nodes = np.asarray(y_nodes)
                            u_fea = np.asarray(u_fea)

                            thickness = float(t1) + float(t2) + float(t3)
                            x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
                            u_pinn_top = _predict_pinn(
                                pinn,
                                device,
                                x_grid.ravel(),
                                y_grid.ravel(),
                                np.full(x_grid.size, thickness),
                                e1,
                                e2,
                                e3,
                                t1,
                                t2,
                                t3,
                            ).reshape(len(x_nodes), len(y_nodes), 3)

                            u_z_fea_top = u_fea[:, :, -1, 2]
                            u_z_pinn_top = u_pinn_top[:, :, 2]
                            mae = _mae_pct(u_z_pinn_top, u_z_fea_top)

                            case_id = f"E1_{e1:g}_E2_{e2:g}_E3_{e3:g}_t1_{t1:g}_t2_{t2:g}_t3_{t3:g}"
                            results.append({
                                "case_id": case_id,
                                "e1": float(e1),
                                "e2": float(e2),
                                "e3": float(e3),
                                "t1": float(t1),
                                "t2": float(t2),
                                "t3": float(t3),
                                "top_uz_mae_pct": float(mae),
                            })
                            print(f"  {case_id}: top MAE = {mae:.2f}%")

    maes = [r["top_uz_mae_pct"] for r in results]
    mean_mae = float(np.mean(maes))
    worst_mae = float(np.max(maes))
    best_mae = float(np.min(maes))
    worst_idx = int(np.argmax(maes))
    worst_case = results[worst_idx]

    summary = {
        "model": "three-layer",
        "model_path": str(os.getenv("PINN_MODEL_PATH") or PINN_WORKFLOW_DIR / "pinn_model.pth"),
        "device": str(device),
        "mesh": {"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "e_values": e_values,
        "t1_values": t1_values,
        "t2_values": t2_values,
        "t3_values": t3_values,
        "n_cases": len(results),
        "mean_top_uz_mae_pct": mean_mae,
        "worst_top_uz_mae_pct": worst_mae,
        "best_top_uz_mae_pct": best_mae,
        "worst_case": worst_case,
        "cases": results,
    }

    out_dir = REPO_ROOT / "graphs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "three_layer_grid_sweep_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("THREE-LAYER GRID SWEEP VERIFICATION")
    print("=" * 60)
    print(f"  Mean top MAE: {mean_mae:.2f}%")
    print(f"  Worst top MAE: {worst_mae:.2f}%")
    print(f"  Best top MAE: {best_mae:.2f}%")
    print(f"  Worst case: E=[{worst_case['e1']},{worst_case['e2']},{worst_case['e3']}] "
          f"t=[{worst_case['t1']},{worst_case['t2']},{worst_case['t3']}]")
    print(f"  Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
