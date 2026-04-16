from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from _common import DATA_DIR, REPO_ROOT


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}). See log: {log_path}")
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three-layer PINN ablation variants (FEM-referenced).")
    parser.add_argument("--epochs-soap", type=int, default=None, help="Override PINN_EPOCHS_SOAP for all runs")
    parser.add_argument("--device", default=None, help="Override PINN_DEVICE (cpu/cuda/mps)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoint exists for a variant")
    parser.add_argument("--n-cases", type=int, default=8, help="Random interior evaluation cases per variant")
    parser.add_argument("--seed", type=int, default=20260415, help="Seed for random interior evaluation")
    args = parser.parse_args()

    out_csv = DATA_DIR / "ablation_results.csv"
    runs_dir = DATA_DIR / "ablation_runs"
    logs_dir = DATA_DIR / "ablation_logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_WARM_START": "0",
        "PINN_SUPERVISION_CACHE": "1",
        "PINN_REGEN_SUPERVISION": "0",
    }
    calibration_path = DATA_DIR / "three_layer_compliance_calibration.json"
    if args.epochs_soap is not None:
        base_env["PINN_EPOCHS_ADAM"] = str(args.epochs_soap)
        base_env["PINN_EPOCHS_SOAP"] = str(args.epochs_soap)
    if args.device:
        base_env["PINN_DEVICE"] = str(args.device)

    # Variants match the paper table.
    variants: list[tuple[str, dict[str, str]]] = [
        (
            "Base parametric PINN",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
        ),
        (
            "+ Compliance-aware scaling",
            {
                "PINN_E_COMPLIANCE_POWER": "0.95",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
                "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
        ),
        (
            "+ Layerwise PDE decomposition",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
        ),
        (
            "+ Interface continuity enforcement",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "300",
                "PINN_N_INTERFACE": "16000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
        ),
        (
            "+ Sparse FEM supervision",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_W_DATA": "400",
                "PINN_N_DATA_POINTS": "36000",
                "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
                "PINN_FEM_NE_X": "10",
                "PINN_FEM_NE_Y": "10",
                "PINN_FEM_NE_Z": "4",
            },
        ),
        (
            "Full framework",
            {
                "PINN_E_COMPLIANCE_POWER": "0.95",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
                "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
                "PINN_W_INTERFACE_U": "300",
                "PINN_W_PDE": "10",
                "PINN_W_DATA": "400",
                "PINN_N_INTERFACE": "16000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_N_DATA_POINTS": "36000",
                "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
                "PINN_FEM_NE_X": "10",
                "PINN_FEM_NE_Y": "10",
                "PINN_FEM_NE_Z": "4",
            },
        ),
    ]

    rows: list[dict[str, str]] = []
    for variant_name, overrides in variants:
        variant_slug = (
            variant_name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("/", "_")
            .replace("-", "_")
        )
        run_dir = runs_dir / variant_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / "pinn_model.pth"

        env = dict(base_env)
        env.update(overrides)
        env["PINN_OUT_DIR"] = str(run_dir)
        env["PINN_EVAL_OUT_DIR"] = str(run_dir / "eval_viz")
        env["PINN_MODEL_PATH"] = str(ckpt_path)
        if variant_name == "Full framework" and calibration_path.exists():
            env["PINN_CALIBRATION_JSON"] = str(calibration_path)

        print("\n" + "=" * 80)
        print(f"Variant: {variant_name}")
        print(f"Output dir: {run_dir}")

        if args.skip_train and ckpt_path.exists():
            print(f"Skipping training (checkpoint exists): {ckpt_path}")
        else:
            _run(
                [_python(), "pinn-workflow/train.py"],
                env=env,
                log_path=logs_dir / f"{variant_slug}_train.log",
            )

        eval_summary = run_dir / "random_interior_generalization_summary.json"
        eval_csv = run_dir / "random_interior_generalization.csv"
        _run(
            [
                _python(),
                "scripts/run_random_interior_generalization.py",
                "--model-path",
                str(ckpt_path),
                "--n-cases",
                str(args.n_cases),
                "--seed",
                str(args.seed),
                "--out-csv",
                str(eval_csv),
                "--out-summary",
                str(eval_summary),
            ],
            env=env,
            log_path=logs_dir / f"{variant_slug}_eval.log",
        )
        summary = json.loads(eval_summary.read_text())
        mean_mae = float(summary["top_uz_mae_pct_mean"])
        worst_mae = float(summary["top_uz_mae_pct_worst"])
        print(f"  mean MAE (%):  {mean_mae:.2f}")
        print(f"  worst MAE (%): {worst_mae:.2f}")

        rows.append({"variant": variant_name, "mean_mae": f"{mean_mae:.4f}", "worst_mae": f"{worst_mae:.4f}"})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_mae", "worst_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nWrote ablation results:")
    print(f"  - {out_csv}")


if __name__ == "__main__":
    main()
