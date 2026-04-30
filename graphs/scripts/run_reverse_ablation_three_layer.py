"""Reverse ablation: start from full framework, remove one component at a time.

This script trains variants of the three-layer model starting from the full
framework configuration and removing each key component individually.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "graphs" / "data"
THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"


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
        print(f"WARNING: Command failed (exit {proc.returncode}). See log: {log_path}")
        return proc.stdout
    return proc.stdout


def _parse_compare_output(stdout: str) -> dict[str, float]:
    """Parse mean MAE and worst MAE from compare_three_layer_pinn_fem.py output."""
    mean_match = re.search(r"Three-layer sweep mean MAE=([0-9.]+)%", stdout)
    worst_match = re.search(r"Three-layer sweep worst MAE=([0-9.]+)%", stdout)
    return {
        "mean_mae": float(mean_match.group(1)) if mean_match else float("nan"),
        "worst_mae": float(worst_match.group(1)) if worst_match else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse ablation: full framework minus one component.")
    parser.add_argument("--epochs-soap", type=int, default=400, help="SOAP training epochs")
    parser.add_argument("--epochs-lbfgs", type=int, default=0, help="L-BFGS fine-tuning epochs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoint exists")
    args = parser.parse_args()

    out_csv = DATA_DIR / "reverse_ablation_three_layer.csv"
    runs_dir = DATA_DIR / "reverse_ablation_runs" / "three_layer"
    logs_dir = DATA_DIR / "reverse_ablation_logs" / "three_layer"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_WARM_START": "0",
        "PINN_SUPERVISION_CACHE": "1",
        "PINN_REGEN_SUPERVISION": "0",
        "PINN_ADAPTIVE_RESAMPLE_EVERY": "0",  # Disable for stability
    }
    if args.epochs_soap is not None:
        base_env["PINN_EPOCHS_ADAM"] = str(args.epochs_soap)
    if args.epochs_lbfgs is not None:
        base_env["PINN_EPOCHS_LBFGS"] = str(args.epochs_lbfgs)
    if args.device:
        base_env["PINN_DEVICE"] = str(args.device)

    # Full framework configuration (what we used to get 5.16%)
    full_overrides = {
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
        "PINN_FEM_NE_X": "16",
        "PINN_FEM_NE_Y": "16",
        "PINN_FEM_NE_Z": "8",
    }

    def without(**overrides: str) -> dict[str, str]:
        env = dict(full_overrides)
        env.update(overrides)
        return env

    # Reverse ablation: start from full, remove one component
    variants: list[tuple[str, dict[str, str]]] = [
        ("Full framework", full_overrides),
        ("Full framework without compliance-aware scaling", without(
            PINN_E_COMPLIANCE_POWER="0",
            PINN_THICKNESS_COMPLIANCE_ALPHA="0",
            PINN_DISPLACEMENT_COMPLIANCE_SCALE="1",
        )),
        ("Full framework without layerwise PDE decomposition", without(
            PINN_PDE_DECOMPOSE_BY_LAYER="0",
        )),
        ("Full framework without interface continuity", without(
            PINN_W_INTERFACE_U="0",
            PINN_N_INTERFACE="2000",
        )),
        ("Full framework without sparse FEM supervision", without(
            PINN_USE_SUPERVISION_DATA="0",
            PINN_W_DATA="0",
            PINN_N_DATA_POINTS="0",
        )),
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
        env["PINN_MODEL_PATH"] = str(ckpt_path)

        print("\n" + "=" * 80)
        print(f"Variant: {variant_name}")
        print(f"Output dir: {run_dir}")

        # Training
        if args.skip_train and ckpt_path.exists():
            print(f"Skipping training (checkpoint exists): {ckpt_path}")
        else:
            _run(
                [_python(), str(THREE_LAYER_DIR / "train.py")],
                env=env,
                log_path=logs_dir / f"{variant_slug}_train.log",
            )

        # Evaluation using compare_three_layer_pinn_fem.py (grid sweep)
        eval_stdout = _run(
            [_python(), "compare_three_layer_pinn_fem.py"],
            env={**env, "PINN_MODEL_PATH": str(ckpt_path)},
            log_path=logs_dir / f"{variant_slug}_eval.log",
        )
        results = _parse_compare_output(eval_stdout)
        mean_mae = results["mean_mae"]
        worst_mae = results["worst_mae"]
        print(f"  mean MAE (%):  {mean_mae:.2f}")
        print(f"  worst MAE (%): {worst_mae:.2f}")

        removed = "none" if variant_name == "Full framework" else variant_name.replace("Full framework without ", "")
        rows.append({
            "variant": variant_name,
            "removed_component": removed,
            "mean_mae": f"{mean_mae:.2f}" if not (mean_mae != mean_mae) else "",
            "worst_mae": f"{worst_mae:.2f}" if not (worst_mae != worst_mae) else "",
            "checkpoint": str(ckpt_path),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "removed_component", "mean_mae", "worst_mae", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nWrote reverse ablation results:")
    print(f"  - {out_csv}")
    print("\nSummary:")
    for row in rows:
        print(f"  {row['variant']:50s}  mean={row['mean_mae']:>6s}%  worst={row['worst_mae']:>6s}%")


if __name__ == "__main__":
    main()
