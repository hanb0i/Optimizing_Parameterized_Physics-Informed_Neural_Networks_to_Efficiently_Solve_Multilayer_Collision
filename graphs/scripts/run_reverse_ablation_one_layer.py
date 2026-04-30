"""Reverse ablation: start from full one-layer framework, remove one component at a time."""

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
ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"


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


def _parse_generalization_output(stdout: str) -> dict[str, float]:
    """Parse mean and worst MAE from run_one_layer_generalization.py output."""
    # The script writes a summary JSON, but we can also parse stdout
    mean_match = re.search(r"top_uz_mae_pct_mean[=:]\s*([0-9.]+)", stdout)
    worst_match = re.search(r"top_uz_mae_pct_worst[=:]\s*([0-9.]+)", stdout)
    return {
        "mean_mae": float(mean_match.group(1)) if mean_match else float("nan"),
        "worst_mae": float(worst_match.group(1)) if worst_match else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse ablation: full one-layer framework minus one component.")
    parser.add_argument("--epochs-soap", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_csv = DATA_DIR / "reverse_ablation_one_layer.csv"
    runs_dir = DATA_DIR / "reverse_ablation_runs" / "one_layer"
    logs_dir = DATA_DIR / "reverse_ablation_logs" / "one_layer"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_WARM_START": "0",
    }
    if args.epochs_soap is not None:
        base_env["PINN_EPOCHS_ADAM"] = str(args.epochs_soap)
        base_env["PINN_EPOCHS_SOAP"] = str(args.epochs_soap)
    if args.device:
        base_env["PINN_DEVICE"] = args.device

    # Full one-layer framework config
    full_overrides = {
        "PINN_E_COMPLIANCE_POWER": "0.973",
        "PINN_THICKNESS_COMPLIANCE_ALPHA": "1.234",
        "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
        "PINN_USE_SUPERVISION_DATA": "1",
        "PINN_W_DATA": "1",
        "PINN_N_DATA_POINTS": "9000",
        "PINN_ADAPTIVE_RESAMPLE_EVERY": "0",  # Disable for stability
    }

    def without(**overrides: str) -> dict[str, str]:
        env = dict(full_overrides)
        env.update(overrides)
        return env

    variants: list[tuple[str, dict[str, str]]] = [
        ("Full framework", full_overrides),
        ("Full framework without compliance-aware scaling", without(
            PINN_E_COMPLIANCE_POWER="0",
            PINN_THICKNESS_COMPLIANCE_ALPHA="0",
        )),
        ("Full framework without FEM supervision", without(
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
        eval_summary = run_dir / "generalization_summary.json"
        eval_csv = run_dir / "generalization.csv"

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
                [_python(), str(ONE_LAYER_DIR / "train.py")],
                env=env,
                log_path=logs_dir / f"{variant_slug}_train.log",
            )

        # Evaluation
        _run(
            [
                _python(),
                "scripts/run_one_layer_generalization.py",
                "--model-path", str(ckpt_path),
                "--n-cases", "8",
                "--out-csv", str(eval_csv),
                "--out-summary", str(eval_summary),
            ],
            env=env,
            log_path=logs_dir / f"{variant_slug}_eval.log",
        )

        summary = json.loads(eval_summary.read_text())
        mean_mae = float(summary["top_uz_mae_pct_mean"])
        worst_mae = float(summary["top_uz_mae_pct_worst"])
        print(f"  mean MAE (%):  {mean_mae:.2f}")
        print(f"  worst MAE (%): {worst_mae:.2f}")

        removed = "none" if variant_name == "Full framework" else variant_name.replace("Full framework without ", "")
        rows.append({
            "variant": variant_name,
            "removed_component": removed,
            "mean_mae": f"{mean_mae:.2f}",
            "worst_mae": f"{worst_mae:.2f}",
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
