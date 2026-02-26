#!/usr/bin/env python3
"""
ISEF Math RRI Surrogate Verification Figures

HOW TO RUN
1) Put your CSV at `data.csv` (or pass `--csv path/to/file.csv`)
2) Run:
   python3 make_isef_rri_figs.py

Outputs (in current directory unless you pass --outdir):
  - parity_plot.png / parity_plot.pdf
  - runtime_speedup.png / runtime_speedup.pdf

Data expectations (robust to column name variations):
  Required:
    - y_true: ground-truth QoI from simulation/FEA
    - y_pred: surrogate predicted QoI
  Optional runtimes (seconds per evaluation):
    - t_sim: simulation/FEA runtime per sample
    - t_surr: surrogate inference runtime per sample

If runtime columns are missing, you can set constants below or pass CLI flags.

Dependencies: pandas, numpy, matplotlib (no seaborn).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Optional manual runtime values
# (used only if runtime columns are missing AND CLI constants are not provided)
# -----------------------------
MANUAL_T_SIM_SECONDS: float | None = None   # e.g., 12.5
MANUAL_T_SURR_SECONDS: float | None = None  # e.g., 0.003


def _setup_matplotlib() -> None:
    # Avoid warnings / slow imports when the default user cache directory
    # isn't writable in the environment (common in sandboxed runs).
    repo_root = Path(__file__).resolve().parent
    cache_dir = repo_root / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _normalize_colname(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch in ("_",))


def _pick_column(df: pd.DataFrame, preferred: str | None, aliases: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred

    norm_map: dict[str, str] = {_normalize_colname(c): c for c in df.columns}
    for a in aliases:
        hit = norm_map.get(_normalize_colname(a))
        if hit is not None:
            return hit
    return None


def _fail_missing_cols(df: pd.DataFrame, missing: list[str], expectations: dict[str, list[str]]) -> None:
    print("\nERROR: Missing required column(s):", ", ".join(missing), file=sys.stderr)
    print("Available columns:", ", ".join(map(str, df.columns)), file=sys.stderr)
    print("\nExpected names (any of):", file=sys.stderr)
    for key, vals in expectations.items():
        print(f"  - {key}: {vals}", file=sys.stderr)
    print("\nTip: pass explicit column names, e.g.:", file=sys.stderr)
    print("  python3 make_isef_rri_figs.py --true-col <col> --pred-col <col>", file=sys.stderr)
    raise SystemExit(2)


def _finite_xy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = _finite_xy(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("No finite y_true/y_pred pairs after filtering NaN/inf.")

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - (ss_res / ss_tot))

    return {"r2": r2, "rmse": rmse, "mae": mae}


def _maybe_log_runtime(median_sim: float, median_surr: float) -> bool:
    if median_sim <= 0 or median_surr <= 0:
        return False
    ratio = max(median_sim, median_surr) / min(median_sim, median_surr)
    return ratio > 100.0


def make_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: Path,
    title: str,
    show_band_10pct: bool,
) -> None:
    import matplotlib.pyplot as plt

    y_true, y_pred = _finite_xy(y_true, y_pred)
    m = _metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolors="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.05
        lo -= pad
        hi += pad

    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.6, label="y = x")

    # Optional ±10% band (only meaningful for positive values)
    if show_band_10pct:
        if lo > 0:
            xline = np.array([lo, hi], dtype=float)
            ax.plot(xline, 1.10 * xline, color="#444444", linewidth=1.0, linestyle="--", alpha=0.9)
            ax.plot(xline, 0.90 * xline, color="#444444", linewidth=1.0, linestyle="--", alpha=0.9)
            ax.fill_between(xline, 0.90 * xline, 1.10 * xline, color="#AAAAAA", alpha=0.15, label="±10%")
        else:
            print("Note: --band10 requested but data include non-positive values; skipping ±10% band.")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True QoI")
    ax.set_ylabel("Predicted QoI")
    ax.set_title(title)

    text = (
        f"$R^2$ = {m['r2']:.4f}\n"
        f"RMSE = {m['rmse']:.4g}\n"
        f"MAE  = {m['mae']:.4g}\n"
        f"n    = {y_true.size}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.95),
    )

    ax.legend(frameon=True, fancybox=True, loc="lower right")
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "parity_plot.png")
    fig.savefig(outdir / "parity_plot.pdf")
    plt.close(fig)


def make_runtime_plot(
    t_sim: np.ndarray,
    t_surr: np.ndarray,
    outdir: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    t_sim = np.asarray(t_sim, dtype=float).ravel()
    t_surr = np.asarray(t_surr, dtype=float).ravel()
    t_sim = t_sim[np.isfinite(t_sim)]
    t_surr = t_surr[np.isfinite(t_surr)]

    if t_sim.size == 0 or t_surr.size == 0:
        raise ValueError("Runtime arrays are empty after filtering NaN/inf.")
    if np.any(t_sim <= 0) or np.any(t_surr <= 0):
        raise ValueError("Runtimes must be > 0 seconds to compute speedup and log-scale safely.")

    med_sim = float(np.median(t_sim))
    med_surr = float(np.median(t_surr))
    speedup = med_sim / med_surr

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    labels = ["Simulation/FEA", "Surrogate"]
    vals = [med_sim, med_surr]
    bars = ax.bar(labels, vals, color=["#4C78A8", "#54A24B"], width=0.6)

    if _maybe_log_runtime(med_sim, med_surr):
        ax.set_yscale("log")

    ax.set_ylabel("Runtime (s) per evaluation")
    ax.set_title(title)

    # Annotate bars with values
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * (1.12 if ax.get_yscale() == "log" else 1.02),
            f"{v:.3g}s",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    ax.text(
        0.5,
        0.92,
        f"Speedup = median(t_sim)/median(t_surr) = {speedup:.1f}×",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.95),
    )

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "runtime_speedup.png")
    fig.savefig(outdir / "runtime_speedup.pdf")
    plt.close(fig)


def main() -> int:
    _setup_matplotlib()

    ap = argparse.ArgumentParser(description="Generate parity + runtime figures for ISEF surrogate verification.")
    ap.add_argument("--csv", default="data.csv", help="Path to CSV file (default: data.csv)")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    ap.add_argument("--true-col", default=None, help="Column name for ground truth QoI")
    ap.add_argument("--pred-col", default=None, help="Column name for predicted QoI")
    ap.add_argument("--t-sim-col", default=None, help="Column name for simulation/FEA runtime (s)")
    ap.add_argument("--t-surr-col", default=None, help="Column name for surrogate runtime (s)")
    ap.add_argument("--t-sim-const", type=float, default=None, help="Constant simulation runtime (s) if column missing")
    ap.add_argument("--t-surr-const", type=float, default=None, help="Constant surrogate runtime (s) if column missing")
    ap.add_argument(
        "--band10",
        action="store_true",
        help="Add ±10 percent band on parity plot (only if values are positive)",
    )
    ap.add_argument("--parity-title", default="Surrogate Parity Plot (Held-out Test Set)", help="Parity plot title")
    ap.add_argument("--runtime-title", default="Runtime Comparison (Simulation/FEA vs Surrogate)", help="Runtime plot title")
    ap.add_argument(
        "--repo-mode",
        action="store_true",
        help="If set (or if --csv is missing), build figures from repo artifacts in pinn-workflow/surrogate_workflow/outputs/",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    use_repo_mode = bool(args.repo_mode) or (not csv_path.exists())

    # -----------------------------
    # Mode A: CSV input (user-provided)
    # -----------------------------
    if not use_repo_mode:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"ERROR: Failed to read CSV: {csv_path}\n{e}", file=sys.stderr)
            raise SystemExit(2)

        expectations = {
            "y_true": ["y_true", "true", "y", "target", "qoi_true", "simulation", "fea", "label", "ground_truth"],
            "y_pred": ["y_pred", "pred", "yhat", "prediction", "qoi_pred", "surrogate", "pinn", "model_pred"],
            "t_sim": ["t_sim", "sim_time", "fea_time", "runtime_sim", "runtime_fea", "time_sim", "seconds_sim"],
            "t_surr": ["t_surr", "surr_time", "runtime_surr", "runtime_surrogate", "time_surr", "seconds_surr"],
        }

        true_col = _pick_column(df, args.true_col, expectations["y_true"])
        pred_col = _pick_column(df, args.pred_col, expectations["y_pred"])
        missing = [k for k, v in (("y_true", true_col), ("y_pred", pred_col)) if v is None]
        if missing:
            _fail_missing_cols(df, missing, expectations)

        y_true = df[true_col].to_numpy()
        y_pred = df[pred_col].to_numpy()
        outdir = Path(args.outdir)

        try:
            make_parity_plot(
                y_true=y_true,
                y_pred=y_pred,
                outdir=outdir,
                title=args.parity_title,
                show_band_10pct=bool(args.band10),
            )
        except Exception as e:
            print(f"ERROR: Failed to create parity plot: {e}", file=sys.stderr)
            raise SystemExit(2)

        # Runtime (column or constants)
        t_sim_col = _pick_column(df, args.t_sim_col, expectations["t_sim"])
        t_surr_col = _pick_column(df, args.t_surr_col, expectations["t_surr"])

        t_sim_const = args.t_sim_const if args.t_sim_const is not None else MANUAL_T_SIM_SECONDS
        t_surr_const = args.t_surr_const if args.t_surr_const is not None else MANUAL_T_SURR_SECONDS

        if t_sim_col is not None and t_surr_col is not None:
            t_sim = df[t_sim_col].to_numpy()
            t_surr = df[t_surr_col].to_numpy()
        elif t_sim_const is not None and t_surr_const is not None:
            n = int(len(df))
            if n <= 0:
                print("ERROR: CSV has no rows; cannot build runtime comparison.", file=sys.stderr)
                raise SystemExit(2)
            t_sim = np.full((n,), float(t_sim_const), dtype=float)
            t_surr = np.full((n,), float(t_surr_const), dtype=float)
            print(
                "Note: Runtime columns not found; using constant runtimes "
                f"(t_sim={t_sim_const}s, t_surr={t_surr_const}s)."
            )
        else:
            print("\nWARNING: Runtime columns not found and constants not provided.", file=sys.stderr)
            print("Skipping runtime plot.", file=sys.stderr)
            print("Available columns:", ", ".join(map(str, df.columns)), file=sys.stderr)
            print(
                "To enable runtime plot, either add columns (t_sim, t_surr) or pass constants:\n"
                "  python3 make_isef_rri_figs.py --t-sim-const 10 --t-surr-const 0.01",
                file=sys.stderr,
            )
            return 0

        try:
            make_runtime_plot(t_sim=t_sim, t_surr=t_surr, outdir=outdir, title=args.runtime_title)
        except Exception as e:
            print(f"ERROR: Failed to create runtime plot: {e}", file=sys.stderr)
            raise SystemExit(2)

        print(f"Saved: {outdir / 'parity_plot.png'} and .pdf")
        print(f"Saved: {outdir / 'runtime_speedup.png'} and .pdf")
        return 0

    # -----------------------------
    # Mode B: Repo artifacts (no CSV)
    # Uses the 5D parametric surrogate workflow outputs already in the repo.
    # -----------------------------
    repo_root = Path(__file__).resolve().parent
    pinn_dir = repo_root / "pinn-workflow"
    if not pinn_dir.exists():
        print("ERROR: repo-mode expects `pinn-workflow/` next to this script.", file=sys.stderr)
        raise SystemExit(2)

    # Load dataset
    try:
        sys.path.insert(0, str(pinn_dir))
        from surrogate_workflow import config as sw_config  # type: ignore
        from surrogate_workflow import data as sw_data  # type: ignore
        from surrogate_workflow import surrogate as sw_surrogate  # type: ignore
    except Exception as e:
        print(f"ERROR: Failed to import surrogate_workflow modules: {e}", file=sys.stderr)
        raise SystemExit(2)

    dataset_path = Path(getattr(sw_config, "DATASET_PATH", pinn_dir / "surrogate_workflow" / "outputs" / "phase1_dataset.npz"))
    model_path = Path(getattr(sw_config, "MODEL_PATH", pinn_dir / "surrogate_workflow" / "outputs" / "surrogate_model.pt"))
    if not dataset_path.exists():
        print(f"ERROR: Repo dataset not found: {dataset_path}", file=sys.stderr)
        raise SystemExit(2)
    if not model_path.exists():
        print(f"ERROR: Repo surrogate model not found: {model_path}", file=sys.stderr)
        raise SystemExit(2)

    dataset = sw_data.load_dataset(str(dataset_path))
    x_norm = np.asarray(dataset["x_norm"], dtype=float)
    y_true_all = np.asarray(dataset["y_raw"], dtype=float)

    # Load model payload + build net
    try:
        import torch
    except Exception as e:
        print(f"ERROR: Repo-mode requires torch to run the surrogate model: {e}", file=sys.stderr)
        raise SystemExit(2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(model_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = sw_surrogate.MLPRegressor(
        input_dim=len(payload.get("param_names", dataset.get("param_names", []))),
        output_dim=1,
        hidden_layers=int(cfg.get("hidden_layers", 4)),
        hidden_units=int(cfg.get("hidden_units", 128)),
        activation=str(cfg.get("activation", "tanh")),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    y_norm_pred = sw_surrogate.predict(model, x_norm.astype(np.float32), device)
    y_min = float(dataset["y_min"])
    y_max = float(dataset["y_max"])
    y_pred_all = y_norm_pred * (y_max - y_min) + y_min

    # Reproducible split: use workflow's seed + fractions
    seed = int(getattr(sw_config, "SEED", 7))
    train_frac = float(getattr(sw_config, "TRAIN_FRACTION", 0.8))
    val_frac = float(getattr(sw_config, "VAL_FRACTION", 0.1))
    _, _, test_idx = sw_data.split_indices(len(y_true_all), train_frac, val_frac, seed)

    outdir = Path(args.outdir)

    try:
        make_parity_plot(
            y_true=y_true_all[test_idx],
            y_pred=y_pred_all[test_idx],
            outdir=outdir,
            title=args.parity_title or "Surrogate Parity Plot (Held-out Test Set)",
            show_band_10pct=bool(args.band10),
        )
    except Exception as e:
        print(f"ERROR: Failed to create parity plot: {e}", file=sys.stderr)
        raise SystemExit(2)

    t_sim_const = args.t_sim_const if args.t_sim_const is not None else MANUAL_T_SIM_SECONDS
    t_surr_const = args.t_surr_const if args.t_surr_const is not None else MANUAL_T_SURR_SECONDS

    # Runtime: if no constants provided, benchmark from repo workflow
    if t_sim_const is not None and t_surr_const is not None:
        t_sim = np.full((50,), float(t_sim_const), dtype=float)
        t_surr = np.full((50,), float(t_surr_const), dtype=float)
        print(f"Note: Using constant runtimes (t_sim={t_sim_const}s, t_surr={t_surr_const}s).")
    else:
        try:
            import time
            from surrogate_workflow import baseline as sw_baseline  # type: ignore
            from surrogate_api import ParametricSurrogate  # type: ignore
        except Exception as e:
            print(
                "WARNING: Could not import runtime benchmark dependencies; skipping runtime plot.\n"
                f"Reason: {e}",
                file=sys.stderr,
            )
            return 0

        # Representative point: midpoint of design space (in raw units)
        x_raw = np.asarray(dataset["x_raw"], dtype=float)
        mu = np.median(x_raw, axis=0)

        # Warmups (exclude compilation/cache effects)
        _ = sw_baseline.compute_response(mu)
        api = ParametricSurrogate(model_path=str(model_path))
        _ = api.predict(dict(zip(dataset["param_names"], mu, strict=True)))

        n_sim = 15
        n_surr = 200
        t_sim = np.zeros((n_sim,), dtype=float)
        t_surr = np.zeros((n_surr,), dtype=float)

        for i in range(n_sim):
            t0 = time.perf_counter()
            _ = sw_baseline.compute_response(mu)
            t_sim[i] = time.perf_counter() - t0

        params = dict(zip(dataset["param_names"], mu, strict=True))
        for i in range(n_surr):
            t0 = time.perf_counter()
            _ = api.predict(params)
            t_surr[i] = time.perf_counter() - t0

    try:
        make_runtime_plot(t_sim=t_sim, t_surr=t_surr, outdir=outdir, title=args.runtime_title)
    except Exception as e:
        print(f"ERROR: Failed to create runtime plot: {e}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Saved: {outdir / 'parity_plot.png'} and .pdf")
    print(f"Saved: {outdir / 'runtime_speedup.png'} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
