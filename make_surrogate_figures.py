#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _setup_matplotlib_cache(repo_root: Path) -> None:
    cache_dir = repo_root / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _setup_mpl_style() -> None:
    import matplotlib as mpl

    # Force a non-interactive backend for headless / automation contexts.
    mpl.use("Agg", force=True)

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def _save_figure(fig, outdir: Path, stem: str, dpi: int, formats: Iterable[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=dpi)


def _load_loss_history(repo_root: Path) -> dict:
    candidates = [
        repo_root / "loss_history.npy",
        repo_root / "pinn-workflow" / "loss_history.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.load(p, allow_pickle=True).item()
    raise FileNotFoundError("Could not find loss_history.npy in repo root or pinn-workflow/")


def _load_fea_solution(repo_root: Path) -> dict:
    candidates = [
        repo_root / "fea_solution.npy",
        repo_root / "pinn-workflow" / "fea_solution.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.load(p, allow_pickle=True).item()
    raise FileNotFoundError("Could not find fea_solution.npy in repo root or pinn-workflow/")


def _import_pinn(repo_root: Path):
    import sys

    pinn_dir = repo_root / "pinn-workflow"
    if str(pinn_dir) not in sys.path:
        sys.path.insert(0, str(pinn_dir))

    import pinn_config as config  # type: ignore
    import model  # type: ignore

    return config, model


def make_training_loss_figure(repo_root: Path, outdir: Path, dpi: int, formats: list[str]) -> None:
    import matplotlib.pyplot as plt

    hist = _load_loss_history(repo_root)
    adam = hist.get("adam", {})
    total_len = len(adam.get("total", []))
    epochs_raw = np.asarray(adam.get("epochs", []))
    epochs = epochs_raw if epochs_raw.size == total_len else np.arange(total_len)

    series_specs = [
        ("total", "Total"),
        ("pde", "PDE"),
        ("bc_sides", "BC (sides)"),
        ("load", "Load"),
        ("impact_contact", "Impact contact"),
        ("friction_coulomb", "Friction (Coulomb)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.3))
    for key, label in series_specs:
        if key not in adam:
            continue
        y = np.asarray(adam[key], dtype=float)
        if y.size == 0:
            continue
        ax.plot(epochs[: y.size], y, linewidth=2.0 if key == "total" else 1.6, label=label)

    ax.set_title("Surrogate Training Loss (Adam)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_yscale("log")
    ax.legend(ncol=2, frameon=True, fancybox=True)
    ax.grid(True, which="both", alpha=0.25)

    _save_figure(fig, outdir, "surrogate_training_loss", dpi=dpi, formats=formats)
    plt.close(fig)


@dataclass(frozen=True)
class _PinnInputs:
    e: float
    thickness: float
    restitution: float
    friction: float
    impact_velocity: float


def _default_pinn_inputs(config) -> _PinnInputs:
    e = float(getattr(config, "E_vals", [1.0])[0])
    thickness = float(getattr(config, "H", 0.1))
    restitution = float(getattr(config, "RESTITUTION_REF", 0.5))
    friction = float(getattr(config, "FRICTION_REF", 0.3))
    impact_velocity = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    return _PinnInputs(e, thickness, restitution, friction, impact_velocity)


def _load_pinn_model(repo_root: Path, model_module, device):
    import torch

    candidates = [
        repo_root / "pinn_model.pth",
        repo_root / "pinn-workflow" / "pinn_model.pth",
    ]
    for p in candidates:
        if p.exists():
            pinn = model_module.MultiLayerPINN().to(device)
            pinn.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            pinn.eval()
            return pinn, p
    raise FileNotFoundError("Could not find pinn_model.pth in repo root or pinn-workflow/")


def _predict_on_fea_grid(repo_root: Path) -> tuple[np.ndarray, dict]:
    import torch

    config, model_module = _import_pinn(repo_root)
    fea = _load_fea_solution(repo_root)
    X = fea["x"]
    Y = fea["y"]
    Z = fea["z"]
    U_fea = fea["u"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn, model_path = _load_pinn_model(repo_root, model_module, device)
    inputs = _default_pinn_inputs(config)

    pts_xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    n = pts_xyz.shape[0]
    params = np.column_stack(
        [
            np.full((n,), inputs.e, dtype=np.float32),
            np.full((n,), inputs.thickness, dtype=np.float32),
            np.full((n,), inputs.restitution, dtype=np.float32),
            np.full((n,), inputs.friction, dtype=np.float32),
            np.full((n,), inputs.impact_velocity, dtype=np.float32),
        ]
    )
    pts = np.concatenate([pts_xyz, params], axis=1)  # (N, 8)

    u_pred = np.zeros((n, 3), dtype=np.float32)
    batch = 8192
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = torch.tensor(pts[i : i + batch], dtype=torch.float32, device=device)
            u_pred[i : i + batch] = pinn(chunk).detach().cpu().numpy()

    U_pinn = u_pred.reshape(U_fea.shape)
    meta = {
        "device": str(device),
        "model_path": str(model_path),
        "inputs": inputs,
    }
    return U_pinn, meta


def make_validation_figure(repo_root: Path, outdir: Path, dpi: int, formats: list[str]) -> None:
    import matplotlib.pyplot as plt

    fea = _load_fea_solution(repo_root)
    X = fea["x"]
    Y = fea["y"]
    U_fea = fea["u"]

    U_pinn, meta = _predict_on_fea_grid(repo_root)
    u_z_fea_top = U_fea[:, :, -1, 2]
    u_z_pinn_top = U_pinn[:, :, -1, 2]
    abs_err = np.abs(u_z_fea_top - u_z_pinn_top)

    mae = float(np.mean(abs_err))
    max_err = float(np.max(abs_err))
    denom = float(np.max(np.abs(u_z_fea_top))) if np.max(np.abs(u_z_fea_top)) != 0 else 1.0
    mae_pct = (mae / denom) * 100.0

    vmin = float(min(u_z_fea_top.min(), u_z_pinn_top.min()))
    vmax = float(max(u_z_fea_top.max(), u_z_pinn_top.max()))

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], wspace=0.22, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    c1 = ax1.contourf(X[:, :, 0], Y[:, :, 0], u_z_fea_top, levels=60, cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("FEA: Top Surface $u_z$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)

    c2 = ax2.contourf(X[:, :, 0], Y[:, :, 0], u_z_pinn_top, levels=60, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("Surrogate (PINN): Top Surface $u_z$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

    c3 = ax3.contourf(X[:, :, 0], Y[:, :, 0], abs_err, levels=60, cmap="magma")
    ax3.set_title("Absolute Error $|u_z^{FEA} - u_z^{PINN}|$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig.colorbar(c3, ax=ax3, fraction=0.046, pad=0.04)

    # Parity plot
    x = u_z_fea_top.ravel()
    y = u_z_pinn_top.ravel()
    ax4.scatter(x, y, s=10, alpha=0.35, edgecolors="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax4.plot([lo, hi], [lo, hi], color="black", linewidth=1.5, label="y = x")
    ax4.set_title("Parity Plot (Top Surface $u_z$)")
    ax4.set_xlabel("FEA $u_z$")
    ax4.set_ylabel("PINN $u_z$")
    ax4.legend(frameon=True, fancybox=True, loc="best")

    inputs = meta["inputs"]
    fig.suptitle(
        "Surrogate Validation vs FEA (Top Surface)\n"
        f"MAE={mae:.3e} ({mae_pct:.2f}% of max |FEA|), MaxErr={max_err:.3e} • "
        f"E={inputs.e:g}, t={inputs.thickness:g}, r={inputs.restitution:g}, μ={inputs.friction:g}, v0={inputs.impact_velocity:g}",
        y=0.98,
        fontsize=14,
    )

    _save_figure(fig, outdir, "surrogate_validation_top_surface", dpi=dpi, formats=formats)
    plt.close(fig)


def make_architecture_figure(repo_root: Path, outdir: Path, dpi: int, formats: list[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    config, _ = _import_pinn(repo_root)
    layers = int(getattr(config, "LAYERS", 4))
    neurons = int(getattr(config, "NEURONS", 64))
    fourier_dim = int(getattr(config, "FOURIER_DIM", 0))
    use_hard_bc = bool(getattr(config, "USE_HARD_SIDE_BC", False))

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, body, fc="#F7F7FB", ec="#3B3B44"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02 * w, y + h - 0.27 * h, title, fontsize=13, fontweight="bold", va="center")
        ax.text(x + 0.02 * w, y + 0.12 * h, body, fontsize=11.2, va="bottom", linespacing=1.25)

    def arrow(x0, y0, x1, y1, text=None):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#2E2E38"),
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, text, ha="center", va="bottom", fontsize=11)

    # Row 1: Inputs -> Feature engineering -> Embedding
    box(
        0.06,
        0.62,
        0.26,
        0.28,
        "Inputs (8D)",
        "Spatial:  x, y, z\n"
        "Params:   E, thickness t, restitution r,\n"
        "          friction μ, impact velocity v₀",
        fc="#EEF6FF",
    )
    box(
        0.37,
        0.62,
        0.26,
        0.28,
        "Feature Engineering",
        "Normalize params to [0,1]\n"
        "Scaled depth:  ẑ = z / t\n"
        "Physics feats: (H/t), (H/t)², (H/t)³",
        fc="#F1FFF4",
    )
    emb_body = "Concat: [x, y, ẑ, E*, t*, r*, μ*, v₀*, physics feats]"
    if fourier_dim > 0:
        emb_body += f"\nFourier on (x,y,ẑ): 2×{fourier_dim} sin/cos"
    else:
        emb_body += "\nFourier features: disabled"
    box(
        0.68,
        0.62,
        0.26,
        0.28,
        "Model Input Vector",
        emb_body,
        fc="#FFF7EE",
    )

    arrow(0.32, 0.76, 0.37, 0.76)
    arrow(0.63, 0.76, 0.68, 0.76)

    # Row 2: MLP -> Outputs -> Constraints
    mlp_body = (
        f"MLP: {layers} hidden layers × {neurons} neurons\n"
        "Activation: tanh\n"
        "Output: 3D displacement (uₓ, uᵧ, u_z)"
    )
    box(0.18, 0.23, 0.30, 0.28, "Surrogate Network", mlp_body, fc="#F7F1FF")

    out_body = "Displacement field:\n(uₓ, uᵧ, u_z) = fθ(x, y, z, params)"
    if use_hard_bc:
        out_body += "\nHard side-BC mask:\n× 16·x(1−x)·y(1−y)"
    box(0.55, 0.23, 0.30, 0.28, "Predictions", out_body, fc="#FFF1F3")

    box(
        0.06,
        0.05,
        0.88,
        0.12,
        "Physics + Boundary Loss (Training-Time Constraints)",
        "PDE: div(σ(u)) = 0 (scaled) • BCs: clamped sides, free surfaces, load patch • "
        "Interface continuity between layers (u continuity)",
        fc="#F7F7FB",
    )

    arrow(0.81, 0.62, 0.33, 0.51, text="forward pass")
    arrow(0.48, 0.37, 0.55, 0.37)
    arrow(0.70, 0.23, 0.50, 0.17, text="constraints\n& loss")

    ax.text(
        0.5,
        0.96,
        "Surrogate Model Overview (PINN-style Neural Surrogate)",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.915,
        f"Config: LAYERS={layers}, NEURONS={neurons}, FOURIER_DIM={fourier_dim}, USE_HARD_SIDE_BC={use_hard_bc}",
        ha="center",
        va="center",
        fontsize=12,
        color="#3B3B44",
    )

    _save_figure(fig, outdir, "surrogate_model_architecture", dpi=dpi, formats=formats)
    plt.close(fig)


def _parse_formats(value: str) -> list[str]:
    fmts = [v.strip().lower() for v in value.split(",") if v.strip()]
    allowed = {"png", "pdf", "svg"}
    bad = [f for f in fmts if f not in allowed]
    if bad:
        raise SystemExit(f"Unsupported format(s): {bad}. Allowed: {sorted(allowed)}")
    return fmts


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    _setup_matplotlib_cache(repo_root)
    _setup_mpl_style()

    ap = argparse.ArgumentParser(description="Generate poster-ready surrogate model figures.")
    ap.add_argument(
        "--outdir",
        default=str(repo_root / "isef graphs" / "poster_baselines"),
        help="Output directory for figures",
    )
    ap.add_argument("--dpi", type=int, default=600, help="Output DPI for raster formats")
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated: png,pdf,svg")
    ap.add_argument(
        "--which",
        default="architecture,validation,loss",
        help="Comma-separated: architecture,validation,loss",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    formats = _parse_formats(args.formats)
    which = {w.strip().lower() for w in str(args.which).split(",") if w.strip()}

    if "architecture" in which:
        make_architecture_figure(repo_root, outdir, dpi=args.dpi, formats=formats)
    if "validation" in which:
        make_validation_figure(repo_root, outdir, dpi=args.dpi, formats=formats)
    if "loss" in which:
        make_training_loss_figure(repo_root, outdir, dpi=args.dpi, formats=formats)

    print(f"Saved figures to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
