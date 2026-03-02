from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
import physics


@dataclass(frozen=True)
class Case:
    name: str
    t: tuple[float, ...]


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    first = sd.get("layers.0.net.0.weight")
    if first is None:
        raise ValueError("Checkpoint does not look like a layered PINN (missing layers.0.net.0.weight).")
    neurons = int(first.shape[0])
    linear_indices = sorted(
        {
            int(k.split(".")[3])
            for k in sd.keys()
            if isinstance(k, str) and k.startswith("layers.0.net.") and k.endswith(".weight") and k.split(".")[3].isdigit()
        }
    )
    num_linears = len(linear_indices)
    hidden_layers = max(1, num_linears - 1)
    return hidden_layers, neurons


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_cases(spec: str | None, H: float, *, num_layers: int) -> list[Case]:
    if not spec:
        if num_layers == 2:
            return [
                Case("balanced", (H / 2.0, H / 2.0)),
                Case("thick_top", (0.35 * H, 0.65 * H)),
                Case("thick_bottom", (0.65 * H, 0.35 * H)),
            ]
        return [
            Case("balanced", (H / 3.0, H / 3.0, H / 3.0)),
            Case("soft_core", (0.2 * H, 0.6 * H, 0.2 * H)),
            Case("thick_top", (0.2 * H, 0.2 * H, 0.6 * H)),
            Case("thick_bottom", (0.6 * H, 0.2 * H, 0.2 * H)),
            Case("thin_mid", (0.45 * H, 0.10 * H, 0.45 * H)),
            Case("thick_mid", (0.10 * H, 0.80 * H, 0.10 * H)),
        ]

    cases: list[Case] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, vals = item.split(":", 1)
            name = name.strip()
        else:
            name, vals = f"case{len(cases)}", item
        parts = [p.strip() for p in vals.split(",")]
        if len(parts) != int(num_layers):
            raise ValueError(
                f"Invalid case '{item}'. Expected {num_layers} thicknesses like 'name:t1,...,t{num_layers}'."
            )
        cases.append(Case(name, tuple(float(p) for p in parts)))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description="PINN-only sweep: vary (t1,t2,t3) and visualize top and XZ uz.")
    ap.add_argument("--model", default="pinn_model.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_top", default="thickness_sweep_top.png")
    ap.add_argument("--out_xz", default="thickness_sweep_xz.png")
    ap.add_argument("--nx", type=int, default=101)
    ap.add_argument("--nz", type=int, default=81)
    ap.add_argument("--H", type=float, default=float(getattr(config, "H", 0.1)))
    ap.add_argument("--y0", type=float, default=0.5)

    ap.add_argument("--E1", type=float, default=1.0)
    ap.add_argument("--E2", type=float, default=5.0)
    ap.add_argument("--E3", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=float(getattr(config, "RESTITUTION_REF", 0.5)))
    ap.add_argument("--mu", type=float, default=float(getattr(config, "FRICTION_REF", 0.3)))
    ap.add_argument("--v0", type=float, default=float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)))
    ap.add_argument("--cases", default=None)
    args = ap.parse_args()

    device = _select_device(args.device)

    model_path = args.model
    if not os.path.exists(model_path):
        alt = os.path.join(PINN_WORKFLOW_DIR, os.path.basename(model_path))
        if os.path.exists(alt):
            model_path = alt
    sd = torch.load(model_path, map_location=device)
    layers, neurons = _infer_arch_from_state_dict(sd)
    config.LAYERS = int(layers)
    config.NEURONS = int(neurons)

    pinn = model.MultiLayerPINN().to(device)
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    H = float(args.H)
    num_layers = int(getattr(config, "NUM_LAYERS", 2))
    cases = _parse_cases(args.cases, H, num_layers=num_layers)
    normalized: list[Case] = []
    for c in cases:
        s = float(sum(c.t))
        if s <= 0:
            raise ValueError(f"Invalid thickness sum for case {c.name}: {s}")
        scale = H / s
        normalized.append(Case(c.name, tuple(float(ti) * scale for ti in c.t)))
    cases = normalized

    x = np.linspace(0.0, float(getattr(config, "Lx", 1.0)), int(args.nx), dtype=np.float32)
    y = np.linspace(0.0, float(getattr(config, "Ly", 1.0)), int(args.nx), dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Xf = X.ravel()
    Yf = Y.ravel()

    z_line = np.linspace(0.0, H, int(args.nz), dtype=np.float32)
    X2, Z2 = np.meshgrid(x, z_line, indexing="xy")

    x0, x1 = map(float, config.LOAD_PATCH_X)
    y0, y1 = map(float, config.LOAD_PATCH_Y)
    patch = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)

    top_maps = []
    xz_maps = []
    stats = []

    for c in cases:
        E_vals = [float(args.E1), float(args.E2), float(args.E3)][:num_layers]
        params_list = []
        for Ei, ti in zip(E_vals, c.t):
            params_list.extend([float(Ei), float(ti)])
        params_list.extend([float(args.r), float(args.mu), float(args.v0)])
        params = np.array(params_list, dtype=np.float32)[None, :]

        inp_top = np.concatenate(
            [np.stack([Xf, Yf, np.full_like(Xf, H)], axis=1).astype(np.float32), np.repeat(params, Xf.shape[0], axis=0)],
            axis=1,
        )
        with torch.no_grad():
            x_t = torch.tensor(inp_top, dtype=torch.float32, device=device)
            v = pinn(x_t)
            uz_top = physics.decode_u(v, x_t)[:, 2].cpu().numpy().reshape(X.shape)
        top_maps.append(uz_top)
        stats.append((c.name, float(uz_top[patch].mean()), float(uz_top[patch].min())))

        inp_xz = np.concatenate(
            [
                np.stack([X2.ravel(), np.full_like(X2.ravel(), float(args.y0)), Z2.ravel()], axis=1).astype(np.float32),
                np.repeat(params, X2.size, axis=0),
            ],
            axis=1,
        )
        with torch.no_grad():
            x_t = torch.tensor(inp_xz, dtype=torch.float32, device=device)
            v = pinn(x_t)
            uz_xz = physics.decode_u(v, x_t)[:, 2].cpu().numpy().reshape(Z2.shape)
        xz_maps.append(uz_xz)

    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    rows = int(np.ceil(len(cases) / cols))

    vmin = float(np.min([m.min() for m in top_maps]))
    vmax = float(np.max([m.max() for m in top_maps]))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.0 * rows), squeeze=False)
    for i, (c, uz) in enumerate(zip(cases, top_maps, strict=False)):
        ax = axes[i // cols][i % cols]
        im = ax.contourf(X, Y, uz, levels=60, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"{c.name}\n(t)=({','.join(f'{float(ti):.4f}' for ti in c.t)})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
        fig.colorbar(im, ax=ax, shrink=0.85)
    for j in range(len(cases), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(args.out_top, dpi=160)
    print(f"Wrote: {args.out_top}")

    vmin_xz = float(np.min([m.min() for m in xz_maps]))
    vmax_xz = float(np.max([m.max() for m in xz_maps]))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.0 * rows), squeeze=False)
    for i, (c, uz) in enumerate(zip(cases, xz_maps, strict=False)):
        ax = axes2[i // cols][i % cols]
        im = ax.contourf(X2, Z2, uz, levels=60, cmap="jet", vmin=vmin_xz, vmax=vmax_xz)
        ax.set_title(f"{c.name} (y={float(args.y0):.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        z_acc = 0.0
        for ti in c.t[:-1]:
            z_acc += float(ti)
            ax.axhline(z_acc, color="k", lw=1.0)
        fig2.colorbar(im, ax=ax, shrink=0.85)
    for j in range(len(cases), rows * cols):
        axes2[j // cols][j % cols].axis("off")
    fig2.tight_layout()
    fig2.savefig(args.out_xz, dpi=160)
    print(f"Wrote: {args.out_xz}")

    print("\nPatch summary (PINN top surface uz):")
    for name, mean_uz, min_uz in stats:
        print(f"  {name:>12s}: mean_uz={mean_uz:+.6f} min_uz={min_uz:+.6f}")


if __name__ == "__main__":
    main()
