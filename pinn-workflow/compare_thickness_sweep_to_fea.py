import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver


@dataclass(frozen=True)
class Case:
    name: str
    t1: float
    t2: float
    t3: float


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    first = sd.get("layers.0.net.0.weight")
    if first is None:
        raise ValueError("Checkpoint does not look like 3-layer PINN (missing layers.0.net.0.weight).")
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


def _parse_cases(spec: str | None, H: float) -> list[Case]:
    if not spec:
        return [
            Case("balanced", H / 3.0, H / 3.0, H / 3.0),
            Case("soft_core", 0.2 * H, 0.6 * H, 0.2 * H),
            Case("thick_top", 0.2 * H, 0.2 * H, 0.6 * H),
            Case("thick_bottom", 0.6 * H, 0.2 * H, 0.2 * H),
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
        if len(parts) != 3:
            raise ValueError(f"Invalid case '{item}'. Expected 'name:t1,t2,t3' or 't1,t2,t3'.")
        t1, t2, t3 = (float(parts[0]), float(parts[1]), float(parts[2]))
        cases.append(Case(name, t1, t2, t3))
    return cases


def _metrics(u_pred: np.ndarray, u_true: np.ndarray) -> dict:
    err = u_pred - u_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Thickness sweep: run layered FEA and compare to PINN predictions.")
    ap.add_argument("--model", default="pinn_model.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="thickness_fea_compare")

    ap.add_argument("--Lx", type=float, default=float(getattr(config, "Lx", 1.0)))
    ap.add_argument("--Ly", type=float, default=float(getattr(config, "Ly", 1.0)))
    ap.add_argument("--H", type=float, default=float(getattr(config, "H", 0.1)))
    ap.add_argument("--nu", type=float, default=float(getattr(config, "NU_FIXED", 0.3)))
    ap.add_argument("--E1", type=float, default=1.0)
    ap.add_argument("--E2", type=float, default=5.0)
    ap.add_argument("--E3", type=float, default=10.0)
    ap.add_argument("--p0", type=float, default=float(getattr(config, "p0", 1.0)))
    ap.add_argument("--use_soft_mask", type=int, default=int(getattr(config, "USE_SOFT_LOAD_MASK", True)))

    ap.add_argument("--ne_x", type=int, default=20)
    ap.add_argument("--ne_y", type=int, default=20)
    ap.add_argument("--ne_z", type=int, default=20)

    ap.add_argument("--cases", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = _select_device(args.device)
    sd = torch.load(args.model, map_location=device)
    layers, neurons = _infer_arch_from_state_dict(sd)
    config.LAYERS = int(layers)
    config.NEURONS = int(neurons)
    pinn = model.MultiLayerPINN().to(device)
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    H = float(args.H)
    cases = _parse_cases(args.cases, H)
    normalized: list[Case] = []
    for c in cases:
        s = float(c.t1 + c.t2 + c.t3)
        scale = H / max(1e-12, s)
        normalized.append(Case(c.name, c.t1 * scale, c.t2 * scale, c.t3 * scale))
    cases = normalized

    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1 = map(float, config.LOAD_PATCH_X)
    y0, y1 = map(float, config.LOAD_PATCH_Y)

    for c in cases:
        cfg = {
            "geometry": {"Lx": float(args.Lx), "Ly": float(args.Ly), "H": H},
            "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
            "layers": [
                {"t": float(c.t1), "E": float(args.E1), "nu": float(args.nu)},
                {"t": float(c.t2), "E": float(args.E2), "nu": float(args.nu)},
                {"t": float(c.t3), "E": float(args.E3), "nu": float(args.nu)},
            ],
            "load_patch": {
                "pressure": float(args.p0),
                "x_start": x0 / float(args.Lx),
                "x_end": x1 / float(args.Lx),
                "y_start": y0 / float(args.Ly),
                "y_end": y1 / float(args.Ly),
            },
            "use_soft_mask": bool(int(args.use_soft_mask)),
        }

        x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_fem(cfg)
        X, Y, Z = np.meshgrid(np.array(x_nodes), np.array(y_nodes), np.array(z_nodes), indexing="ij")
        u_true = np.array(u_fea, dtype=np.float32)

        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        params = np.array([args.E1, c.t1, args.E2, c.t2, args.E3, c.t3, 0.5, 0.3, 1.0], dtype=np.float32)[None, :]
        inp = np.concatenate([pts, np.repeat(params, pts.shape[0], axis=0)], axis=1)
        with torch.no_grad():
            u_pred = pinn(torch.tensor(inp, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32).reshape(u_true.shape)

        # Metrics
        flat_pred = u_pred.reshape(-1, 3)
        flat_true = u_true.reshape(-1, 3)
        m_all = _metrics(flat_pred, flat_true)

        # TOP + patch masks
        top = np.isclose(Z, H)
        patch = top & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        free_top = top & (~patch)

        def _maybe(mask: np.ndarray) -> dict:
            if int(mask.sum()) == 0:
                return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
            return _metrics(flat_pred[mask.reshape(-1)], flat_true[mask.reshape(-1)])

        m_top = _maybe(top)
        m_patch = _maybe(patch)
        m_free = _maybe(free_top)
        print(
            f"{c.name}: ALL mae={m_all['mae']:.4f} | TOP mae={m_top['mae']:.4f} | PATCH mae={m_patch['mae']:.4f} | FREE mae={m_free['mae']:.4f}"
        )

        # Plots: top uz + xz slice at mid y
        uz_fea_top = u_true[:, :, -1, 2]
        uz_pinn_top = u_pred[:, :, -1, 2]
        vmin = float(min(uz_fea_top.min(), uz_pinn_top.min()))
        vmax = float(max(uz_fea_top.max(), uz_pinn_top.max()))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].contourf(X[:, :, 0], Y[:, :, 0], uz_fea_top, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title("FEA $u_z$ (top)")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].contourf(X[:, :, 0], Y[:, :, 0], uz_pinn_top, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title("PINN $u_z$ (top)")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].contourf(X[:, :, 0], Y[:, :, 0], np.abs(uz_pinn_top - uz_fea_top), 50, cmap="magma")
        axes[2].set_title("|error| (top)")
        plt.colorbar(im2, ax=axes[2])
        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
        fig.tight_layout()
        out_top = os.path.join(args.out_dir, f"{c.name}_top.png")
        fig.savefig(out_top, dpi=150)
        plt.close(fig)

        mid_y = len(y_nodes) // 2
        Xs = X[:, mid_y, :]
        Zs = Z[:, mid_y, :]
        uz_fea_xz = u_true[:, mid_y, :, 2]
        uz_pinn_xz = u_pred[:, mid_y, :, 2]
        vmin = float(min(uz_fea_xz.min(), uz_pinn_xz.min()))
        vmax = float(max(uz_fea_xz.max(), uz_pinn_xz.max()))
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes2[0].contourf(Xs, Zs, uz_fea_xz, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes2[0].set_title("FEA $u_z$ (XZ)")
        plt.colorbar(im0, ax=axes2[0])
        im1 = axes2[1].contourf(Xs, Zs, uz_pinn_xz, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes2[1].set_title("PINN $u_z$ (XZ)")
        plt.colorbar(im1, ax=axes2[1])
        im2 = axes2[2].contourf(Xs, Zs, np.abs(uz_pinn_xz - uz_fea_xz), 50, cmap="magma")
        axes2[2].set_title("|error| (XZ)")
        plt.colorbar(im2, ax=axes2[2])
        for ax in axes2:
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.axhline(c.t1, color="k", lw=1.0)
            ax.axhline(c.t1 + c.t2, color="k", lw=1.0)
        fig2.tight_layout()
        out_xz = os.path.join(args.out_dir, f"{c.name}_xz.png")
        fig2.savefig(out_xz, dpi=150)
        plt.close(fig2)

        print(f"  wrote: {out_top}")
        print(f"  wrote: {out_xz}")


if __name__ == "__main__":
    main()

