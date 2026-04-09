from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from _common import REPO_ROOT, apply_ieee_style, save_figure, watermark_placeholder, print_inputs_used

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


@dataclass(frozen=True)
class Case:
    name: str
    e: tuple[float, float, float]
    t: tuple[float, float, float]


def _import_pinn_stack():
    pinn_dir = REPO_ROOT / "pinn-workflow"
    fea_dir = REPO_ROOT / "fea-workflow" / "solver"
    for p in (pinn_dir, fea_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import pinn_config as pc  # noqa: WPS433
    import model  # noqa: WPS433
    import fem_solver  # noqa: WPS433
    return pc, model, fem_solver


def _u_from_v(pc, v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # Consistent with training/evaluation scripts.
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(pc, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(pc, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(pc, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(pc, "H", 1.0))
    return scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _predict_top_u_z(pc, pinn, device, x_nodes: np.ndarray, y_nodes: np.ndarray, thickness: float, e, t) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    z = np.full(x_grid.size, thickness, dtype=float)

    r_ref = float(getattr(pc, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(pc, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(pc, "IMPACT_VELOCITY_REF", 1.0))

    e1, e2, e3 = e
    t1, t2, t3 = t
    pts = np.stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            z,
            np.full_like(z, e1),
            np.full_like(z, t1),
            np.full_like(z, e2),
            np.full_like(z, t2),
            np.full_like(z, e3),
            np.full_like(z, t3),
            np.full_like(z, r_ref),
            np.full_like(z, mu_ref),
            np.full_like(z, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    u = _u_from_v(pc, v, pts)
    return u.reshape(len(x_nodes), len(y_nodes), 3)[:, :, 2]


def main() -> None:
    apply_ieee_style()

    # Default representative case aligns with the evaluator naming.
    case = Case(
        name=os.getenv("PINN_ERROR_CASE", "three_layer_soft_bottom"),
        e=tuple(float(x) for x in os.getenv("PINN_ERROR_E", "1.0,10.0,10.0").split(",")),  # type: ignore[arg-type]
        t=tuple(float(x) for x in os.getenv("PINN_ERROR_T", "0.10,0.02,0.02").split(",")),  # type: ignore[arg-type]
    )
    if len(case.e) != 3 or len(case.t) != 3:
        raise ValueError("Expected PINN_ERROR_E and PINN_ERROR_T to have 3 comma-separated values each.")

    model_path = os.getenv("PINN_MODEL_PATH") or str(REPO_ROOT / "pinn-workflow" / "pinn_model.pth")
    model_path_p = Path(model_path)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    if not model_path_p.exists():
        ax.set_title("Error Heatmap (Top Surface)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        watermark_placeholder(ax, "PLACEHOLDER\n(missing PINN checkpoint)")
        ax.text(
            0.02,
            0.02,
            f"Missing:\n- {model_path_p}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
        out_paths = save_figure(fig, "fig_error_heatmap_placeholder")
        plt.close(fig)
        print("Wrote placeholder error-heatmap figure.")
        for p in out_paths:
            print(f"  - {p}")
        return

    pc, model, fem_solver = _import_pinn_stack()
    print_inputs_used([model_path_p])

    device = torch.device("cpu") if os.getenv("PINN_FORCE_CPU", "0") == "1" else (
        torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
    )

    pinn = model.MultiLayerPINN().to(device)
    sd = torch.load(str(model_path_p), map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    e1, e2, e3 = case.e
    t1, t2, t3 = case.t
    thickness = float(t1 + t2 + t3)

    cfg = {
        "geometry": {"Lx": pc.Lx, "Ly": pc.Ly, "H": thickness, "ne_x": 10, "ne_y": 10, "ne_z": 4},
        "material": {"E_layers": [e1, e2, e3], "t_layers": [t1, t2, t3], "nu": pc.nu_vals[0]},
        "load_patch": {
            "pressure": pc.p0,
            "x_start": pc.LOAD_PATCH_X[0] / pc.Lx,
            "x_end": pc.LOAD_PATCH_X[1] / pc.Lx,
            "y_start": pc.LOAD_PATCH_Y[0] / pc.Ly,
            "y_end": pc.LOAD_PATCH_Y[1] / pc.Ly,
        },
    }
    x_nodes, y_nodes, _z_nodes, u_fea = fem_solver.solve_three_layer_fem(cfg)
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    u_z_fea = np.asarray(u_fea, dtype=float)[:, :, -1, 2]
    u_z_pred = _predict_top_u_z(pc, pinn, device, x_nodes, y_nodes, thickness, case.e, case.t)

    abs_err = np.abs(u_z_pred - u_z_fea)

    im = ax.imshow(
        abs_err.T,
        origin="lower",
        extent=(float(x_nodes.min()), float(x_nodes.max()), float(y_nodes.min()), float(y_nodes.max())),
        cmap="magma",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|u_z^{PINN} - u_z^{FEM}|$")

    # Overlay load patch.
    patch = Rectangle(
        (pc.LOAD_PATCH_X[0], pc.LOAD_PATCH_Y[0]),
        pc.LOAD_PATCH_X[1] - pc.LOAD_PATCH_X[0],
        pc.LOAD_PATCH_Y[1] - pc.LOAD_PATCH_Y[0],
        fill=False,
        edgecolor="white",
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )
    ax.add_patch(patch)

    ax.set_title(f"Top-Surface Error Heatmap ({case.name})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_paths = save_figure(fig, "fig_error_heatmap")
    plt.close(fig)

    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
