from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
import physics


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    weights = []
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if k.startswith("layers.0.net.") and k.endswith(".weight"):
            weights.append((k, v))
    if not weights:
        raise ValueError("Could not infer architecture from checkpoint keys.")

    # neurons from first linear layer
    first = sd["layers.0.net.0.weight"]
    neurons = int(first.shape[0])

    # hidden layer count from number of Linear layers minus output layer
    linear_indices = sorted(
        {
            int(k.split(".")[3])
            for k, _ in weights
            if k.split(".")[3].isdigit()
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
    # Avoid MPS by default (it is not available on all macOS versions).
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: PINN vs FEM (top + cross-section)."
    )
    parser.add_argument("--fea", default="fea_solution.npy", help="Path to fea_solution.npy")
    parser.add_argument("--model", default="pinn_model.pth", help="Path to trained PINN checkpoint")
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (default: auto, cpu-preferred)")
    parser.add_argument(
        "--out_prefix",
        default="comparison_latest",
        help="Output prefix for images (e.g., comparison_latest)",
    )
    parser.add_argument(
        "--hard_clamp_sides",
        type=int,
        default=None,
        help="If set, overrides pinn_config.HARD_CLAMP_SIDES for evaluation (0/1).",
    )

    # Model architecture (optional; inferred if omitted)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--neurons", type=int, default=None)

    # 3-layer params (defaults match fixed-param training used for FEM parity)
    h = float(getattr(config, "H", 0.1))
    parser.add_argument("--E1", type=float, default=1.0)
    parser.add_argument("--E2", type=float, default=1.0)
    parser.add_argument("--E3", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=h / 3.0)
    parser.add_argument("--t2", type=float, default=h / 3.0)
    parser.add_argument("--t3", type=float, default=h / 3.0)
    parser.add_argument("--r", type=float, default=float(getattr(config, "RESTITUTION_REF", 0.5)))
    parser.add_argument("--mu", type=float, default=float(getattr(config, "FRICTION_REF", 0.3)))
    parser.add_argument("--v0", type=float, default=float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)))
    args = parser.parse_args()

    fea_path = args.fea
    if not os.path.exists(fea_path):
        alt = os.path.join(PINN_WORKFLOW_DIR, os.path.basename(fea_path))
        if os.path.exists(alt):
            fea_path = alt
    if not os.path.exists(fea_path):
        raise FileNotFoundError(f"FEA file not found: {args.fea}")

    model_path = args.model
    if not os.path.exists(model_path):
        alt = os.path.join(PINN_WORKFLOW_DIR, os.path.basename(model_path))
        if os.path.exists(alt):
            model_path = alt
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    device = _select_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading FEA: {fea_path}")
    data = np.load(fea_path, allow_pickle=True).item()
    X_fea = data["x"]
    Y_fea = data["y"]
    Z_fea = data["z"]
    U_fea = data["u"]
    print(f"FEA grid: {X_fea.shape} (u: {U_fea.shape})")

    print(f"Loading PINN: {model_path}")
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(model_path, map_location=device)

    if args.layers is None or args.neurons is None:
        inferred_layers, inferred_neurons = _infer_arch_from_state_dict(sd)
        layers = inferred_layers if args.layers is None else args.layers
        neurons = inferred_neurons if args.neurons is None else args.neurons
    else:
        layers = args.layers
        neurons = args.neurons

    config.LAYERS = int(layers)
    config.NEURONS = int(neurons)
    if args.hard_clamp_sides is not None:
        config.HARD_CLAMP_SIDES = bool(int(args.hard_clamp_sides))

    pinn = model.MultiLayerPINN().to(device)
    pinn.load_state_dict(sd, strict=True)
    pinn.eval()
    print(f"Model loaded (layers={config.LAYERS}, neurons={config.NEURONS})")

    pts = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1).astype(np.float32)
    params = np.array(
        [args.E1, args.t1, args.E2, args.t2, args.E3, args.t3, args.r, args.mu, args.v0],
        dtype=np.float32,
    )[None, :]
    params = np.repeat(params, repeats=pts.shape[0], axis=0)
    pts12 = np.concatenate([pts, params], axis=1)

    with torch.no_grad():
        pts_tensor = torch.tensor(pts12, dtype=torch.float32, device=device)
        v_flat = pinn(pts_tensor)
        u_flat = physics.decode_u(v_flat, pts_tensor).cpu().numpy()
    U_pinn = u_flat.reshape(U_fea.shape)
    print("Computed PINN predictions on FEM grid.")

    u_z_fea_top = U_fea[:, :, -1, 2]
    u_z_pinn_top = U_pinn[:, :, -1, 2]
    abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)

    # --- Full-field metric (matches training/verify_fea_match style) ---
    err_all = (U_pinn - U_fea).reshape(-1, 3)
    mae_all = float(np.mean(np.abs(err_all)))
    rmse_all = float(np.sqrt(np.mean(err_all**2)))
    max_abs_all = float(np.max(np.abs(err_all)))
    denom_all = float(np.max(np.abs(U_fea)))
    mae_all_pct = (mae_all / denom_all) * 100.0 if denom_all > 0 else 0.0

    # --- Top-surface u_z metric (for visualization parity with older plots) ---
    mae_top = float(np.mean(abs_diff))
    max_err_top = float(np.max(abs_diff))
    denom_top = float(np.max(np.abs(u_z_fea_top)))
    mae_top_pct = (mae_top / denom_top) * 100.0 if denom_top > 0 else 0.0

    print("\n==================================================")
    print("Comparison Results (Full Field u):")
    print("==================================================")
    print(f"MAE: {mae_all:.6f}")
    print(f"RMSE: {rmse_all:.6f}")
    print(f"Max |Error|: {max_abs_all:.6f}")
    print(f"MAE % of max |FEA u|: {mae_all_pct:.2f}%")
    print("--------------------------------------------------")
    print("Top Surface u_z (for reference):")
    print(f"MAE: {mae_top:.6f}")
    print(f"MAE % of max |FEA u_z|: {mae_top_pct:.2f}%")
    print(f"Max Error: {max_err_top:.6f}")
    print(f"Peak Deflection FEA: {u_z_fea_top.min():.6f}")
    print(f"Peak Deflection PINN: {u_z_pinn_top.min():.6f}")
    print("==================================================")

    vmin = float(min(u_z_fea_top.min(), u_z_pinn_top.min()))
    vmax = float(max(u_z_fea_top.max(), u_z_pinn_top.max()))
    err_max = float(abs_diff.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c1 = axes[0].contourf(X_fea[:, :, 0], Y_fea[:, :, 0], u_z_fea_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title("FEA Displacement u_z (Top)", fontsize=14)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X_fea[:, :, 0], Y_fea[:, :, 0], u_z_pinn_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title("PINN Displacement u_z (Top)", fontsize=14)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(c2, ax=axes[1])

    c3 = axes[2].contourf(X_fea[:, :, 0], Y_fea[:, :, 0], abs_diff, levels=50, cmap="magma", vmin=0.0, vmax=err_max)
    axes[2].set_title("Absolute Error |FEA - PINN|", fontsize=14)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    out_top = f"{args.out_prefix}_top.png"
    plt.savefig(out_top, dpi=150)
    print(f"Saved {out_top}")

    ny = Y_fea.shape[1]
    mid_y_idx = ny // 2
    X_slice = X_fea[:, mid_y_idx, :]
    Z_slice = Z_fea[:, mid_y_idx, :]
    u_z_fea_slice = U_fea[:, mid_y_idx, :, 2]
    u_z_pinn_slice = U_pinn[:, mid_y_idx, :, 2]
    abs_diff_slice = np.abs(u_z_fea_slice - u_z_pinn_slice)

    vmin_s = float(min(u_z_fea_slice.min(), u_z_pinn_slice.min()))
    vmax_s = float(max(u_z_fea_slice.max(), u_z_pinn_slice.max()))
    err_max_s = float(abs_diff_slice.max())

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    c1 = axes2[0].contourf(X_slice, Z_slice, u_z_fea_slice, levels=50, cmap="jet", vmin=vmin_s, vmax=vmax_s)
    axes2[0].set_title(f"FEA u_z (Slice @ Y={Y_fea[0, mid_y_idx, 0]:.2f})", fontsize=14)
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("z")
    plt.colorbar(c1, ax=axes2[0])

    c2 = axes2[1].contourf(X_slice, Z_slice, u_z_pinn_slice, levels=50, cmap="jet", vmin=vmin_s, vmax=vmax_s)
    axes2[1].set_title("PINN u_z (Slice)", fontsize=14)
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("z")
    plt.colorbar(c2, ax=axes2[1])

    c3 = axes2[2].contourf(X_slice, Z_slice, abs_diff_slice, levels=50, cmap="magma", vmin=0.0, vmax=err_max_s)
    axes2[2].set_title("Absolute Error |FEA - PINN|", fontsize=14)
    axes2[2].set_xlabel("x")
    axes2[2].set_ylabel("z")
    plt.colorbar(c3, ax=axes2[2])

    plt.tight_layout()
    out_xz = f"{args.out_prefix}_xz.png"
    plt.savefig(out_xz, dpi=150)
    print(f"Saved {out_xz}")


if __name__ == "__main__":
    main()
