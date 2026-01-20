import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config  # noqa: F401
import model


def main():
    print("Loading FEA Solution...")
    fea_path = os.path.join(PINN_WORKFLOW_DIR, "fea_solution.npy")
    if not os.path.exists(fea_path):
        # Fallback to local
        fea_path = "fea_solution.npy"
    print(f"Loading from: {fea_path}")
    data = np.load(fea_path, allow_pickle=True).item()
    X_fea = data["x"]
    Y_fea = data["y"]
    Z_fea = data["z"]
    U_fea = data["u"]

    print(f"FEA Grid: {X_fea.shape}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    if not os.path.exists(model_path):
        model_path = "pinn_model.pth"
    print(f"Loading PINN from: {model_path}")
    pinn.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    pinn.eval()
    print("PINN model loaded")

    pts = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1)
    with torch.no_grad():
        pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
        U_pinn_flat = pinn(pts_tensor, 0).cpu().numpy()

    U_pinn = U_pinn_flat.reshape(U_fea.shape)
    print("PINN predictions computed")

    u_z_fea_top = U_fea[:, :, -1, 2]
    u_z_pinn_top = U_pinn[:, :, -1, 2]

    abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    denom = np.max(np.abs(u_z_fea_top))
    mae_pct = (mae / denom) * 100.0 if denom > 0 else 0.0

    print(f"\n{'=' * 50}")
    print("Comparison Results (Top Surface u_z):")
    print(f"{'=' * 50}")
    print(f"MAE: {mae:.6f}")
    print(f"MAE % of max |FEA u_z|: {mae_pct:.2f}%")
    print(f"Max Error: {max_err:.6f}")
    print(f"Peak Deflection FEA: {u_z_fea_top.min():.6f}")
    print(f"Peak Deflection PINN: {u_z_pinn_top.min():.6f}")
    print(f"{'=' * 50}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c1 = axes[0].contourf(
        X_fea[:, :, 0], Y_fea[:, :, 0], u_z_fea_top, levels=50, cmap="jet"
    )
    axes[0].set_title("FEA Displacement u_z (Top)", fontsize=14)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(
        X_fea[:, :, 0], Y_fea[:, :, 0], u_z_pinn_top, levels=50, cmap="jet"
    )
    axes[1].set_title("PINN Displacement u_z (Top)", fontsize=14)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(c2, ax=axes[1])

    c3 = axes[2].contourf(
        X_fea[:, :, 0], Y_fea[:, :, 0], abs_diff, levels=50, cmap="magma"
    )
    axes[2].set_title("Absolute Error |FEA - PINN|", fontsize=14)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("comparison_top.png", dpi=150)
    print("Saved comparison_top.png")
    # plt.show() # Commented out show to prevent blocking in automation

    # --- Cross-Section Plot (XZ plane at mid Y) ---
    ny = Y_fea.shape[1]
    mid_y_idx = ny // 2
    
    # Extract slices
    X_slice = X_fea[:, mid_y_idx, :]
    Z_slice = Z_fea[:, mid_y_idx, :]
    u_z_fea_slice = U_fea[:, mid_y_idx, :, 2]
    u_z_pinn_slice = U_pinn[:, mid_y_idx, :, 2]
    abs_diff_slice = np.abs(u_z_fea_slice - u_z_pinn_slice)
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. FEA
    c1 = axes2[0].contourf(
        X_slice, Z_slice, u_z_fea_slice, levels=50, cmap="jet"
    )
    axes2[0].set_title(f"FEA u_z (Slice @ Y={Y_fea[0, mid_y_idx, 0]:.2f})", fontsize=14)
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("z")
    plt.colorbar(c1, ax=axes2[0])
    
    # 2. PINN
    c2 = axes2[1].contourf(
        X_slice, Z_slice, u_z_pinn_slice, levels=50, cmap="jet"
    )
    axes2[1].set_title("PINN u_z (Slice)", fontsize=14)
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("z")
    plt.colorbar(c2, ax=axes2[1])
    
    # 3. Error
    c3 = axes2[2].contourf(
        X_slice, Z_slice, abs_diff_slice, levels=50, cmap="magma"
    )
    axes2[2].set_title("Absolute Error |FEA - PINN|", fontsize=14)
    axes2[2].set_xlabel("x")
    axes2[2].set_ylabel("z")
    plt.colorbar(c3, ax=axes2[2])
    
    plt.tight_layout()
    plt.savefig("comparison_cross_section.png", dpi=150)
    print("Saved comparison_cross_section.png")


if __name__ == "__main__":
    main()
