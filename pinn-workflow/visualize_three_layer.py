import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model


def _layer_id_grid(z: np.ndarray, t1: float, t2: float) -> np.ndarray:
    z1 = float(t1)
    z2 = float(t1 + t2)
    out = np.full_like(z, 2, dtype=np.int32)
    out[z < z1] = 0
    out[(z >= z1) & (z < z2)] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize 3-layer laminate PiNN cross-section (x-z at fixed y).")
    ap.add_argument("--model", default="pinn_model.pth", help="Path to trained PINN checkpoint (state_dict).")
    ap.add_argument("--out", default="three_layer_xz.png", help="Output PNG path.")
    ap.add_argument("--show", action="store_true", help="Show interactive window instead of saving.")
    ap.add_argument("--layers", type=int, default=None, help="Hidden layer count used by the checkpoint (auto if omitted).")
    ap.add_argument("--neurons", type=int, default=None, help="Hidden width used by the checkpoint (auto if omitted).")

    ap.add_argument("--E1", type=float, default=10.0)
    ap.add_argument("--E2", type=float, default=1.0)
    ap.add_argument("--E3", type=float, default=5.0)
    ap.add_argument("--t1", type=float, default=0.03)
    ap.add_argument("--t2", type=float, default=0.04)
    ap.add_argument("--t3", type=float, default=0.03)
    ap.add_argument("--r", type=float, default=float(getattr(config, "RESTITUTION_REF", 0.5)))
    ap.add_argument("--mu", type=float, default=float(getattr(config, "FRICTION_REF", 0.3)))
    ap.add_argument("--v0", type=float, default=float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)))

    ap.add_argument("--y", type=float, default=0.5, help="Fixed y for the x-z slice.")
    ap.add_argument("--nx", type=int, default=151)
    ap.add_argument("--nz", type=int, default=121)
    args = ap.parse_args()

    def _infer_arch(sd: dict) -> tuple[int, int]:
        w0 = sd.get("layers.0.net.0.weight", None)
        if w0 is None or not hasattr(w0, "shape") or len(w0.shape) != 2:
            raise ValueError("Cannot infer architecture from checkpoint (missing layers.0.net.0.weight).")
        neurons = int(w0.shape[0])
        linear_keys = [k for k in sd.keys() if k.startswith("layers.0.net.") and k.endswith(".weight")]
        # Count Linear layers in the Sequential. hidden_layers = (#linear_layers - 1).
        hidden_layers = max(1, len(linear_keys) - 1)
        return hidden_layers, neurons

    # Ensure the instantiated model matches the checkpoint architecture.
    if args.model and os.path.exists(args.model):
        sd_peek = torch.load(args.model, map_location="cpu", weights_only=False)
        inferred_layers, inferred_neurons = _infer_arch(sd_peek)
    else:
        inferred_layers, inferred_neurons = int(getattr(config, "LAYERS", 4)), int(getattr(config, "NEURONS", 64))

    config.LAYERS = int(args.layers) if args.layers is not None else inferred_layers
    config.NEURONS = int(args.neurons) if args.neurons is not None else inferred_neurons

    t1 = max(args.t1, 1e-4)
    t2 = max(args.t2, 1e-4)
    t3 = max(args.t3, 1e-4)
    T = t1 + t2 + t3

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(args.model):
        sd = torch.load(args.model, map_location=device, weights_only=False)
        pinn.load_state_dict(sd, strict=False)
        print(f"Loaded model: {args.model}")
    else:
        print(f"Warning: model not found at {args.model}; using random weights.")
    pinn.eval()

    x = np.linspace(0.0, float(config.Lx), int(args.nx))
    z = np.linspace(0.0, float(T), int(args.nz))
    X, Z = np.meshgrid(x, z, indexing="xy")  # (nz, nx)
    Y = np.full_like(X, float(args.y))

    pts = np.zeros((X.size, 12), dtype=np.float32)
    pts[:, 0] = X.reshape(-1)
    pts[:, 1] = Y.reshape(-1)
    pts[:, 2] = Z.reshape(-1)
    pts[:, 3] = float(args.E1)
    pts[:, 4] = float(t1)
    pts[:, 5] = float(args.E2)
    pts[:, 6] = float(t2)
    pts[:, 7] = float(args.E3)
    pts[:, 8] = float(t3)
    pts[:, 9] = float(args.r)
    pts[:, 10] = float(args.mu)
    pts[:, 11] = float(args.v0)

    with torch.no_grad():
        v = pinn(torch.tensor(pts, device=device)).cpu().numpy()

    # Visualization uses the same decoding convention as train-time metrics (applied pointwise with local E).
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    t_scale = 1.0 if alpha == 0.0 else (float(getattr(config, "H", T)) / float(T)) ** alpha

    # local E depends on layer (piecewise by z)
    layer_id = _layer_id_grid(Z, t1, t2)
    E_local = np.where(layer_id == 0, float(args.E1), np.where(layer_id == 1, float(args.E2), float(args.E3))).astype(np.float32)

    uz = v[:, 2].reshape(Z.shape)
    uz_u = (uz / (E_local**e_pow)) * t_scale

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        uz_u,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    axes[0].axhline(t1, color="w", linewidth=1)
    axes[0].axhline(t1 + t2, color="w", linewidth=1)
    axes[0].set_title("Predicted $u_z$ (x-z slice) with layer interfaces")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        layer_id,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    axes[1].axhline(t1, color="w", linewidth=1)
    axes[1].axhline(t1 + t2, color="w", linewidth=1)
    axes[1].set_title("Layer ID map (0/1/2)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    fig.colorbar(im1, ax=axes[1], ticks=[0, 1, 2])

    if args.show:
        plt.show()
    else:
        fig.savefig(args.out, dpi=180)
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
