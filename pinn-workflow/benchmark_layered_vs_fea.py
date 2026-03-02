from __future__ import annotations

import argparse
import os
import sys

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
import physics
import fem_solver


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    w0 = sd.get("layers.0.net.0.weight", None)
    if w0 is None:
        return int(getattr(config, "LAYERS", 4)), int(getattr(config, "NEURONS", 64))
    neurons = int(w0.shape[0])
    linear_indices = set()
    for k in sd.keys():
        if not (isinstance(k, str) and k.startswith("layers.0.net.") and k.endswith(".weight")):
            continue
        try:
            idx = int(k.split(".")[3])
        except Exception:
            continue
        linear_indices.add(idx)
    hidden_layers = max(1, len(linear_indices) - 1) if linear_indices else int(getattr(config, "LAYERS", 4))
    return hidden_layers, neurons


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(den, eps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark random 2-layer PINN cases vs on-the-fly layered FEA.")
    ap.add_argument("--model", default="pinn_model.pth", help="PINN checkpoint path.")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (auto if omitted).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", type=int, default=10, help="Number of random FEA cases.")
    ap.add_argument("--err_thresh_pct", type=float, default=5.0, help="Pass threshold as percent.")

    ap.add_argument("--ne_x", type=int, default=16)
    ap.add_argument("--ne_y", type=int, default=16)
    ap.add_argument("--ne_z", type=int, default=16)
    ap.add_argument("--nu", type=float, default=float(getattr(config, "NU_FIXED", 0.3)))
    ap.add_argument("--p0", type=float, default=float(getattr(config, "p0", 1.0)))
    ap.add_argument("--use_soft_mask", type=int, default=int(getattr(config, "USE_SOFT_LOAD_MASK", True)))
    args = ap.parse_args()

    if int(getattr(config, "NUM_LAYERS", 2)) != 2:
        raise ValueError(f"This benchmark script currently supports NUM_LAYERS=2 only (got {getattr(config, 'NUM_LAYERS', None)}).")

    rng = np.random.default_rng(int(args.seed))
    device = _select_device(args.device)

    pinn = model.MultiLayerPINN().to(device)
    if args.model and os.path.exists(args.model):
        sd = torch.load(args.model, map_location=device, weights_only=False)
        inferred_layers, inferred_neurons = _infer_arch_from_state_dict(sd)
        config.LAYERS = int(inferred_layers)
        config.NEURONS = int(inferred_neurons)
        pinn = model.MultiLayerPINN().to(device)
        pinn.load_state_dict(sd, strict=False)
        print(f"Loaded model: {args.model}")
    else:
        print(f"Warning: model not found at {args.model!r}; benchmarking with randomly initialized weights.")
    pinn.eval()

    e_min, e_max = map(float, getattr(config, "E_RANGE", (1.0, 10.0)))
    t_min, t_max = map(float, getattr(config, "THICKNESS_RANGE", (float(getattr(config, "H", 0.1)), float(getattr(config, "H", 0.1)))))
    frac_min = float(getattr(config, "LAYER_THICKNESS_FRACTION_MIN", 0.05))
    frac_min = max(1e-4, min(frac_min, 0.49))

    x0, x1 = map(float, getattr(config, "LOAD_PATCH_X", (1.0 / 3.0, 2.0 / 3.0)))
    y0, y1 = map(float, getattr(config, "LOAD_PATCH_Y", (1.0 / 3.0, 2.0 / 3.0)))

    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    thresh = float(args.err_thresh_pct) / 100.0
    pass_peak = 0
    pass_l2 = 0

    for k in range(int(args.cases)):
        H = float(rng.uniform(t_min, t_max))
        f = float(rng.uniform(frac_min, 1.0 - frac_min))
        t1 = H * f
        t2 = H - t1
        E1 = float(rng.uniform(e_min, e_max))
        E2 = float(rng.uniform(e_min, e_max))

        cfg = {
            "geometry": {"Lx": float(getattr(config, "Lx", 1.0)), "Ly": float(getattr(config, "Ly", 1.0)), "H": H},
            "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
            "layers": [
                {"t": float(t1), "E": float(E1), "nu": float(args.nu)},
                {"t": float(t2), "E": float(E2), "nu": float(args.nu)},
            ],
            "load_patch": {
                "pressure": float(args.p0),
                "x_start": x0 / float(getattr(config, "Lx", 1.0)),
                "x_end": x1 / float(getattr(config, "Lx", 1.0)),
                "y_start": y0 / float(getattr(config, "Ly", 1.0)),
                "y_end": y1 / float(getattr(config, "Ly", 1.0)),
            },
            "use_soft_mask": bool(int(args.use_soft_mask)),
        }

        x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_fem(cfg)
        X, Y, Z = np.meshgrid(np.array(x_nodes), np.array(y_nodes), np.array(z_nodes), indexing="ij")
        u_true = np.array(u_fea, dtype=np.float32)

        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32, copy=False)
        params = np.array([E1, t1, E2, t2, r_ref, mu_ref, v0_ref], dtype=np.float32)[None, :]
        x_in = np.concatenate([pts, np.repeat(params, pts.shape[0], axis=0)], axis=1)

        with torch.no_grad():
            x_t = torch.tensor(x_in, dtype=torch.float32, device=device)
            v = pinn(x_t)
            u_pred = physics.decode_u(v, x_t).cpu().numpy().astype(np.float32).reshape(u_true.shape)

        # Top-surface error metrics (u_z).
        top = np.isclose(Z, H)
        uz_fea_top = u_true[:, :, -1, 2]
        uz_pinn_top = u_pred[:, :, -1, 2]
        peak_fea = float(np.min(uz_fea_top))
        peak_pinn = float(np.min(uz_pinn_top))
        peak_rel = abs(peak_pinn - peak_fea) / max(abs(peak_fea), 1e-12)
        l2_rel = _relative_l2(uz_pinn_top, uz_fea_top)

        pass_peak += int(peak_rel <= thresh)
        pass_l2 += int(l2_rel <= thresh)
        print(
            f"case{k}: H={H:.4f} t1/T={f:.2f} E1={E1:.2f} E2={E2:.2f} | peak_rel={peak_rel*100:.2f}% l2_rel={l2_rel*100:.2f}%"
        )

    n = max(1, int(args.cases))
    print(f"\nPeak pass-rate: {pass_peak}/{n} ({(pass_peak/n)*100:.1f}%) at <= {args.err_thresh_pct:.1f}%")
    print(f"L2 pass-rate:   {pass_l2}/{n} ({(pass_l2/n)*100:.1f}%) at <= {args.err_thresh_pct:.1f}%")


if __name__ == "__main__":
    main()
