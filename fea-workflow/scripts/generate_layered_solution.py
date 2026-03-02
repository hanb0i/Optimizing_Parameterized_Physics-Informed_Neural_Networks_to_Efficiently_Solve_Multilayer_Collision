import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import fem_solver


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a layered FEA solution (x,y,z,u) npy for PINN parity.")
    ap.add_argument("--out", default="fea_solution_layered.npy", help="Output .npy path")

    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--H", type=float, default=0.1)

    ap.add_argument("--num_layers", type=int, default=2, help="Number of layers (2 or 3 supported by this script).")
    ap.add_argument("--E1", type=float, default=1.0)
    ap.add_argument("--E2", type=float, default=1.0)
    ap.add_argument("--E3", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--t2", type=float, default=None)
    ap.add_argument("--t3", type=float, default=None)

    ap.add_argument("--p0", type=float, default=1.0)
    ap.add_argument("--use_soft_mask", type=int, default=1)
    ap.add_argument("--x0", type=float, default=1.0 / 3.0)
    ap.add_argument("--x1", type=float, default=2.0 / 3.0)
    ap.add_argument("--y0", type=float, default=1.0 / 3.0)
    ap.add_argument("--y1", type=float, default=2.0 / 3.0)

    ap.add_argument("--ne_x", type=int, default=30)
    ap.add_argument("--ne_y", type=int, default=30)
    ap.add_argument("--ne_z", type=int, default=30)
    args = ap.parse_args()

    H = float(args.H)
    L = int(args.num_layers)
    if L not in (2, 3):
        raise ValueError(f"--num_layers must be 2 or 3, got {L}")

    E_vals = [float(args.E1), float(args.E2), float(args.E3)][:L]
    t_in = [args.t1, args.t2, args.t3][:L]
    if any(v is None for v in t_in):
        t_vals = [H / float(L)] * L
    else:
        t_vals = [float(v) for v in t_in]
    # Normalize to match H exactly.
    tsum = sum(t_vals)
    if tsum <= 0:
        raise ValueError("Sum of thicknesses must be > 0.")
    t_vals = [tv * (H / tsum) for tv in t_vals]

    cfg = {
        "geometry": {"Lx": float(args.Lx), "Ly": float(args.Ly), "H": H},
        "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
        "layers": [{"t": float(ti), "E": float(Ei), "nu": float(args.nu)} for Ei, ti in zip(E_vals, t_vals)],
        "load_patch": {
            "pressure": float(args.p0),
            "x_start": float(args.x0),
            "x_end": float(args.x1),
            "y_start": float(args.y0),
            "y_end": float(args.y1),
        },
        "use_soft_mask": bool(int(args.use_soft_mask)),
    }

    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
    X, Y, Z = np.meshgrid(np.array(x_nodes), np.array(y_nodes), np.array(z_nodes), indexing="ij")
    out = {
        "x": X,
        "y": Y,
        "z": Z,
        "u": np.array(u_grid),
        "meta": {
            "Lx": float(args.Lx),
            "Ly": float(args.Ly),
            "H": H,
            "layers": [{"t": float(ti), "E": float(Ei), "nu": float(args.nu)} for Ei, ti in zip(E_vals, t_vals)],
            "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
            "load_patch": cfg["load_patch"],
            "use_soft_mask": bool(int(args.use_soft_mask)),
        },
    }
    np.save(args.out, out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
