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
    if args.t1 is None or args.t2 is None or args.t3 is None:
        t1 = t2 = t3 = H / 3.0
    else:
        t1, t2, t3 = float(args.t1), float(args.t2), float(args.t3)

    cfg = {
        "geometry": {"Lx": float(args.Lx), "Ly": float(args.Ly), "H": H},
        "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
        "layers": [
            {"t": t1, "E": float(args.E1), "nu": float(args.nu)},
            {"t": t2, "E": float(args.E2), "nu": float(args.nu)},
            {"t": t3, "E": float(args.E3), "nu": float(args.nu)},
        ],
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
            "layers": [
                {"t": float(t1), "E": float(args.E1), "nu": float(args.nu)},
                {"t": float(t2), "E": float(args.E2), "nu": float(args.nu)},
                {"t": float(t3), "E": float(args.E3), "nu": float(args.nu)},
            ],
            "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
            "load_patch": cfg["load_patch"],
            "use_soft_mask": bool(int(args.use_soft_mask)),
        },
    }
    np.save(args.out, out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

