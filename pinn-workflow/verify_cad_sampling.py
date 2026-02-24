import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import data


def main():
    config.GEOMETRY_MODE = "cad"
    if config.CAD_STL_PATH is None:
        config.CAD_STL_PATH = os.path.join(
            PINN_WORKFLOW_DIR, "stl", "unit_plate_1x1x0p1.stl"
        )
    # Try tessellation sampler by default for the CAD demo
    config.CAD_SAMPLER = "tessellation"

    d = data.get_data()
    print(f"CAD STL: {config.CAD_STL_PATH}")
    print(f"CAD_SAMPLER: {getattr(config, 'CAD_SAMPLER', None)}")
    print(
        "CAD BC normal filter:",
        getattr(config, "CAD_BC_NORMAL_FILTER", None),
        "cos_min=",
        getattr(config, "CAD_BC_NORMAL_COS_MIN", None),
    )
    if "domain_volume" in d:
        print(f"domain_volume (est): {float(d['domain_volume']):.6g}")
    if "top_load_area" in d:
        print(f"top_load_area (est): {float(d['top_load_area']):.6g}")
    for k in ["interior"]:
        pts = d[k][0]
        xyz = pts[:, 0:3]
        xyz_min = tuple(torch.min(xyz, dim=0).values.tolist())
        xyz_max = tuple(torch.max(xyz, dim=0).values.tolist())
        print(f"{k:>8}: {tuple(pts.shape)} xyz_min={xyz_min} xyz_max={xyz_max}")
    for k in ["bottom_clamp", "top_load", "top_free", "side_free"]:
        pts = d[k]
        xyz = pts[:, 0:3]
        z_min = float(torch.min(xyz[:, 2]))
        z_max = float(torch.max(xyz[:, 2]))
        print(f"{k:>8}: {tuple(pts.shape)} z_range=[{z_min:.6f}, {z_max:.6f}]")


if __name__ == "__main__":
    main()
