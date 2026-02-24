import os
import sys
import argparse

import numpy as np
import torch

# Avoid .pyc writes in locked-down environments.
sys.dont_write_bytecode = True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import data
import model
import physics


def _set_cfg(**kwargs) -> dict:
    prev = {}
    for k, v in kwargs.items():
        prev[k] = getattr(config, k, None)
        setattr(config, k, v)
    return prev


def _restore_cfg(prev: dict) -> None:
    for k, v in prev.items():
        setattr(config, k, v)


def _assert_finite(name: str, arr) -> None:
    if isinstance(arr, torch.Tensor):
        ok = torch.isfinite(arr).all().item()
    else:
        ok = np.isfinite(np.asarray(arr)).all()
    if not ok:
        raise AssertionError(f"{name} has non-finite values")


def _smoke_case(label: str, *, stl_path: str, sampler: str, normalize: bool, load_dir: str, normal_filter: bool) -> None:
    prev = _set_cfg(
        GEOMETRY_MODE="cad",
        CAD_STL_PATH=stl_path,
        CAD_SAMPLER=sampler,
        CAD_NORMALIZE_TO_CONFIG_BOUNDS=normalize,
        CAD_LOAD_DIRECTION=load_dir,
        CAD_BC_NORMAL_FILTER=normal_filter,
        CAD_BC_NORMAL_COS_MIN=0.5,
        CAD_BOTTOM_CLAMPED=True,
        # Keep it fast
        N_INTERIOR=400,
        N_SIDES=200,
        N_TOP_LOAD=200,
        N_TOP_FREE=200,
        N_BOTTOM=200,
    )
    try:
        d = data.get_data()

        pts = d["interior"][0]
        assert isinstance(pts, torch.Tensor)
        assert pts.ndim == 2 and pts.shape[1] == 12, ("interior", tuple(pts.shape))
        _assert_finite("interior", pts)

        for k in ["bottom_clamp", "top_load", "top_free", "side_free"]:
            pts = d[k]
            assert isinstance(pts, torch.Tensor)
            assert pts.ndim == 2 and pts.shape[1] == 12, (k, tuple(pts.shape))
            _assert_finite(k, pts)

        assert "interfaces" in d and isinstance(d["interfaces"], list) and len(d["interfaces"]) == 2
        for i, pts in enumerate(d["interfaces"]):
            assert isinstance(pts, torch.Tensor)
            assert pts.ndim == 2 and pts.shape[1] == 12, (f"interfaces[{i}]", tuple(pts.shape))
            _assert_finite(f"interfaces[{i}]", pts)

        assert "top_load_normal" in d and "top_free_normal" in d and "side_free_normal" in d, "boundary normals missing"
        if sampler == "tessellation":
            assert "domain_volume" in d and float(d["domain_volume"]) > 0.0, "domain_volume missing/invalid"
            assert "top_load_area" in d and float(d["top_load_area"]) >= 0.0, "top_load_area missing/invalid"
            _assert_finite("top_load_normal", d["top_load_normal"])
            _assert_finite("top_free_normal", d["top_free_normal"])
            _assert_finite("side_free_normal", d["side_free_normal"])

        # Physics smoke: forward/backward through the loss (catches shape/BC/area-volume wiring issues).
        device = torch.device("cpu")
        pinn = model.MultiLayerPINN().to(device)
        loss, losses = physics.compute_loss(pinn, d, device, weights=dict(config.WEIGHTS))
        if not torch.isfinite(loss).item():
            raise AssertionError(f"total loss is non-finite: {loss.item()}")
        for name, val in losses.items():
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                if not torch.isfinite(val).item():
                    raise AssertionError(f"loss component {name} is non-finite: {val.item()}")

        print(f"OK: {label}")
    finally:
        _restore_cfg(prev)


def main() -> None:
    stl_dir = os.path.join(PINN_WORKFLOW_DIR, "stl")
    plate = os.path.join(stl_dir, "unit_plate_1x1x0p1.stl")
    sphere = os.path.join(stl_dir, "sphere.stl")

    parser = argparse.ArgumentParser(description="Fast smoke checks for CAD sampling + CAD physics wiring.")
    parser.add_argument(
        "--stl",
        action="append",
        default=None,
        help="STL path to test. May be repeated. Default: built-in plate + sphere under `pinn-workflow/stl/`.",
    )
    args = parser.parse_args()

    if args.stl:
        stls = [os.path.abspath(p) for p in args.stl]
    else:
        stls = [plate, sphere]

    for p in stls:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing STL: {p}")

    for stl_path in stls:
        name = os.path.basename(stl_path)
        # AABB sampler (no normals) — just ensure CAD mode is stable.
        _smoke_case(f"{name}/aabb/normalize", stl_path=stl_path, sampler="aabb", normalize=True, load_dir="normal", normal_filter=False)

        # Tessellation sampler — exercise CAD BC classification + area/volume wiring.
        _smoke_case(f"{name}/tess/normalize/normal", stl_path=stl_path, sampler="tessellation", normalize=True, load_dir="normal", normal_filter=True)
        _smoke_case(f"{name}/tess/normalize/global_z", stl_path=stl_path, sampler="tessellation", normalize=True, load_dir="global_z", normal_filter=True)

        # Non-normalized CAD (not “FEA-like” by default, but should not crash).
        _smoke_case(f"{name}/tess/no-normalize/normal", stl_path=stl_path, sampler="tessellation", normalize=False, load_dir="normal", normal_filter=True)

    print("All CAD smoke checks passed.")


if __name__ == "__main__":
    main()
