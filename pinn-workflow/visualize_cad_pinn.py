import os
import sys

import numpy as np
import torch

# Matplotlib needs writable cache + temp dirs in this environment.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_tmp = os.path.join(REPO_ROOT, ".tmp")
_mpl_cache = os.path.join(REPO_ROOT, ".cache", "matplotlib")
os.makedirs(_local_tmp, exist_ok=True)
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("TMPDIR", _local_tmp)
os.environ.setdefault("TEMP", _local_tmp)
os.environ.setdefault("TMP", _local_tmp)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib.pyplot as plt

PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
import data
from tessellated_geometry import load_stl_surface, affine_map_surface_to_bounds, sample_boundary


def _load_model(model_path: str, device: torch.device) -> model.MultiLayerPINN:
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device, weights_only=True)
        target_sd = pinn.state_dict()
        w_key = "layer.net.0.weight"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0]:
                if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:5] = src_w[:, 0:5]
                    adapted[:, 8:11] = src_w[:, 5:8]
                    sd[w_key] = adapted
                elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:7] = src_w[:, 0:7]
                    adapted[:, 8:11] = src_w[:, 7:10]
                    sd[w_key] = adapted
        pinn.load_state_dict(sd, strict=False)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    pinn.eval()
    return pinn


def _u_from_v(v: np.ndarray, E_val: float, thickness: float) -> np.ndarray:
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    t_scale = 1.0 if alpha == 0.0 else (float(config.H) / max(1e-8, float(thickness))) ** alpha
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    return (v / (float(E_val) ** e_pow)) * t_scale


def main():
    device = torch.device("cpu")

    # Configure CAD mode on the included simple STL plate
    config.GEOMETRY_MODE = "cad"
    config.CAD_SAMPLER = "tessellation"
    if config.CAD_STL_PATH is None:
        config.CAD_STL_PATH = os.path.join(PINN_WORKFLOW_DIR, "stl", "unit_plate_1x1x0p1.stl")
    config.CAD_NORMALIZE_TO_CONFIG_BOUNDS = True

    out_dir = os.path.join(REPO_ROOT, "impact_pipeline_outputs", "cad_viz")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_candidates = [
        os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth"),
        os.path.join(REPO_ROOT, "pinn_model.pth"),
    ]
    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError("Could not find `pinn_model.pth` in repo root or `pinn-workflow/`.")

    pinn = _load_model(ckpt, device)

    # Use CAD-aware training-data sampler to get top/bottom/side point sets
    d = data.get_data()

    # For visualization, sample additional boundary points directly on the STL surface.
    surface = load_stl_surface(config.CAD_STL_PATH)
    surface = affine_map_surface_to_bounds(surface, (0.0, 0.0, 0.0), (float(config.Lx), float(config.Ly), float(config.H)))
    bnd = sample_boundary(surface, nr_points=20000)
    xyz = np.concatenate([bnd["x"], bnd["y"], bnd["z"]], axis=1)

    # Use reference parameters for a single forward pass
    E_val = float(getattr(config, "E_vals", [1.0])[0])
    thickness = float(getattr(config, "H", 0.1))
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    params = np.stack(
        [
            np.full(len(xyz), E_val),
            np.full(len(xyz), thickness),
            np.full(len(xyz), r_ref),
            np.full(len(xyz), mu_ref),
            np.full(len(xyz), v0_ref),
        ],
        axis=1,
    )
    pts = np.concatenate([xyz, params], axis=1).astype(np.float32)

    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
    u = _u_from_v(v, E_val, thickness)

    uz = u[:, 2]
    uz_min, uz_max = float(np.min(uz)), float(np.max(uz))
    print(f"CAD surface Uz range: [{uz_min:.6f}, {uz_max:.6f}]")

    # 3D scatter on CAD surface, colored by Uz
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=uz, cmap="viridis", s=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title("CAD STL surface sampled points colored by PiNN Uz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.15))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_surface_uz_scatter.png"), dpi=200)
    plt.close(fig)

    # Deformed visualization (scaled for visibility)
    scale = 2.0
    xyz_def = xyz.copy()
    xyz_def[:, 2] = xyz_def[:, 2] + scale * uz
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz_def[:, 0], xyz_def[:, 1], xyz_def[:, 2], c=uz, cmap="viridis", s=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title(f"CAD surface (deformed) colored by PiNN Uz (scale={scale})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.25))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_surface_uz_deformed_scatter.png"), dpi=200)
    plt.close(fig)

    # Top surface (load vs free) quick sanity plot
    top_pts = torch.cat([d["top_load"], d["top_free"]], dim=0).cpu().numpy()
    top_xyz = top_pts[:, 0:3]
    with torch.no_grad():
        v_top = pinn(torch.tensor(top_pts, dtype=torch.float32).to(device)).cpu().numpy()
    u_top = _u_from_v(v_top, E_val, thickness)
    uz_top = u_top[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(top_xyz[:, 0], top_xyz[:, 1], top_xyz[:, 2], c=uz_top, cmap="viridis", s=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title("Top surface points (load + free) colored by PiNN Uz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.15))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_top_surface_uz.png"), dpi=200)
    plt.close(fig)

    print(f"Wrote CAD visualization outputs to: {out_dir}")


if __name__ == "__main__":
    main()
