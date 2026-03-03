
from __future__ import annotations

import torch
import numpy as np
import os
import pinn_config as config
from cad_geometry import stl_bounds, sample_uniform_box, sample_uniform_rect_on_plane
from tessellated_geometry import (
    load_stl_surface,
    affine_map_surface_to_bounds,
    sample_boundary as tess_sample_boundary,
    sample_interior as tess_sample_interior,
)

# -------------------------------
# Layered laminate PiNN inputs
# -------------------------------
#
# Input layout (for NUM_LAYERS = L):
#   [x, y, z, E1, t1, ..., EL, tL, restitution, friction, impact_velocity]
#
# Total thickness is T = sum(t_i), and interfaces are at cumulative thicknesses:
#   z = t1, z = t1+t2, ..., z = t1+...+t_{L-1}.

def _num_layers() -> int:
    n = int(getattr(config, "NUM_LAYERS", 2))
    if n < 1:
        raise ValueError(f"config.NUM_LAYERS must be >= 1, got {n}")
    return n

def _param_dim() -> int:
    return 2 * _num_layers() + 3

def _din() -> int:
    return 3 + _param_dim()

_DIN = _din()


def _uniform(n: int, lo: float, hi: float, *, device=None) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (hi - lo) + lo

def _load_mask_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Soft quadratic load mask on the top patch, matching `physics.load_mask`/FEA.
    Returns (N,1) tensor in [0,1].
    """
    x_min, x_max = map(float, config.LOAD_PATCH_X)
    y_min, y_max = map(float, config.LOAD_PATCH_Y)
    in_patch = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if not bool(getattr(config, "USE_SOFT_LOAD_MASK", True)):
        return in_patch.to(dtype=torch.float32)
    dx = float(x_max - x_min) if float(x_max - x_min) != 0.0 else 1.0
    dy = float(y_max - y_min) if float(y_max - y_min) != 0.0 else 1.0
    x_norm = (x - x_min) / dx
    y_norm = (y - y_min) / dy
    soft = 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)
    soft = torch.clamp(soft, min=0.0)
    return soft.to(dtype=torch.float32) * in_patch.to(dtype=torch.float32)

def _sample_xy_on_patch_biased(n: int, *, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x,y) on the patch, optionally biased by mask^p (renormalized) to emphasize center.
    """
    x_min, x_max = map(float, config.LOAD_PATCH_X)
    y_min, y_max = map(float, config.LOAD_PATCH_Y)
    power = float(getattr(config, "LOAD_MASK_SAMPLING_POWER", 0.0))
    frac_biased = float(getattr(config, "LOAD_MASK_SAMPLING_BIASED_FRACTION", 1.0))
    frac_biased = max(0.0, min(frac_biased, 1.0))
    if (not bool(getattr(config, "USE_SOFT_LOAD_MASK", True))) or power <= 0.0 or frac_biased <= 0.0:
        return _uniform(n, x_min, x_max, device=device), _uniform(n, y_min, y_max, device=device)

    n_b = int(round(n * frac_biased))
    n_u = max(0, int(n - n_b))

    # Oversample candidates then draw with probability ∝ mask^power.
    m = max(n_b, int(6 * n_b))
    xc = _uniform(m, x_min, x_max, device=device)
    yc = _uniform(m, y_min, y_max, device=device)
    w = _load_mask_xy(xc, yc).clamp_min(0.0) ** power
    if float(w.sum()) <= 0.0 or torch.isnan(w).any():
        idx = torch.randint(0, m, (n_b,), device=xc.device)
    else:
        p = (w[:, 0] / w.sum()).to(dtype=torch.float32)
        idx = torch.multinomial(p, num_samples=n_b, replacement=True)
    xb = xc[idx]
    yb = yc[idx]
    if n_u == 0:
        return xb, yb
    xu = _uniform(n_u, x_min, x_max, device=device)
    yu = _uniform(n_u, y_min, y_max, device=device)
    return torch.cat([xu, xb], dim=0), torch.cat([yu, yb], dim=0)

def _sample_xy_top_free_ring(n: int, *, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points on the top surface just outside the load patch boundary (a thin ring).
    """
    x0, x1 = map(float, config.LOAD_PATCH_X)
    y0, y1 = map(float, config.LOAD_PATCH_Y)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    width_frac = float(getattr(config, "TOP_FREE_RING_WIDTH_FRAC", 0.08))
    w = width_frac * max(dx, dy)
    w = max(w, 1e-6)

    # Candidate region: expanded patch bounding box.
    xmin = max(0.0, x0 - w)
    xmax = min(float(config.Lx), x1 + w)
    ymin = max(0.0, y0 - w)
    ymax = min(float(config.Ly), y1 + w)

    out_x, out_y = [], []
    need = int(n)
    for _ in range(20):
        if need <= 0:
            break
        m = max(2000, int(6 * need))
        xc = _uniform(m, xmin, xmax, device=device)
        yc = _uniform(m, ymin, ymax, device=device)
        in_patch = ((xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1))[:, 0]
        # outside patch but inside expanded region (i.e. within ~w of patch boundary)
        sel = ~in_patch
        xs = xc[sel]
        ys = yc[sel]
        if xs.numel() == 0:
            continue
        take = min(need, xs.shape[0])
        out_x.append(xs[:take])
        out_y.append(ys[:take])
        need -= take
    if need > 0:
        # fallback: uniform outside patch
        m = max(2 * need, 2000)
        xc = _uniform(m, 0.0, float(config.Lx), device=device)
        yc = _uniform(m, 0.0, float(config.Ly), device=device)
        in_patch = ((xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1))[:, 0]
        xs = xc[~in_patch][:need]
        ys = yc[~in_patch][:need]
        out_x.append(xs)
        out_y.append(ys)
    return torch.cat(out_x, dim=0), torch.cat(out_y, dim=0)


def _sample_layer_params(n: int, *, total_thickness: float | None = None, device=None) -> torch.Tensor:
    """
    Returns params tensor shaped (n, 2*L+3):
      [E1,t1,...,EL,tL,r,mu,v0]
    """
    L = _num_layers()
    if bool(getattr(config, "TRAIN_FIXED_PARAMS", False)):
        E = float(getattr(config, "TRAIN_FIXED_E", 1.0))
        T = float(getattr(config, "TRAIN_FIXED_TOTAL_THICKNESS", getattr(config, "H", 0.1)))
        if total_thickness is not None:
            T = float(total_thickness)

        # Per-layer E_i / t_i (fallbacks: TRAIN_FIXED_E and equal thickness splits).
        E_vals = []
        for i in range(L):
            e_cfg = getattr(config, f"TRAIN_FIXED_E{i+1}", None)
            E_vals.append(float(E if e_cfg is None else e_cfg))

        t_cfgs = [getattr(config, f"TRAIN_FIXED_T{i+1}", None) for i in range(L)]
        if all(v is None for v in t_cfgs):
            t_vals = [float(T) / float(L)] * L
        else:
            t_vals = [float((T / L) if v is None else v) for v in t_cfgs]
            tsum = max(1e-8, float(sum(t_vals)))
            scale = float(T) / tsum
            t_vals = [tv * scale for tv in t_vals]

        t_tensors = [torch.full((n, 1), tv, dtype=torch.float32, device=device).clamp_min(1e-4) for tv in t_vals]
        e_tensors = [torch.full((n, 1), ev, dtype=torch.float32, device=device) for ev in E_vals]
        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        r = torch.full((n, 1), r_ref, dtype=torch.float32, device=device)
        mu = torch.full((n, 1), mu_ref, dtype=torch.float32, device=device)
        v0 = torch.full((n, 1), v0_ref, dtype=torch.float32, device=device)
        parts = []
        for e_i, t_i in zip(e_tensors, t_tensors):
            parts.extend([e_i, t_i])
        parts.extend([r, mu, v0])
        return torch.cat(parts, dim=1)

    e_min, e_max = _get_e_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()

    # Total thickness, then split by fractions that sum to 1.
    if total_thickness is None:
        t_min, t_max = _get_thickness_range()
        T = _uniform(n, t_min, t_max, device=device)
    else:
        T = torch.full((n, 1), float(total_thickness), dtype=torch.float32, device=device)

    frac_min = float(getattr(config, "LAYER_THICKNESS_FRACTION_MIN", 0.05))
    frac_min = max(0.0, min(frac_min, 1.0 / float(L) - 1e-6))
    raw = torch.rand(n, L, device=device)
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    frac = frac_min + (1.0 - float(L) * frac_min) * raw

    t_list = [(T * frac[:, i : i + 1]).clamp_min(1e-4) for i in range(L)]
    tsum = sum(t_list).clamp_min(1e-12)
    scale = T / tsum
    t_list = [ti * scale for ti in t_list]

    E_list = [_uniform(n, e_min, e_max, device=device) for _ in range(L)]
    r = _uniform(n, r_min, r_max, device=device)
    mu = _uniform(n, mu_min, mu_max, device=device)
    v0 = _uniform(n, v0_min, v0_max, device=device)

    parts = []
    for e_i, t_i in zip(E_list, t_list):
        parts.extend([e_i, t_i])
    parts.extend([r, mu, v0])
    return torch.cat(parts, dim=1)


def _assemble_input(xyz: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shaped (n,3), got {tuple(xyz.shape)}")
    if params.ndim != 2 or params.shape[1] != _param_dim():
        raise ValueError(f"Expected params shaped (n,{_param_dim()}), got {tuple(params.shape)}")
    x = torch.cat([xyz, params], dim=1)
    if x.shape[1] != _DIN:
        raise ValueError(f"Internal error: expected input dim {_DIN}, got {x.shape[1]}")
    return x


def _total_thickness_from_params(params: torch.Tensor) -> torch.Tensor:
    L = _num_layers()
    t_cols = params[:, 1 : 2 * L : 2]
    return t_cols.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _interfaces_from_params(params: torch.Tensor) -> list[torch.Tensor]:
    L = _num_layers()
    if L <= 1:
        return []
    t_cols = params[:, 1 : 2 * L : 2].clamp_min(0.0)  # (N, L)
    cum = torch.cumsum(t_cols, dim=1)
    return [cum[:, i : i + 1] for i in range(L - 1)]

# Parameter range helper (keeps baseline configs intact)
def _get_e_range():
    if hasattr(config, "E_RANGE"):
        e_min, e_max = config.E_RANGE
    else:
        e_vals = getattr(config, "E_vals", [1.0])
        e_min, e_max = min(e_vals), max(e_vals)
        if e_min == e_max:
            e_max = e_min + 1.0
    return float(e_min), float(e_max)

def _get_thickness_range():
    if hasattr(config, "THICKNESS_RANGE"):
        t_min, t_max = config.THICKNESS_RANGE
    else:
        t_min, t_max = float(getattr(config, "H", 0.1)), float(getattr(config, "H", 0.1))
        if t_min == t_max:
            t_max = t_min + 0.01
    return float(t_min), float(t_max)

def _get_restitution_range():
    if hasattr(config, "RESTITUTION_RANGE"):
        r_min, r_max = config.RESTITUTION_RANGE
    else:
        r_min, r_max = 0.5, 0.5
    return float(r_min), float(r_max)

def _get_friction_range():
    if hasattr(config, "FRICTION_RANGE"):
        mu_min, mu_max = config.FRICTION_RANGE
    else:
        mu_min, mu_max = 0.3, 0.3
    return float(mu_min), float(mu_max)

def _get_impact_velocity_range():
    if hasattr(config, "IMPACT_VELOCITY_RANGE"):
        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
    else:
        v0_min, v0_max = 1.0, 1.0
    return float(v0_min), float(v0_max)

# Import FEM solver for generating supervision data
import sys
FEA_SOLVER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fea-workflow", "solver")
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)


def load_fem_supervision_data(n_points_per_e=None, e_values=None, thickness_values=None):
    import fem_solver
    
    if e_values is None:
        if hasattr(config, "DATA_E_VALUES"):
            e_values = config.DATA_E_VALUES
        else:
            e_min, e_max = _get_e_range()
            e_values = [e_min, 0.5 * (e_min + e_max), e_max]
    if thickness_values is None:
        if hasattr(config, "DATA_THICKNESS_VALUES"):
            thickness_values = config.DATA_THICKNESS_VALUES
        else:
            t_min, t_max = _get_thickness_range()
            thickness_values = [t_min, 0.5 * (t_min + t_max), t_max]
    
    L = _num_layers()
    layer_fractions = getattr(config, "DATA_LAYER_FRACTIONS", None)
    if layer_fractions is None:
        # For 2 layers this is t1/T. For L!=2, it is ignored and we use equal splits.
        layer_fractions = [0.5]
    layered_random_cases = int(getattr(config, "DATA_LAYERED_RANDOM_CASES", 0) or 0)
    mesh_cfg = None
    if hasattr(config, "DATA_FEA_NE_X") and hasattr(config, "DATA_FEA_NE_Y") and hasattr(config, "DATA_FEA_NE_Z"):
        mesh_cfg = {
            "ne_x": int(getattr(config, "DATA_FEA_NE_X")),
            "ne_y": int(getattr(config, "DATA_FEA_NE_Y")),
            "ne_z": int(getattr(config, "DATA_FEA_NE_Z")),
        }

    # Count cases to allocate points per FEA solve.
    if L <= 1:
        n_cases = len(e_values) * len(thickness_values)
    elif L == 2:
        if layered_random_cases > 0:
            n_cases = layered_random_cases
        else:
            n_cases = len(thickness_values) * len(layer_fractions) * (len(e_values) ** 2)
    else:
        # Avoid combinatorial explosion; use equal thickness splits and same E-values per layer.
        n_cases = len(e_values) * len(thickness_values)

    if n_points_per_e is None:
        if hasattr(config, "N_DATA_POINTS"):
            n_points_per_e = int(config.N_DATA_POINTS) // max(1, int(n_cases))
        else:
            n_points_per_e = 0
    
    x_data_list = []
    u_data_list = []
    
    def _solve_and_sample(cfg, layer_E, layer_t, *, label: str):
        print(f"  Generating FEM supervision for {label}...")
        x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
        X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        u_flat = np.asarray(u_grid, dtype=float).reshape(-1, 3)
        total_points = len(x_flat)
        if total_points == 0 or int(n_points_per_e) <= 0:
            return
        indices = np.random.choice(total_points, size=min(int(n_points_per_e), total_points), replace=False)

        # FEA supervision is quasi-static and does not depend on restitution/friction/impact_velocity.
        # To make the surrogate robust across the full parameter ranges, randomize these "impact"
        # parameters during supervision so the network learns invariance.
        r_min, r_max = _get_restitution_range()
        mu_min, mu_max = _get_friction_range()
        v0_min, v0_max = _get_impact_velocity_range()
        restitution = (np.random.rand(len(indices)).astype(np.float32) * (r_max - r_min) + r_min).astype(np.float32)
        friction = (np.random.rand(len(indices)).astype(np.float32) * (mu_max - mu_min) + mu_min).astype(np.float32)
        impact_velocity = (np.random.rand(len(indices)).astype(np.float32) * (v0_max - v0_min) + v0_min).astype(np.float32)

        cols = [
            x_flat[indices].astype(np.float32),
            y_flat[indices].astype(np.float32),
            z_flat[indices].astype(np.float32),
        ]
        for Ei, ti in zip(layer_E, layer_t):
            cols.append(np.ones(len(indices), dtype=np.float32) * float(Ei))
            cols.append(np.ones(len(indices), dtype=np.float32) * float(ti))
        cols.extend([restitution, friction, impact_velocity])
        x_sampled = np.stack(cols, axis=1)
        u_sampled = u_flat[indices].astype(np.float32, copy=False)
        x_data_list.append(torch.tensor(x_sampled, dtype=torch.float32))
        u_data_list.append(torch.tensor(u_sampled, dtype=torch.float32))

    if L == 2 and layered_random_cases > 0:
        e_min, e_max = _get_e_range()
        for k in range(layered_random_cases):
            thickness = float(np.random.choice(thickness_values))
            f = float(np.random.choice(layer_fractions)) if layer_fractions else float(np.random.rand())
            f = max(1e-4, min(1.0 - 1e-4, f))
            E1 = float(np.random.rand() * (e_max - e_min) + e_min)
            E2 = float(np.random.rand() * (e_max - e_min) + e_min)
            t1 = thickness * f
            t2 = thickness - t1
            cfg = {
                "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness},
                **({"mesh": dict(mesh_cfg)} if mesh_cfg is not None else {}),
                "layers": [
                    {"t": float(t1), "E": float(E1), "nu": float(getattr(config, "NU_FIXED", config.nu_vals[0]))},
                    {"t": float(t2), "E": float(E2), "nu": float(getattr(config, "NU_FIXED", config.nu_vals[0]))},
                ],
                "load_patch": {
                    "pressure": config.p0,
                    "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                    "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                    "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                    "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
                },
            }
            _solve_and_sample(cfg, [E1, E2], [t1, t2], label=f"rand{k}: E1={E1:.3f},E2={E2:.3f},H={thickness:.3f},t1/T={f:.2f}")
    else:
        for thickness in thickness_values:
            thickness = float(thickness)
            if L <= 1:
                for E_val in e_values:
                    E_val = float(E_val)
                    cfg = {
                        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness},
                        **({"mesh": dict(mesh_cfg)} if mesh_cfg is not None else {}),
                        "material": {"E": E_val, "nu": config.nu_vals[0]},
                        "load_patch": {
                            "pressure": config.p0,
                            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
                        },
                    }
                    _solve_and_sample(cfg, [E_val], [thickness], label=f"E={E_val}, H={thickness}")
            elif L == 2:
                for frac in layer_fractions:
                    f = float(frac)
                    f = max(1e-4, min(1.0 - 1e-4, f))
                    t1 = thickness * f
                    t2 = thickness - t1
                    for E1 in e_values:
                        for E2 in e_values:
                            E1 = float(E1)
                            E2 = float(E2)
                            cfg = {
                                "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness},
                                **({"mesh": dict(mesh_cfg)} if mesh_cfg is not None else {}),
                                "layers": [
                                    {"t": float(t1), "E": float(E1), "nu": float(getattr(config, "NU_FIXED", config.nu_vals[0]))},
                                    {"t": float(t2), "E": float(E2), "nu": float(getattr(config, "NU_FIXED", config.nu_vals[0]))},
                                ],
                                "load_patch": {
                                    "pressure": config.p0,
                                    "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                                    "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                                    "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                                    "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
                                },
                            }
                            _solve_and_sample(cfg, [E1, E2], [t1, t2], label=f"E1={E1},E2={E2},H={thickness},t1/T={f:.2f}")
            else:
                for E_val in e_values:
                    E_val = float(E_val)
                    t_list = [thickness / float(L)] * L
                    cfg = {
                        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness},
                        **({"mesh": dict(mesh_cfg)} if mesh_cfg is not None else {}),
                        "layers": [{"t": float(ti), "E": float(E_val), "nu": float(getattr(config, "NU_FIXED", config.nu_vals[0]))} for ti in t_list],
                        "load_patch": {
                            "pressure": config.p0,
                            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
                            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
                            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
                            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
                        },
                    }
                    _solve_and_sample(cfg, [E_val] * L, t_list, label=f"E={E_val}, H={thickness}, L={L}")
    
    if not x_data_list:
        x_data = torch.zeros((0, _din()), dtype=torch.float32)
        u_data = torch.zeros((0, 3), dtype=torch.float32)
    else:
        x_data = torch.cat(x_data_list, dim=0)
        u_data = torch.cat(u_data_list, dim=0)
    
    print(f"  Loaded {len(x_data)} sparse FEM supervision points")
    return x_data, u_data


def sample_domain(n, z_min, z_max):
    # Uniform sampling
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    t_min, t_max = _get_thickness_range()
    t = torch.rand(n, 1) * (t_max - t_min) + t_min
    z = torch.rand(n, 1) * t
    
    # Sample Young's Modulus E
    e_min, e_max = _get_e_range()
    e = torch.rand(n, 1) * (e_max - e_min) + e_min

    r_min, r_max = _get_restitution_range()
    restitution = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    friction = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    impact_velocity = torch.rand(n, 1) * (v0_max - v0_min) + v0_min

    return torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1)

def sample_domain_under_patch(n, z_min, z_max):
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    x = torch.rand(n, 1) * (x_max - x_min) + x_min
    y = torch.rand(n, 1) * (y_max - y_min) + y_min
    t_min, t_max = _get_thickness_range()
    t = torch.rand(n, 1) * (t_max - t_min) + t_min
    z = torch.rand(n, 1) * t
    e_min, e_max = _get_e_range()
    e = torch.rand(n, 1) * (e_max - e_min) + e_min
    r_min, r_max = _get_restitution_range()
    restitution = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    friction = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    impact_velocity = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1)

def sample_domain_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_domain(n, z_min, z_max)
    
    # Normalize residuals to probabilities
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10  # Add small epsilon for numerical stability
    residual_probs = residual_probs / residual_probs.sum()  # Renormalize
    
    # Sample indices based on residual weights
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    # Add noise to create new points nearby
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    e_min, e_max = _get_e_range()
    t_min, t_max = _get_thickness_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    noise_z = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t_max - t_min)
    noise_e = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_t = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t_max - t_min)
    noise_r = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (r_max - r_min)
    noise_mu = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    noise_v0 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (v0_max - v0_min)

    noise = torch.cat([noise_x, noise_y, noise_z, noise_e, noise_t, noise_r, noise_mu, noise_v0], dim=1)
    
    new_pts = sampled_pts + noise
    
    # Clamp to domain bounds
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t_min, t_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], r_min, r_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], mu_min, mu_max)
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], v0_min, v0_max)
    new_pts[:, 2] = torch.maximum(new_pts[:, 2], torch.zeros_like(new_pts[:, 2]))
    new_pts[:, 2] = torch.minimum(new_pts[:, 2], new_pts[:, 4])
    
    return new_pts

def sample_boundaries(n, z_min, z_max):
    # 4 Side faces: x=0, x=Lx, y=0, y=Ly
    # Split n among 4 faces
    n_face = n // 4
    
    e_min, e_max = _get_e_range()
    t_min, t_max = _get_thickness_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    # x=0
    y1 = torch.rand(n_face, 1) * config.Ly
    t1 = torch.rand(n_face, 1) * (t_max - t_min) + t_min
    z1 = torch.rand(n_face, 1) * t1
    x1 = torch.zeros(n_face, 1)
    e1 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    r1 = torch.rand(n_face, 1) * (r_max - r_min) + r_min
    mu1 = torch.rand(n_face, 1) * (mu_max - mu_min) + mu_min
    v01 = torch.rand(n_face, 1) * (v0_max - v0_min) + v0_min
    p1 = torch.cat([x1, y1, z1, e1, t1, r1, mu1, v01], dim=1)
    
    # x=Lx
    y2 = torch.rand(n_face, 1) * config.Ly
    t2 = torch.rand(n_face, 1) * (t_max - t_min) + t_min
    z2 = torch.rand(n_face, 1) * t2
    x2 = torch.ones(n_face, 1) * config.Lx
    e2 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    r2 = torch.rand(n_face, 1) * (r_max - r_min) + r_min
    mu2 = torch.rand(n_face, 1) * (mu_max - mu_min) + mu_min
    v02 = torch.rand(n_face, 1) * (v0_max - v0_min) + v0_min
    p2 = torch.cat([x2, y2, z2, e2, t2, r2, mu2, v02], dim=1)
    
    # y=0
    x3 = torch.rand(n_face, 1) * config.Lx
    t3 = torch.rand(n_face, 1) * (t_max - t_min) + t_min
    z3 = torch.rand(n_face, 1) * t3
    y3 = torch.zeros(n_face, 1)
    e3 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    r3 = torch.rand(n_face, 1) * (r_max - r_min) + r_min
    mu3 = torch.rand(n_face, 1) * (mu_max - mu_min) + mu_min
    v03 = torch.rand(n_face, 1) * (v0_max - v0_min) + v0_min
    p3 = torch.cat([x3, y3, z3, e3, t3, r3, mu3, v03], dim=1)
    
    # y=Ly
    x4 = torch.rand(n_face, 1) * config.Lx
    t4 = torch.rand(n_face, 1) * (t_max - t_min) + t_min
    z4 = torch.rand(n_face, 1) * t4
    y4 = torch.ones(n_face, 1) * config.Ly
    e4 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    r4 = torch.rand(n_face, 1) * (r_max - r_min) + r_min
    mu4 = torch.rand(n_face, 1) * (mu_max - mu_min) + mu_min
    v04 = torch.rand(n_face, 1) * (v0_max - v0_min) + v0_min
    p4 = torch.cat([x4, y4, z4, e4, t4, r4, mu4, v04], dim=1)
    
    return torch.cat([p1, p2, p3, p4], dim=0)

def sample_boundaries_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_boundaries(n, z_min, z_max)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    # Keep boundary constraints while perturbing
    new_pts = sampled_pts.clone()
    
    # Add noise to E for all points
    e_min, e_max = _get_e_range()
    t_min, t_max = _get_thickness_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    noise_e = (torch.rand(n) - 0.5) * 2 * noise_scale * (e_max - e_min)
    new_pts[:, 3] += noise_e
    noise_t = (torch.rand(n) - 0.5) * 2 * noise_scale * (t_max - t_min)
    new_pts[:, 4] += noise_t
    noise_r = (torch.rand(n) - 0.5) * 2 * noise_scale * (r_max - r_min)
    new_pts[:, 5] += noise_r
    noise_mu = (torch.rand(n) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    new_pts[:, 6] += noise_mu
    noise_v0 = (torch.rand(n) - 0.5) * 2 * noise_scale * (v0_max - v0_min)
    new_pts[:, 7] += noise_v0
    
    # For each face, perturb only the non-fixed coordinates
    for i in range(n):
        pt = new_pts[i]
        t_val = torch.clamp(pt[4], t_min, t_max)
        if torch.abs(pt[0]) < 1e-6:  # x=0 face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * t_val
            new_pts[i, 0] = 0.0
        elif torch.abs(pt[0] - config.Lx) < 1e-6:  # x=Lx face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * t_val
            new_pts[i, 0] = config.Lx
        elif torch.abs(pt[1]) < 1e-6:  # y=0 face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * t_val
            new_pts[i, 1] = 0.0
        elif torch.abs(pt[1] - config.Ly) < 1e-6:  # y=Ly face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * t_val
            new_pts[i, 1] = config.Ly
    
    # Clamp
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t_min, t_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], r_min, r_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], mu_min, mu_max)
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], v0_min, v0_max)
    new_pts[:, 2] = torch.maximum(new_pts[:, 2], torch.zeros_like(new_pts[:, 2]))
    new_pts[:, 2] = torch.minimum(new_pts[:, 2], new_pts[:, 4])
    
    return new_pts

def sample_top_load(n):
    # Loaded Patch: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    xl = torch.rand(n, 1) * (config.Lx/3) + config.Lx/3
    yl = torch.rand(n, 1) * (config.Ly/3) + config.Ly/3
    t_min, t_max = _get_thickness_range()
    tl = torch.rand(n, 1) * (t_max - t_min) + t_min
    zl = tl
    e_min, e_max = _get_e_range()
    el = torch.rand(n, 1) * (e_max - e_min) + e_min
    r_min, r_max = _get_restitution_range()
    rl = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    mul = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    v0l = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return torch.cat([xl, yl, zl, el, tl, rl, mul, v0l], dim=1)

def sample_top_free(n):
    # Rejection sampling for points outside patch
    pts_free_list = []
    count = 0
    while count < n:
        batch = 1000
        x = torch.rand(batch, 1) * config.Lx
        y = torch.rand(batch, 1) * config.Ly
        
        in_patch = (x > config.Lx/3) & (x < 2*config.Lx/3) & \
                   (y > config.Ly/3) & (y < 2*config.Ly/3)
        
        mask_free = ~in_patch.squeeze()
        xf, yf = x[mask_free], y[mask_free]
        if len(xf) > 0:
            t_min, t_max = _get_thickness_range()
            tf = torch.rand(len(xf), 1) * (t_max - t_min) + t_min
            zf = tf
            e_min, e_max = _get_e_range()
            ef = torch.rand(len(xf), 1) * (e_max - e_min) + e_min
            r_min, r_max = _get_restitution_range()
            rf = torch.rand(len(xf), 1) * (r_max - r_min) + r_min
            mu_min, mu_max = _get_friction_range()
            muf = torch.rand(len(xf), 1) * (mu_max - mu_min) + mu_min
            v0_min, v0_max = _get_impact_velocity_range()
            v0f = torch.rand(len(xf), 1) * (v0_max - v0_min) + v0_min
            batch_pts = torch.cat([xf, yf, zf, ef, tf, rf, muf, v0f], dim=1)
            pts_free_list.append(batch_pts)
            count += len(xf)
    
    pts_free = torch.cat(pts_free_list, dim=0)[:n]
    return pts_free

def sample_surface_residual_based(n, z_val, prev_pts, prev_residuals, constrain_load_patch=False, is_load_patch=False):
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        if constrain_load_patch and is_load_patch:
            return sample_top_load(n)
        elif constrain_load_patch and not is_load_patch:
            return sample_top_free(n)
        else:
            # General surface sampling
            x = torch.rand(n, 1) * config.Lx
            y = torch.rand(n, 1) * config.Ly
            t_min, t_max = _get_thickness_range()
            t = torch.rand(n, 1) * (t_max - t_min) + t_min
            if z_val == 0.0:
                z = torch.zeros(n, 1)
            else:
                z = t
            e_min, e_max = _get_e_range()
            e = torch.rand(n, 1) * (e_max - e_min) + e_min
            r_min, r_max = _get_restitution_range()
            r = torch.rand(n, 1) * (r_max - r_min) + r_min
            mu_min, mu_max = _get_friction_range()
            mu = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
            v0_min, v0_max = _get_impact_velocity_range()
            v0 = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
            return torch.cat([x, y, z, e, t, r, mu, v0], dim=1)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    e_min, e_max = _get_e_range()
    t_min, t_max = _get_thickness_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    noise_e = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_t = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t_max - t_min)
    noise_r = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (r_max - r_min)
    noise_mu = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    noise_v0 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (v0_max - v0_min)
    noise = torch.cat([noise_x, noise_y, torch.zeros(n, 1), noise_e, noise_t, noise_r, noise_mu, noise_v0], dim=1)
    
    new_pts = sampled_pts + noise
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t_min, t_max)
    if z_val == 0.0:
        new_pts[:, 2] = 0.0
    else:
        new_pts[:, 2] = new_pts[:, 4]  # Fix z to top surface
    
    # Clamp to domain
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], r_min, r_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], mu_min, mu_max)
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], v0_min, v0_max)
    
    # If constrained to load patch or free region
    if constrain_load_patch:
        if is_load_patch:
            # Clamp to load patch
            new_pts[:, 0] = torch.clamp(new_pts[:, 0], config.Lx/3, 2*config.Lx/3)
            new_pts[:, 1] = torch.clamp(new_pts[:, 1], config.Ly/3, 2*config.Ly/3)
        else:
            # Keep outside load patch - if inside, push to nearest edge
            for i in range(n):
                x, y = new_pts[i, 0].item(), new_pts[i, 1].item()
                if config.Lx/3 < x < 2*config.Lx/3 and config.Ly/3 < y < 2*config.Ly/3:
                    # Inside patch, push out to nearest boundary
                    dx_low = x - config.Lx/3
                    dx_high = 2*config.Lx/3 - x
                    dy_low = y - config.Ly/3
                    dy_high = 2*config.Ly/3 - y
                    min_dist = min(dx_low, dx_high, dy_low, dy_high)
                    if min_dist == dx_low:
                        new_pts[i, 0] = config.Lx/3 - 0.01
                    elif min_dist == dx_high:
                        new_pts[i, 0] = 2*config.Lx/3 + 0.01
                    elif min_dist == dy_low:
                        new_pts[i, 1] = config.Ly/3 - 0.01
                    else:
                        new_pts[i, 1] = 2*config.Ly/3 + 0.01
                    # Re-clamp
                    new_pts[i, 0] = torch.clamp(new_pts[i, 0], torch.tensor(0.0), torch.tensor(config.Lx))
                    new_pts[i, 1] = torch.clamp(new_pts[i, 1], torch.tensor(0.0), torch.tensor(config.Ly))
    
    return new_pts

def sample_top(n):
    # DEPRECATED: Use sample_top_load and sample_top_free separately
    n_load = n // 2
    n_free = n - n_load
    
    pts_load = sample_top_load(n_load)
    pts_free = sample_top_free(n_free)
    
    return pts_load, pts_free

def sample_interface(n, z_val):
    # z = z_val
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    t_min, t_max = _get_thickness_range()
    t = torch.rand(n, 1) * (t_max - t_min) + t_min
    z = torch.minimum(torch.ones(n, 1) * z_val, t)
    e = torch.rand(n, 1) * (config.E_RANGE[1] - config.E_RANGE[0]) + config.E_RANGE[0]
    r_min, r_max = _get_restitution_range()
    r = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    mu = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    v0 = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return torch.cat([x, y, z, e, t, r, mu, v0], dim=1)

def sample_bottom(n):
    x_bot = torch.rand(n, 1) * config.Lx
    y_bot = torch.rand(n, 1) * config.Ly
    z_bot = torch.zeros(n, 1)
    e_min, e_max = _get_e_range()
    e_bot = torch.rand(n, 1) * (e_max - e_min) + e_min
    t_min, t_max = _get_thickness_range()
    t_bot = torch.rand(n, 1) * (t_max - t_min) + t_min
    r_min, r_max = _get_restitution_range()
    r_bot = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    mu_bot = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    v0_bot = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return torch.cat([x_bot, y_bot, z_bot, e_bot, t_bot, r_bot, mu_bot, v0_bot], dim=1)

def get_data(prev_data=None, residuals=None):
    if getattr(config, "GEOMETRY_MODE", "box").lower() == "cad":
        return get_data_cad(prev_data=prev_data, residuals=residuals)

    def _weighted_choice(weights: torch.Tensor, k: int) -> torch.Tensor:
        w = torch.clamp(weights, min=0.0)
        if w.numel() == 0:
            return torch.zeros((0,), dtype=torch.long)
        if float(w.detach().sum()) <= 0.0 or torch.isnan(w).any():
            return torch.randint(0, w.numel(), (k,), dtype=torch.long)
        probs = (w / w.sum()).to(dtype=torch.float32)
        return torch.multinomial(probs, num_samples=k, replacement=True)

    def _perturb_xyz(x: torch.Tensor, *, t_total: torch.Tensor) -> torch.Tensor:
        noise = float(getattr(config, "SAMPLING_NOISE_SCALE", 0.05))
        out = x.clone()
        out[:, 0:1] = out[:, 0:1] + (torch.rand_like(out[:, 0:1]) - 0.5) * 2.0 * noise * float(config.Lx)
        out[:, 1:2] = out[:, 1:2] + (torch.rand_like(out[:, 1:2]) - 0.5) * 2.0 * noise * float(config.Ly)
        out[:, 2:3] = out[:, 2:3] + (torch.rand_like(out[:, 2:3]) - 0.5) * 2.0 * noise * t_total
        out[:, 0:1] = torch.clamp(out[:, 0:1], 0.0, float(config.Lx))
        out[:, 1:2] = torch.clamp(out[:, 1:2], 0.0, float(config.Ly))
        out[:, 2:3] = torch.clamp(out[:, 2:3], min=0.0)
        out[:, 2:3] = torch.minimum(out[:, 2:3], t_total)
        return out

    def _resample_region(prev_pts: torch.Tensor, prev_res: torch.Tensor, n: int, *, region: str) -> torch.Tensor:
        if prev_pts is None or prev_pts.shape[0] == 0 or n == 0:
            return prev_pts[:0] if prev_pts is not None else torch.zeros((0, _DIN), dtype=torch.float32)
        power = float(getattr(config, "RESAMPLE_RESIDUAL_POWER", 1.0))
        w = torch.clamp(prev_res.to(prev_pts.device), min=0.0) ** power
        idx = _weighted_choice(w, n)
        pts = prev_pts[idx].clone()
        params = pts[:, 3:]
        t_total = _total_thickness_from_params(params)
        pts = _perturb_xyz(pts, t_total=t_total)

        if region == "bottom":
            pts[:, 2:3] = 0.0
        elif region in {"top_load", "top_free"}:
            pts[:, 2:3] = t_total
        elif region == "sides":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            d_x0 = torch.abs(x - 0.0)
            d_x1 = torch.abs(x - float(config.Lx))
            d_y0 = torch.abs(y - 0.0)
            d_y1 = torch.abs(y - float(config.Ly))
            d = torch.cat([d_x0, d_x1, d_y0, d_y1], dim=1)
            which = torch.argmin(d, dim=1)
            pts[which == 0, 0] = 0.0
            pts[which == 1, 0] = float(config.Lx)
            pts[which == 2, 1] = 0.0
            pts[which == 3, 1] = float(config.Ly)
        return pts

    def _side_normals_from_points(pts: torch.Tensor) -> torch.Tensor:
        if pts.shape[0] == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        d_x0 = torch.abs(x - 0.0)
        d_x1 = torch.abs(x - float(config.Lx))
        d_y0 = torch.abs(y - 0.0)
        d_y1 = torch.abs(y - float(config.Ly))
        d = torch.cat([d_x0, d_x1, d_y0, d_y1], dim=1)
        which = torch.argmin(d, dim=1)
        nrm = torch.zeros((pts.shape[0], 3), dtype=torch.float32)
        nrm[which == 0, 0] = -1.0
        nrm[which == 1, 0] = 1.0
        nrm[which == 2, 1] = -1.0
        nrm[which == 3, 1] = 1.0
        return nrm

    # Interior (optionally bias under the load patch)
    n_int = int(config.N_INTERIOR)
    frac_patch = float(getattr(config, "UNDER_PATCH_FRACTION", 0.0))
    frac_patch = max(0.0, min(frac_patch, 1.0))
    n_patch = int(round(n_int * frac_patch))
    n_uniform = max(0, n_int - n_patch)

    parts = []
    if n_uniform > 0:
        if prev_data is not None and residuals is not None and "interior" in residuals:
            prev_int = prev_data["interior"][0]
            prev_res = residuals["interior"].to(dtype=torch.float32)
            parts.append(_resample_region(prev_int, prev_res, n_uniform, region="interior"))
        else:
            params_u = _sample_layer_params(n_uniform)
            T_u = _total_thickness_from_params(params_u)
            xu = _uniform(n_uniform, 0.0, float(config.Lx))
            yu = _uniform(n_uniform, 0.0, float(config.Ly))
            zu = torch.rand(n_uniform, 1) * T_u
            parts.append(_assemble_input(torch.cat([xu, yu, zu], dim=1), params_u))
    if n_patch > 0:
        if prev_data is not None and residuals is not None and "interior" in residuals:
            prev_int = prev_data["interior"][0]
            prev_res = residuals["interior"].to(dtype=torch.float32)
            pts = _resample_region(prev_int, prev_res, n_patch, region="interior")
            pts[:, 0:1] = torch.clamp(pts[:, 0:1], float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
            pts[:, 1:2] = torch.clamp(pts[:, 1:2], float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
            parts.append(pts)
        else:
            params_p = _sample_layer_params(n_patch)
            T_p = _total_thickness_from_params(params_p)
            xp = _uniform(n_patch, float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
            yp = _uniform(n_patch, float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
            zp = torch.rand(n_patch, 1) * T_p
            parts.append(_assemble_input(torch.cat([xp, yp, zp], dim=1), params_p))

    interior = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    # Extra uniformly-sampled points for unbiased energy/work integration.
    # These are independent of residual-based resampling and patch-bias, by design.
    if bool(getattr(config, "ENERGY_UNBIASED_SAMPLES", False)):
        n_int_e = int(getattr(config, "N_INTERIOR_ENERGY", 0))
        if n_int_e > 0:
            params_e = _sample_layer_params(n_int_e)
            T_e = _total_thickness_from_params(params_e)
            xe = _uniform(n_int_e, 0.0, float(config.Lx))
            ye = _uniform(n_int_e, 0.0, float(config.Ly))
            ze = torch.rand(n_int_e, 1) * T_e
            interior_energy = _assemble_input(torch.cat([xe, ye, ze], dim=1), params_e)
        else:
            interior_energy = interior[:0]

        n_top_e = int(getattr(config, "N_TOP_LOAD_ENERGY", 0))
        if n_top_e > 0:
            params_te = _sample_layer_params(n_top_e)
            T_te = _total_thickness_from_params(params_te)
            xte = _uniform(n_top_e, float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
            yte = _uniform(n_top_e, float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
            zte = T_te
            top_load_energy = _assemble_input(torch.cat([xte, yte, zte], dim=1), params_te)
        else:
            # `top_load` is defined later; use an empty tensor with the correct shape now.
            top_load_energy = interior[:0]
    else:
        interior_energy = None
        top_load_energy = None

    # Bottom plane samples (traction-free in box mode; Dirichlet handled on side faces to match FEA).
    n_bot = int(config.N_BOTTOM)
    if prev_data is not None and residuals is not None and "bottom" in residuals:
        bottom = _resample_region(prev_data["bottom"], residuals["bottom"].to(dtype=torch.float32), n_bot, region="bottom")
    else:
        params_bot = _sample_layer_params(n_bot)
        xb = _uniform(n_bot, 0.0, float(config.Lx))
        yb = _uniform(n_bot, 0.0, float(config.Ly))
        zb = torch.zeros(n_bot, 1)
        bottom = _assemble_input(torch.cat([xb, yb, zb], dim=1), params_bot)

    # Top surface (split into load patch and free)
    n_load = int(config.N_TOP_LOAD)
    if prev_data is not None and residuals is not None and "top_load" in residuals:
        top_load = _resample_region(prev_data["top_load"], residuals["top_load"].to(dtype=torch.float32), n_load, region="top_load")
        top_load[:, 0:1] = torch.clamp(top_load[:, 0:1], float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
        top_load[:, 1:2] = torch.clamp(top_load[:, 1:2], float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
    else:
        params_load = _sample_layer_params(n_load)
        T_load = _total_thickness_from_params(params_load)
        xl, yl = _sample_xy_on_patch_biased(n_load)
        zl = T_load
        top_load = _assemble_input(torch.cat([xl, yl, zl], dim=1), params_load)

    n_top_free = int(config.N_TOP_FREE)
    if prev_data is not None and residuals is not None and "top_free" in residuals:
        top_free = _resample_region(prev_data["top_free"], residuals["top_free"].to(dtype=torch.float32), n_top_free, region="top_free")
        xt = top_free[:, 0:1]
        yt = top_free[:, 1:2]
        in_patch = (
            (xt[:, 0] >= float(config.LOAD_PATCH_X[0]))
            & (xt[:, 0] <= float(config.LOAD_PATCH_X[1]))
            & (yt[:, 0] >= float(config.LOAD_PATCH_Y[0]))
            & (yt[:, 0] <= float(config.LOAD_PATCH_Y[1]))
        )
        if bool(in_patch.any()):
            replace_n = int(in_patch.sum().item())
            params_tf = _sample_layer_params(replace_n)
            T_tf = _total_thickness_from_params(params_tf)
            xr = _uniform(replace_n * 2, 0.0, float(config.Lx))
            yr = _uniform(replace_n * 2, 0.0, float(config.Ly))
            in_patch_r = (
                (xr[:, 0] >= float(config.LOAD_PATCH_X[0]))
                & (xr[:, 0] <= float(config.LOAD_PATCH_X[1]))
                & (yr[:, 0] >= float(config.LOAD_PATCH_Y[0]))
                & (yr[:, 0] <= float(config.LOAD_PATCH_Y[1]))
            )
            xr = xr[~in_patch_r][:replace_n]
            yr = yr[~in_patch_r][:replace_n]
            if xr.shape[0] < replace_n:
                xr = _uniform(replace_n, 0.0, float(config.Lx))
                yr = _uniform(replace_n, 0.0, float(config.Ly))
            zr = T_tf
            repl = _assemble_input(torch.cat([xr, yr, zr], dim=1), params_tf)
            top_free[in_patch] = repl
    else:
        params_tf = _sample_layer_params(n_top_free)
        T_tf = _total_thickness_from_params(params_tf)
        ring_frac = float(getattr(config, "TOP_FREE_RING_FRACTION", 0.0))
        ring_frac = max(0.0, min(ring_frac, 1.0))
        n_ring = int(round(n_top_free * ring_frac))
        n_uni = max(0, n_top_free - n_ring)

        xt_parts, yt_parts = [], []
        if n_ring > 0:
            xr, yr = _sample_xy_top_free_ring(n_ring)
            xt_parts.append(xr)
            yt_parts.append(yr)
        if n_uni > 0:
            xt = _uniform(n_uni * 2, 0.0, float(config.Lx))
            yt = _uniform(n_uni * 2, 0.0, float(config.Ly))
            in_patch = (
                (xt[:, 0] >= float(config.LOAD_PATCH_X[0]))
                & (xt[:, 0] <= float(config.LOAD_PATCH_X[1]))
                & (yt[:, 0] >= float(config.LOAD_PATCH_Y[0]))
                & (yt[:, 0] <= float(config.LOAD_PATCH_Y[1]))
            )
            xt = xt[~in_patch][:n_uni]
            yt = yt[~in_patch][:n_uni]
            if xt.shape[0] < n_uni:
                xt = _uniform(n_uni, 0.0, float(config.Lx))
                yt = _uniform(n_uni, 0.0, float(config.Ly))
            xt_parts.append(xt)
            yt_parts.append(yt)

        xt_all = torch.cat(xt_parts, dim=0) if len(xt_parts) > 1 else xt_parts[0]
        yt_all = torch.cat(yt_parts, dim=0) if len(yt_parts) > 1 else yt_parts[0]
        zt = T_tf
        top_free = _assemble_input(torch.cat([xt_all, yt_all, zt], dim=1), params_tf)

    # Side walls: either clamped (Dirichlet) to match FEA, or traction-free.
    n_side = int(config.N_SIDES)
    n_face = max(1, n_side // 4)
    side_pts = []
    side_normals = []
    for face in ["x0", "x1", "y0", "y1"]:
        params = _sample_layer_params(n_face)
        T = _total_thickness_from_params(params)
        zf = torch.rand(n_face, 1) * T
        if face == "x0":
            xf = torch.zeros(n_face, 1)
            yf = _uniform(n_face, 0.0, float(config.Ly))
            nrm = torch.tensor([-1.0, 0.0, 0.0]).view(1, 3).repeat(n_face, 1)
        elif face == "x1":
            xf = torch.full((n_face, 1), float(config.Lx))
            yf = _uniform(n_face, 0.0, float(config.Ly))
            nrm = torch.tensor([1.0, 0.0, 0.0]).view(1, 3).repeat(n_face, 1)
        elif face == "y0":
            xf = _uniform(n_face, 0.0, float(config.Lx))
            yf = torch.zeros(n_face, 1)
            nrm = torch.tensor([0.0, -1.0, 0.0]).view(1, 3).repeat(n_face, 1)
        else:
            xf = _uniform(n_face, 0.0, float(config.Lx))
            yf = torch.full((n_face, 1), float(config.Ly))
            nrm = torch.tensor([0.0, 1.0, 0.0]).view(1, 3).repeat(n_face, 1)
        pts = _assemble_input(torch.cat([xf, yf, zf], dim=1), params)
        side_pts.append(pts)
        side_normals.append(nrm)
    side_all = torch.cat(side_pts, dim=0)[:n_side]
    side_all_normal = torch.cat(side_normals, dim=0)[:n_side].to(dtype=torch.float32)

    clamp_sides = bool(getattr(config, "BOX_CLAMP_SIDES", True))
    if prev_data is not None and residuals is not None:
        if clamp_sides and "sides" in residuals and "sides" in prev_data:
            side_all = _resample_region(prev_data["sides"][0], residuals["sides"].to(dtype=torch.float32), n_side, region="sides")
            side_all_normal = _side_normals_from_points(side_all)
        if (not clamp_sides) and "side_free" in residuals and "side_free" in prev_data:
            side_all = _resample_region(prev_data["side_free"], residuals["side_free"].to(dtype=torch.float32), n_side, region="sides")
            side_all_normal = _side_normals_from_points(side_all)

    if clamp_sides:
        sides = [side_all]
        side_free = side_all[:0]
        side_free_normal = side_all_normal[:0]
    else:
        sides = None
        side_free = side_all
        side_free_normal = side_all_normal

    # Interface samples (one plane per interface).
    n_intf = int(getattr(config, "N_INTERFACES", 4000))
    params_i = _sample_layer_params(n_intf)
    z_intfs = _interfaces_from_params(params_i)
    xi = _uniform(n_intf, 0.0, float(config.Lx))
    yi = _uniform(n_intf, 0.0, float(config.Ly))
    interfaces = [_assemble_input(torch.cat([xi, yi, zi], dim=1), params_i) for zi in z_intfs]

    # Near-interface band samples (for displacement continuity smoothing).
    n_band = int(getattr(config, "N_INTERFACE_BAND", 0))
    if n_band > 0:
        params_b = _sample_layer_params(n_band)
        T_b = _total_thickness_from_params(params_b)
        z_intfs_b = _interfaces_from_params(params_b)
        band_frac = float(getattr(config, "INTERFACE_BAND_FRAC", 0.05))
        band = (band_frac * T_b).clamp_min(1e-6)
        xb = _uniform(n_band, 0.0, float(config.Lx))
        yb = _uniform(n_band, 0.0, float(config.Ly))
        interfaces_band = []
        for zi in z_intfs_b:
            delta = (torch.rand(n_band, 1) - 0.5) * 2.0 * band
            zb = torch.clamp(zi + delta, min=0.0)
            zb = torch.minimum(zb, T_b)
            interfaces_band.append(_assemble_input(torch.cat([xb, yb, zb], dim=1), params_b))
    else:
        interfaces_band = None

    out = {
        "interior": [interior],
        "bottom": bottom,
        "top_load": top_load,
        "top_load_normal": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_load.shape[0], 1),
        "top_free": top_free,
        "top_free_normal": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_free.shape[0], 1),
        "side_free": side_free,
        "side_free_normal": side_free_normal.to(dtype=torch.float32),
        "interfaces": interfaces,
    }
    if interior_energy is not None:
        out["interior_energy"] = interior_energy
    if top_load_energy is not None:
        out["top_load_energy"] = top_load_energy
    if interfaces_band is not None:
        out["interfaces_band"] = interfaces_band
    if sides is not None:
        out["sides"] = sides
    return out


def _cad_layer_params_for_points(n: int, thickness: float) -> torch.Tensor:
    # Sample layer params but force the total thickness to match the CAD z-extent.
    return _sample_layer_params(n, total_thickness=float(thickness))


def get_data_cad(prev_data=None, residuals=None):
    """
    CAD/STL-backed sampling.

    Current PiNN physics assumes a plate-like domain aligned with axes (top/bottom normals
    aligned to +z/-z, and clamped sides at x/y min/max). For "simple CAD" plates exported
    as STL, we approximate the domain as the STL's axis-aligned bounding box.
    """
    stl_path = getattr(config, "CAD_STL_PATH", None)
    if not stl_path:
        raise ValueError("`CAD_STL_PATH` must be set when `GEOMETRY_MODE='cad'`.")

    sampler = str(getattr(config, "CAD_SAMPLER", "aabb")).lower()
    if sampler not in {"aabb", "tessellation"}:
        raise ValueError(f"Unknown CAD_SAMPLER: {sampler}")

    b = stl_bounds(stl_path)
    z_span = float(max(1e-12, b.size[2]))

    # Optionally normalize CAD coordinates to the training coordinate system.
    # This keeps compatibility with hard side BC masks and the existing load patch definition.
    if getattr(config, "CAD_NORMALIZE_TO_CONFIG_BOUNDS", True):
        dst_min = (0.0, 0.0, 0.0)
        dst_max = (float(config.Lx), float(config.Ly), float(config.H))
        thickness = float(config.H)
    else:
        dst_min = tuple(map(float, b.min_xyz))
        dst_max = tuple(map(float, b.max_xyz))
        thickness = z_span

    # Residual-based sampling is currently disabled for CAD mode; it relies on the
    # previous sampling distributions for each face type (box mode).
    use_residual = False
    _ = (prev_data, residuals, use_residual)

    if sampler == "aabb":
        # Sampling: interior/boundaries in the (possibly normalized) AABB.
        interior_xyz = sample_uniform_box(int(config.N_INTERIOR), dst_min, dst_max)

        # Sides: x=0/x=Lx/y=0/y=Ly; keep z uniform.
        n_face = int(config.N_SIDES) // 4
        x0 = sample_uniform_box(
            n_face,
            (dst_min[0], dst_min[1], dst_min[2]),
            (dst_min[0], dst_max[1], dst_max[2]),
        )
        x1 = sample_uniform_box(
            n_face,
            (dst_max[0], dst_min[1], dst_min[2]),
            (dst_max[0], dst_max[1], dst_max[2]),
        )
        y0 = sample_uniform_box(
            n_face,
            (dst_min[0], dst_min[1], dst_min[2]),
            (dst_max[0], dst_min[1], dst_max[2]),
        )
        y1 = sample_uniform_box(
            n_face,
            (dst_min[0], dst_max[1], dst_min[2]),
            (dst_max[0], dst_max[1], dst_max[2]),
        )
        sides_xyz = np.concatenate([x0, x1, y0, y1], axis=0)

        # Top (z = max): split into load patch vs free surface using existing patch bounds.
        z_top = float(dst_max[2])
        top_load_xyz = sample_uniform_rect_on_plane(
            int(config.N_TOP_LOAD),
            float(config.LOAD_PATCH_X[0]),
            float(config.LOAD_PATCH_X[1]),
            float(config.LOAD_PATCH_Y[0]),
            float(config.LOAD_PATCH_Y[1]),
            z_top,
        )
        # Top free: sample full top, then reject points inside load patch.
        top_free_xyz = sample_uniform_rect_on_plane(
            int(config.N_TOP_FREE) * 2,
            float(dst_min[0]),
            float(dst_max[0]),
            float(dst_min[1]),
            float(dst_max[1]),
            z_top,
        )
        in_patch = (
            (top_free_xyz[:, 0] >= float(config.LOAD_PATCH_X[0]))
            & (top_free_xyz[:, 0] <= float(config.LOAD_PATCH_X[1]))
            & (top_free_xyz[:, 1] >= float(config.LOAD_PATCH_Y[0]))
            & (top_free_xyz[:, 1] <= float(config.LOAD_PATCH_Y[1]))
        )
        top_free_xyz = top_free_xyz[~in_patch]
        if top_free_xyz.shape[0] < int(config.N_TOP_FREE):
            top_free_xyz = sample_uniform_rect_on_plane(
                int(config.N_TOP_FREE),
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                z_top,
            )
        else:
            top_free_xyz = top_free_xyz[: int(config.N_TOP_FREE)]

        # Bottom (z = min)
        z_bot = float(dst_min[2])
        bottom_xyz = sample_uniform_rect_on_plane(
            int(config.N_BOTTOM),
            float(dst_min[0]),
            float(dst_max[0]),
            float(dst_min[1]),
            float(dst_max[1]),
            z_bot,
        )
    else:
        # PhysicsNeMo-like tessellation workflow:
        # - boundary points sampled on surface triangles (with normals)
        # - interior points sampled by rejecting points with point-in-mesh test; SDF is available for debugging
        surface = load_stl_surface(stl_path)
        if getattr(config, "CAD_NORMALIZE_TO_CONFIG_BOUNDS", True):
            surface = affine_map_surface_to_bounds(surface, dst_min, dst_max)
            bounds_min = dst_min
            bounds_max = dst_max
        else:
            bounds_min = tuple(map(float, surface.bounds_min))
            bounds_max = tuple(map(float, surface.bounds_max))

        interior_dict = tess_sample_interior(
            surface,
            int(config.N_INTERIOR),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            compute_sdf=False,
            compute_sdf_derivatives=False,
        )
        interior_xyz = np.concatenate([interior_dict["x"], interior_dict["y"], interior_dict["z"]], axis=1)

        # Auto boundary-condition split by z:
        # - bottom_clamp: bottom cap (Dirichlet u=0) using z <= z_min + eps
        # - top_load: top cap within XY patch (pressure traction)
        # - top_free: top cap outside XY patch (traction-free)
        # - side_free: all remaining boundary points (traction-free)
        clamp_frac = float(getattr(config, "CAD_CLAMP_Z_FRAC", 0.02))
        load_frac = float(getattr(config, "CAD_LOAD_Z_FRAC", 0.02))
        use_normal_filter = bool(getattr(config, "CAD_BC_NORMAL_FILTER", False))
        cos_min = float(getattr(config, "CAD_BC_NORMAL_COS_MIN", 0.0))
        z_min = float(surface.bounds_min[2])
        z_max = float(surface.bounds_max[2])
        z_span_eff = max(1e-12, z_max - z_min)
        z_clamp = z_min + clamp_frac * z_span_eff
        z_load = z_max - load_frac * z_span_eff

        need_clamp = int(config.N_BOTTOM)
        need_load = int(config.N_TOP_LOAD)
        need_top_free = int(config.N_TOP_FREE)
        need_side_free = int(config.N_SIDES)

        clamp_pts_list, clamp_n_list = [], []
        load_pts_list, load_n_list = [], []
        top_free_pts_list, top_free_n_list = [], []
        side_free_pts_list, side_free_n_list = [], []

        total_bnd = 0
        load_bnd = 0

        def _stack_region(pts_list, n_list, n_take):
            if not pts_list:
                return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
            pts = np.concatenate(pts_list, axis=0)
            nrm = np.concatenate(n_list, axis=0)
            return pts[:n_take], nrm[:n_take]

        batch = int(10 * (need_clamp + need_load + need_top_free + need_side_free))
        for _ in range(20):
            boundary = tess_sample_boundary(surface, batch)
            bx = boundary["x"][:, 0]
            by = boundary["y"][:, 0]
            bz = boundary["z"][:, 0]
            bnx = boundary["normal_x"][:, 0]
            bny = boundary["normal_y"][:, 0]
            bnz = boundary["normal_z"][:, 0]

            pts = np.stack([bx, by, bz], axis=1)
            nrm = np.stack([bnx, bny, bnz], axis=1)

            is_clamp = bz <= z_clamp
            is_top = bz >= z_load
            if use_normal_filter and cos_min > 0.0:
                is_clamp = is_clamp & (bnz <= -cos_min)
                is_top = is_top & (bnz >= cos_min)
            in_patch = (
                (bx >= float(config.LOAD_PATCH_X[0]))
                & (bx <= float(config.LOAD_PATCH_X[1]))
                & (by >= float(config.LOAD_PATCH_Y[0]))
                & (by <= float(config.LOAD_PATCH_Y[1]))
            )
            is_load = is_top & in_patch
            is_top_free = is_top & (~in_patch) & (~is_clamp)
            is_side_free = ~(is_clamp | is_load | is_top_free)

            total_bnd += int(len(bz))
            load_bnd += int(np.count_nonzero(is_load))

            if sum(p.shape[0] for p in clamp_pts_list) < need_clamp:
                clamp_pts_list.append(pts[is_clamp])
                clamp_n_list.append(nrm[is_clamp])
            if sum(p.shape[0] for p in load_pts_list) < need_load:
                load_pts_list.append(pts[is_load])
                load_n_list.append(nrm[is_load])
            if sum(p.shape[0] for p in top_free_pts_list) < need_top_free:
                top_free_pts_list.append(pts[is_top_free])
                top_free_n_list.append(nrm[is_top_free])
            if sum(p.shape[0] for p in side_free_pts_list) < need_side_free:
                side_free_pts_list.append(pts[is_side_free])
                side_free_n_list.append(nrm[is_side_free])

            if (
                sum(p.shape[0] for p in clamp_pts_list) >= need_clamp
                and sum(p.shape[0] for p in load_pts_list) >= need_load
                and sum(p.shape[0] for p in top_free_pts_list) >= need_top_free
                and sum(p.shape[0] for p in side_free_pts_list) >= need_side_free
            ):
                break

        sides_xyz, sides_n = _stack_region(clamp_pts_list, clamp_n_list, need_clamp)
        top_load_xyz, top_load_n = _stack_region(load_pts_list, load_n_list, need_load)
        top_free_xyz, top_free_n = _stack_region(top_free_pts_list, top_free_n_list, need_top_free)
        side_free_xyz, side_free_n = _stack_region(side_free_pts_list, side_free_n_list, need_side_free)

        mesh_area_total = float(np.sum(surface.tri_areas))
        if total_bnd > 0:
            top_load_area = mesh_area_total * (float(load_bnd) / float(total_bnd))
        else:
            top_load_area = 0.0

        if sides_xyz.shape[0] < need_clamp:
            sides_xyz = sample_uniform_rect_on_plane(
                need_clamp,
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                float(dst_min[2]),
            )
            sides_n = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (need_clamp, 1))
        if top_load_xyz.shape[0] < need_load:
            top_load_xyz = sample_uniform_rect_on_plane(
                need_load,
                float(config.LOAD_PATCH_X[0]),
                float(config.LOAD_PATCH_X[1]),
                float(config.LOAD_PATCH_Y[0]),
                float(config.LOAD_PATCH_Y[1]),
                float(dst_max[2]),
            )
            top_load_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (need_load, 1))
        if top_free_xyz.shape[0] < need_top_free:
            top_free_xyz = sample_uniform_rect_on_plane(
                need_top_free,
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                float(dst_max[2]),
            )
            top_free_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (need_top_free, 1))
        if side_free_xyz.shape[0] < need_side_free:
            # Fallback: sample from the CAD AABB faces as a proxy for side walls.
            side_free_xyz = sample_uniform_box(
                need_side_free,
                (float(dst_min[0]), float(dst_min[1]), float(dst_min[2])),
                (float(dst_max[0]), float(dst_max[1]), float(dst_max[2])),
            )
            side_free_n = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (need_side_free, 1))

        bottom_xyz = None

    # NOTE: The PiNN is trained/defined in the returned coordinate system. If you disable
    # normalization, you must also disable hard side BC masks (`USE_HARD_SIDE_BC`) or
    # ensure coordinates are compatible with the mask logic in `pinn-workflow/model.py`.

    # Assemble tensors (x,y,z,E1,t1,...,EL,tL,r,mu,v0)
    interior_t = torch.tensor(interior_xyz, dtype=torch.float32)
    top_load_t = torch.tensor(top_load_xyz, dtype=torch.float32)
    top_free_t = torch.tensor(top_free_xyz, dtype=torch.float32)
    bottom_t = torch.tensor(bottom_xyz, dtype=torch.float32) if bottom_xyz is not None else None

    interior = torch.cat([interior_t, _cad_layer_params_for_points(len(interior_t), thickness)], dim=1)
    top_load = torch.cat([top_load_t, _cad_layer_params_for_points(len(top_load_t), thickness)], dim=1)
    top_free = torch.cat([top_free_t, _cad_layer_params_for_points(len(top_free_t), thickness)], dim=1)
    if bottom_t is not None:
        bottom_clamp = torch.cat([bottom_t, _cad_layer_params_for_points(len(bottom_t), thickness)], dim=1)
    else:
        sides_t = torch.tensor(sides_xyz, dtype=torch.float32)
        bottom_clamp = torch.cat([sides_t, _cad_layer_params_for_points(len(sides_t), thickness)], dim=1)

    # Side walls: in AABB mode we already have explicit side samples; in tessellation mode we split.
    if sampler == "aabb":
        sides_t = torch.tensor(sides_xyz, dtype=torch.float32)
        side_free = torch.cat([sides_t, _cad_layer_params_for_points(len(sides_t), thickness)], dim=1)
        # Build outward normals for the axis-aligned AABB faces.
        # Note: sides_xyz is concatenated [x0,x1,y0,y1] with equal n_face sizing in this function.
        n_face = max(1, int(config.N_SIDES) // 4)
        nrm = torch.cat(
            [
                torch.tensor([-1.0, 0.0, 0.0]).view(1, 3).repeat(n_face, 1),
                torch.tensor([1.0, 0.0, 0.0]).view(1, 3).repeat(n_face, 1),
                torch.tensor([0.0, -1.0, 0.0]).view(1, 3).repeat(n_face, 1),
                torch.tensor([0.0, 1.0, 0.0]).view(1, 3).repeat(n_face, 1),
            ],
            dim=0,
        )[: side_free.shape[0]]
        side_free_normal = nrm.to(dtype=torch.float32)
    else:
        # tessellation: side walls are returned explicitly as side_free_xyz/side_free_n
        side_free_t = torch.tensor(side_free_xyz, dtype=torch.float32)
        side_free = torch.cat([side_free_t, _cad_layer_params_for_points(len(side_free_t), thickness)], dim=1)
        side_free_normal = torch.tensor(side_free_n, dtype=torch.float32)

    # Interfaces in CAD mode: sample (x,y) in bounds and z at layer interfaces; force sum thickness=CAD thickness.
    n_intf = int(getattr(config, "N_INTERFACES", 4000))
    params_i = _cad_layer_params_for_points(n_intf, thickness)
    z_intfs = _interfaces_from_params(params_i)
    # Use CAD bounds in returned coordinate system (normalized or raw).
    bmin = torch.tensor(dst_min, dtype=torch.float32).view(1, 3)
    bmax = torch.tensor(dst_max, dtype=torch.float32).view(1, 3)
    xi = _uniform(n_intf, float(bmin[0, 0]), float(bmax[0, 0]))
    yi = _uniform(n_intf, float(bmin[0, 1]), float(bmax[0, 1]))
    interfaces = [_assemble_input(torch.cat([xi, yi, zi], dim=1), params_i) for zi in z_intfs]

    out = {
        "interior": [interior],
        "bottom_clamp": bottom_clamp,
        "top_load": top_load,
        "top_load_normal": torch.tensor(top_load_n, dtype=torch.float32) if sampler == "tessellation" else torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_load.shape[0], 1),
        "top_free": top_free,
        "top_free_normal": torch.tensor(top_free_n, dtype=torch.float32) if sampler == "tessellation" else torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_free.shape[0], 1),
        "side_free": side_free,
        "side_free_normal": side_free_normal,
        "interfaces": interfaces,
    }
    if sampler == "tessellation":
        out["top_load_area"] = float(top_load_area)
        if "volume" in interior_dict:
            out["domain_volume"] = float(np.asarray(interior_dict["volume"]).reshape(-1)[0])
    return out
