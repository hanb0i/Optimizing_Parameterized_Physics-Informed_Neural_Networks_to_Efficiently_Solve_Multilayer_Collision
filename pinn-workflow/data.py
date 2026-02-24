
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

# -----------------------
# 3-layer laminate inputs
# -----------------------
#
# Input layout used across the PiNN:
#   [x, y, z, E1, t1, E2, t2, E3, t3, restitution, friction, impact_velocity]
#
# Total thickness is T = t1+t2+t3, and interfaces are at z=t1 and z=t1+t2.

_DIN = 12


def _uniform(n: int, lo: float, hi: float, *, device=None) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (hi - lo) + lo


def _sample_layer_params(n: int, *, total_thickness: float | None = None, device=None) -> torch.Tensor:
    """
    Returns params tensor shaped (n,9):
      [E1,t1,E2,t2,E3,t3,r,mu,v0]
    """
    if bool(getattr(config, "TRAIN_FIXED_PARAMS", False)):
        E = float(getattr(config, "TRAIN_FIXED_E", 1.0))
        T = float(getattr(config, "TRAIN_FIXED_TOTAL_THICKNESS", getattr(config, "H", 0.1)))
        t1 = torch.full((n, 1), T / 3.0, dtype=torch.float32, device=device).clamp_min(1e-4)
        t2 = torch.full((n, 1), T / 3.0, dtype=torch.float32, device=device).clamp_min(1e-4)
        t3 = torch.full((n, 1), T / 3.0, dtype=torch.float32, device=device).clamp_min(1e-4)
        E1 = torch.full((n, 1), E, dtype=torch.float32, device=device)
        E2 = torch.full((n, 1), E, dtype=torch.float32, device=device)
        E3 = torch.full((n, 1), E, dtype=torch.float32, device=device)
        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        r = torch.full((n, 1), r_ref, dtype=torch.float32, device=device)
        mu = torch.full((n, 1), mu_ref, dtype=torch.float32, device=device)
        v0 = torch.full((n, 1), v0_ref, dtype=torch.float32, device=device)
        return torch.cat([E1, t1, E2, t2, E3, t3, r, mu, v0], dim=1)

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
    frac_min = max(0.0, min(frac_min, 1.0 / 3.0 - 1e-6))
    raw = torch.rand(n, 3, device=device)
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    frac = frac_min + (1.0 - 3.0 * frac_min) * raw

    t1 = (T * frac[:, 0:1]).clamp_min(1e-4)
    t2 = (T * frac[:, 1:2]).clamp_min(1e-4)
    t3 = (T * frac[:, 2:3]).clamp_min(1e-4)

    # Re-normalize to guarantee exact sum T after clamping.
    tsum = (t1 + t2 + t3).clamp_min(1e-12)
    scale = T / tsum
    t1 = t1 * scale
    t2 = t2 * scale
    t3 = t3 * scale

    E1 = _uniform(n, e_min, e_max, device=device)
    E2 = _uniform(n, e_min, e_max, device=device)
    E3 = _uniform(n, e_min, e_max, device=device)
    r = _uniform(n, r_min, r_max, device=device)
    mu = _uniform(n, mu_min, mu_max, device=device)
    v0 = _uniform(n, v0_min, v0_max, device=device)

    return torch.cat([E1, t1, E2, t2, E3, t3, r, mu, v0], dim=1)


def _assemble_input(xyz: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shaped (n,3), got {tuple(xyz.shape)}")
    if params.ndim != 2 or params.shape[1] != 9:
        raise ValueError(f"Expected params shaped (n,9), got {tuple(params.shape)}")
    x = torch.cat([xyz, params], dim=1)
    if x.shape[1] != _DIN:
        raise ValueError(f"Internal error: expected input dim {_DIN}, got {x.shape[1]}")
    return x


def _total_thickness_from_params(params: torch.Tensor) -> torch.Tensor:
    return (params[:, 1:2] + params[:, 3:4] + params[:, 5:6]).clamp_min(1e-8)


def _interfaces_from_params(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = params[:, 1:2].clamp_min(0.0)
    z2 = (params[:, 1:2] + params[:, 3:4]).clamp_min(0.0)
    return z1, z2

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
    
    if n_points_per_e is None:
        if hasattr(config, "N_DATA_POINTS"):
            n_points_per_e = config.N_DATA_POINTS // max(1, (len(e_values) * len(thickness_values)))
        else:
            n_points_per_e = 0
    
    x_data_list = []
    u_data_list = []
    
    for thickness in thickness_values:
        for E_val in e_values:
            print(f"  Generating FEM supervision for E={E_val}, thickness={thickness}...")
            
            # Run FEM solver
            cfg = {
                'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': thickness},
                'material': {'E': E_val, 'nu': config.nu_vals[0]},
                'load_patch': {
                    'pressure': config.p0,
                    'x_start': config.LOAD_PATCH_X[0] / config.Lx,
                    'x_end': config.LOAD_PATCH_X[1] / config.Lx,
                    'y_start': config.LOAD_PATCH_Y[0] / config.Ly,
                    'y_end': config.LOAD_PATCH_Y[1] / config.Ly
                }
            }
            x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
            
            # Create mesh grid for all FEM nodes
            nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
            X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
            
            # Flatten to get all points
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            u_flat = u_grid.reshape(-1, 3)
            
            # Random sampling (sparse)
            total_points = len(x_flat)
            indices = np.random.choice(total_points, size=min(n_points_per_e, total_points), replace=False)
            
            # Create input points with (E1,t1,E2,t2,E3,t3) and global params.
            r_min, r_max = _get_restitution_range()
            mu_min, mu_max = _get_friction_range()
            v0_min, v0_max = _get_impact_velocity_range()
            restitution = np.ones(len(indices)) * (0.5 * (r_min + r_max))
            friction = np.ones(len(indices)) * (0.5 * (mu_min + mu_max))
            impact_velocity = np.ones(len(indices)) * (0.5 * (v0_min + v0_max))
            t1 = np.ones(len(indices)) * (thickness / 3.0)
            t2 = np.ones(len(indices)) * (thickness / 3.0)
            t3 = np.ones(len(indices)) * (thickness / 3.0)

            x_sampled = np.stack([
                x_flat[indices],
                y_flat[indices],
                z_flat[indices],
                np.ones(len(indices)) * E_val,
                t1,
                np.ones(len(indices)) * E_val,
                t2,
                np.ones(len(indices)) * E_val,
                t3,
                restitution,
                friction,
                impact_velocity
            ], axis=1)
            
            u_sampled = u_flat[indices]
            
            x_data_list.append(torch.tensor(x_sampled, dtype=torch.float32))
            u_data_list.append(torch.tensor(u_sampled, dtype=torch.float32))
    
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

    # NOTE: This 3-layer laminate sampler currently ignores residual-based adaptive sampling.
    # It can be reintroduced after the new BC key structure stabilizes.
    _ = (prev_data, residuals)

    # Interior (optionally bias under the load patch)
    n_int = int(config.N_INTERIOR)
    frac_patch = float(getattr(config, "UNDER_PATCH_FRACTION", 0.0))
    frac_patch = max(0.0, min(frac_patch, 1.0))
    n_patch = int(round(n_int * frac_patch))
    n_uniform = max(0, n_int - n_patch)

    parts = []
    if n_uniform > 0:
        params_u = _sample_layer_params(n_uniform)
        T_u = _total_thickness_from_params(params_u)
        xu = _uniform(n_uniform, 0.0, float(config.Lx))
        yu = _uniform(n_uniform, 0.0, float(config.Ly))
        zu = torch.rand(n_uniform, 1) * T_u
        parts.append(_assemble_input(torch.cat([xu, yu, zu], dim=1), params_u))
    if n_patch > 0:
        params_p = _sample_layer_params(n_patch)
        T_p = _total_thickness_from_params(params_p)
        xp = _uniform(n_patch, float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
        yp = _uniform(n_patch, float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
        zp = torch.rand(n_patch, 1) * T_p
        parts.append(_assemble_input(torch.cat([xp, yp, zp], dim=1), params_p))

    interior = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    # Bottom plane samples (traction-free in box mode; Dirichlet handled on side faces to match FEA).
    n_bot = int(config.N_BOTTOM)
    params_bot = _sample_layer_params(n_bot)
    xb = _uniform(n_bot, 0.0, float(config.Lx))
    yb = _uniform(n_bot, 0.0, float(config.Ly))
    zb = torch.zeros(n_bot, 1)
    bottom = _assemble_input(torch.cat([xb, yb, zb], dim=1), params_bot)

    # Top surface (split into load patch and free)
    n_load = int(config.N_TOP_LOAD)
    params_load = _sample_layer_params(n_load)
    T_load = _total_thickness_from_params(params_load)
    xl = _uniform(n_load, float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1]))
    yl = _uniform(n_load, float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1]))
    zl = T_load
    top_load = _assemble_input(torch.cat([xl, yl, zl], dim=1), params_load)

    n_top_free = int(config.N_TOP_FREE)
    params_tf = _sample_layer_params(n_top_free)
    T_tf = _total_thickness_from_params(params_tf)
    # Sample full top and reject points inside patch.
    xt = _uniform(n_top_free * 2, 0.0, float(config.Lx))
    yt = _uniform(n_top_free * 2, 0.0, float(config.Ly))
    in_patch = (
        (xt[:, 0] >= float(config.LOAD_PATCH_X[0]))
        & (xt[:, 0] <= float(config.LOAD_PATCH_X[1]))
        & (yt[:, 0] >= float(config.LOAD_PATCH_Y[0]))
        & (yt[:, 0] <= float(config.LOAD_PATCH_Y[1]))
    )
    xt = xt[~in_patch][:n_top_free]
    yt = yt[~in_patch][:n_top_free]
    if xt.shape[0] < n_top_free:
        xt = _uniform(n_top_free, 0.0, float(config.Lx))
        yt = _uniform(n_top_free, 0.0, float(config.Ly))
    zt = T_tf
    top_free = _assemble_input(torch.cat([xt, yt, zt], dim=1), params_tf)

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
    if clamp_sides:
        sides = [side_all]
        side_free = side_all[:0]
        side_free_normal = side_all_normal[:0]
    else:
        sides = None
        side_free = side_all
        side_free_normal = side_all_normal

    # Interface samples
    n_intf = int(getattr(config, "N_INTERFACES", 4000))
    params_i = _sample_layer_params(n_intf)
    z1, z2 = _interfaces_from_params(params_i)
    xi = _uniform(n_intf, 0.0, float(config.Lx))
    yi = _uniform(n_intf, 0.0, float(config.Ly))
    intf1 = _assemble_input(torch.cat([xi, yi, z1], dim=1), params_i)
    intf2 = _assemble_input(torch.cat([xi, yi, z2], dim=1), params_i)

    out = {
        "interior": [interior],
        "bottom": bottom,
        "top_load": top_load,
        "top_load_normal": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_load.shape[0], 1),
        "top_free": top_free,
        "top_free_normal": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_free.shape[0], 1),
        "side_free": side_free,
        "side_free_normal": side_free_normal.to(dtype=torch.float32),
        "interfaces": [intf1, intf2],
    }
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

    # Assemble tensors (x,y,z,E1,t1,E2,t2,E3,t3,r,mu,v0)
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
    z1, z2 = _interfaces_from_params(params_i)
    # Use CAD bounds in returned coordinate system (normalized or raw).
    bmin = torch.tensor(dst_min, dtype=torch.float32).view(1, 3)
    bmax = torch.tensor(dst_max, dtype=torch.float32).view(1, 3)
    xi = _uniform(n_intf, float(bmin[0, 0]), float(bmax[0, 0]))
    yi = _uniform(n_intf, float(bmin[0, 1]), float(bmax[0, 1]))
    intf1 = _assemble_input(torch.cat([xi, yi, z1], dim=1), params_i)
    intf2 = _assemble_input(torch.cat([xi, yi, z2], dim=1), params_i)

    out = {
        "interior": [interior],
        "bottom_clamp": bottom_clamp,
        "top_load": top_load,
        "top_load_normal": torch.tensor(top_load_n, dtype=torch.float32) if sampler == "tessellation" else torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_load.shape[0], 1),
        "top_free": top_free,
        "top_free_normal": torch.tensor(top_free_n, dtype=torch.float32) if sampler == "tessellation" else torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(top_free.shape[0], 1),
        "side_free": side_free,
        "side_free_normal": side_free_normal,
        "interfaces": [intf1, intf2],
    }
    if sampler == "tessellation":
        out["top_load_area"] = float(top_load_area)
        if "volume" in interior_dict:
            out["domain_volume"] = float(np.asarray(interior_dict["volume"]).reshape(-1)[0])
    return out
