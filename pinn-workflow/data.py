
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
            
            # Create input points with E and thickness values
            r_min, r_max = _get_restitution_range()
            mu_min, mu_max = _get_friction_range()
            v0_min, v0_max = _get_impact_velocity_range()
            restitution = np.ones(len(indices)) * (0.5 * (r_min + r_max))
            friction = np.ones(len(indices)) * (0.5 * (mu_min + mu_max))
            impact_velocity = np.ones(len(indices)) * (0.5 * (v0_min + v0_max))

            x_sampled = np.stack([
                x_flat[indices],
                y_flat[indices],
                z_flat[indices],
                np.ones(len(indices)) * E_val,
                np.ones(len(indices)) * thickness,
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
    z_min, z_max = config.Layer_Interfaces[0], config.Layer_Interfaces[1]
    
    # Decide whether to use residual-based sampling (50% uniform, 50% residual-based)
    use_residual = (prev_data is not None and residuals is not None)
    
    n_patch = int(config.N_INTERIOR * config.UNDER_PATCH_FRACTION)
    if n_patch < 0:
        n_patch = 0
    if n_patch > config.N_INTERIOR:
        n_patch = config.N_INTERIOR
    n_interior = config.N_INTERIOR - n_patch
    
    if use_residual:
        n_uniform = n_interior // 2
        n_residual = n_interior - n_uniform
        
        # Interior: half uniform, half residual-based
        interior_uniform = sample_domain(n_uniform, z_min, z_max)
        interior_residual = sample_domain_residual_based(
            n_residual, z_min, z_max,
            prev_data['interior'][0], residuals['interior']
        )
        interior = torch.cat([interior_uniform, interior_residual], dim=0)
        if n_patch > 0:
            interior_patch = sample_domain_under_patch(n_patch, z_min, z_max)
            interior = torch.cat([interior, interior_patch], dim=0)
        
        # BC Sides: half uniform, half residual-based
        n_uniform_bc = config.N_SIDES // 2
        n_residual_bc = config.N_SIDES - n_uniform_bc
        bc_uniform = sample_boundaries(n_uniform_bc, z_min, z_max)
        bc_residual = sample_boundaries_residual_based(
            n_residual_bc, z_min, z_max,
            prev_data['sides'][0], residuals['sides']
        )
        bc_sides = torch.cat([bc_uniform, bc_residual], dim=0)
        
        # Top Load: half uniform, half residual-based
        n_uniform_load = config.N_TOP_LOAD // 2
        n_residual_load = config.N_TOP_LOAD - n_uniform_load
        load_uniform = sample_top_load(n_uniform_load)
        load_residual = sample_surface_residual_based(
            n_residual_load, config.H,
            prev_data['top_load'], residuals['top_load'],
            constrain_load_patch=True, is_load_patch=True
        )
        top_load = torch.cat([load_uniform, load_residual], dim=0)
        
        # Top Free: half uniform, half residual-based
        n_uniform_free = config.N_TOP_FREE // 2
        n_residual_free = config.N_TOP_FREE - n_uniform_free
        free_uniform = sample_top_free(n_uniform_free)
        free_residual = sample_surface_residual_based(
            n_residual_free, config.H,
            prev_data['top_free'], residuals['top_free'],
            constrain_load_patch=True, is_load_patch=False
        )
        top_free = torch.cat([free_uniform, free_residual], dim=0)
        
        # Bottom: half uniform, half residual-based
        n_uniform_bot = config.N_BOTTOM // 2
        n_residual_bot = config.N_BOTTOM - n_uniform_bot
        bot_uniform = sample_bottom(n_uniform_bot)
        bot_residual = sample_surface_residual_based(
            n_residual_bot, 0.0,
            prev_data['bottom'], residuals['bottom']
        )
        bot_free = torch.cat([bot_uniform, bot_residual], dim=0)
        
    else:
        # Uniform sampling (initial or when no residuals provided)
        interior_uniform = sample_domain(n_interior, z_min, z_max)
        if n_patch > 0:
            interior_patch = sample_domain_under_patch(n_patch, z_min, z_max)
            interior = torch.cat([interior_uniform, interior_patch], dim=0)
        else:
            interior = interior_uniform
        bc_sides = sample_boundaries(config.N_SIDES, z_min, z_max)
        top_load = sample_top_load(config.N_TOP_LOAD)
        top_free = sample_top_free(config.N_TOP_FREE)
        bot_free = sample_bottom(config.N_BOTTOM)
    
    return {
        'interior': [interior],
        'sides': [bc_sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bot_free
    }


def _cad_params_for_points(n: int, thickness: float) -> torch.Tensor:
    # Reuse the existing parameter sampling helpers where possible but keep CAD thickness fixed
    # to match the CAD domain z-extent.
    e_min, e_max = _get_e_range()
    e = torch.rand(n, 1) * (e_max - e_min) + e_min

    # Thickness: fixed to CAD z-extent for geometry consistency
    t = torch.full((n, 1), float(thickness), dtype=torch.float32)

    r_min, r_max = _get_restitution_range()
    restitution = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    friction = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    impact_velocity = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return torch.cat([e, t, restitution, friction, impact_velocity], dim=1)


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
            compute_sdf_derivatives=False,
        )
        interior_xyz = np.concatenate([interior_dict["x"], interior_dict["y"], interior_dict["z"]], axis=1)

        boundary = tess_sample_boundary(surface, int(config.N_SIDES + config.N_TOP_LOAD + config.N_TOP_FREE + config.N_BOTTOM) * 2)
        bx = boundary["x"][:, 0]
        by = boundary["y"][:, 0]
        bz = boundary["z"][:, 0]
        bnz = boundary["normal_z"][:, 0]
        nz_thresh = float(getattr(config, "CAD_NORMAL_Z_THRESH", 0.85))

        top_mask = bnz >= nz_thresh
        bot_mask = bnz <= -nz_thresh
        side_mask = ~(top_mask | bot_mask)

        top_pts = np.stack([bx[top_mask], by[top_mask], bz[top_mask]], axis=1)
        bot_pts = np.stack([bx[bot_mask], by[bot_mask], bz[bot_mask]], axis=1)
        side_pts = np.stack([bx[side_mask], by[side_mask], bz[side_mask]], axis=1)

        if side_pts.shape[0] < int(config.N_SIDES):
            # Fallback to AABB sides if STL classification is insufficient
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
        else:
            sides_xyz = side_pts[: int(config.N_SIDES)]

        z_top = float(dst_max[2])
        if top_pts.shape[0] == 0:
            top_pts = sample_uniform_rect_on_plane(
                int(config.N_TOP_LOAD + config.N_TOP_FREE),
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                z_top,
            )

        # Split top into load patch vs free
        in_patch = (
            (top_pts[:, 0] >= float(config.LOAD_PATCH_X[0]))
            & (top_pts[:, 0] <= float(config.LOAD_PATCH_X[1]))
            & (top_pts[:, 1] >= float(config.LOAD_PATCH_Y[0]))
            & (top_pts[:, 1] <= float(config.LOAD_PATCH_Y[1]))
        )
        load_pts = top_pts[in_patch]
        free_pts = top_pts[~in_patch]
        if load_pts.shape[0] < int(config.N_TOP_LOAD):
            top_load_xyz = sample_uniform_rect_on_plane(
                int(config.N_TOP_LOAD),
                float(config.LOAD_PATCH_X[0]),
                float(config.LOAD_PATCH_X[1]),
                float(config.LOAD_PATCH_Y[0]),
                float(config.LOAD_PATCH_Y[1]),
                z_top,
            )
        else:
            top_load_xyz = load_pts[: int(config.N_TOP_LOAD)]
        if free_pts.shape[0] < int(config.N_TOP_FREE):
            top_free_xyz = sample_uniform_rect_on_plane(
                int(config.N_TOP_FREE),
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                z_top,
            )
        else:
            top_free_xyz = free_pts[: int(config.N_TOP_FREE)]

        z_bot = float(dst_min[2])
        if bot_pts.shape[0] < int(config.N_BOTTOM):
            bottom_xyz = sample_uniform_rect_on_plane(
                int(config.N_BOTTOM),
                float(dst_min[0]),
                float(dst_max[0]),
                float(dst_min[1]),
                float(dst_max[1]),
                z_bot,
            )
        else:
            bottom_xyz = bot_pts[: int(config.N_BOTTOM)]

    # NOTE: The PiNN is trained/defined in the returned coordinate system. If you disable
    # normalization, you must also disable hard side BC masks (`USE_HARD_SIDE_BC`) or
    # ensure coordinates are compatible with the mask logic in `pinn-workflow/model.py`.

    # Assemble tensors (x,y,z,E,t,r,mu,v0)
    interior_t = torch.tensor(interior_xyz, dtype=torch.float32)
    sides_t = torch.tensor(sides_xyz, dtype=torch.float32)
    top_load_t = torch.tensor(top_load_xyz, dtype=torch.float32)
    top_free_t = torch.tensor(top_free_xyz, dtype=torch.float32)
    bottom_t = torch.tensor(bottom_xyz, dtype=torch.float32)

    interior = torch.cat([interior_t, _cad_params_for_points(len(interior_t), thickness)], dim=1)
    sides = torch.cat([sides_t, _cad_params_for_points(len(sides_t), thickness)], dim=1)
    top_load = torch.cat([top_load_t, _cad_params_for_points(len(top_load_t), thickness)], dim=1)
    top_free = torch.cat([top_free_t, _cad_params_for_points(len(top_free_t), thickness)], dim=1)
    bottom = torch.cat([bottom_t, _cad_params_for_points(len(bottom_t), thickness)], dim=1)

    return {
        "interior": [interior],
        "sides": [sides],
        "top_load": top_load,
        "top_free": top_free,
        "bottom": bottom,
    }
