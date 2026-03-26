
import torch
import numpy as np
import os
import pinn_config as config

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

def _get_t1_range():
    if hasattr(config, "T1_RANGE"):
        t_min, t_max = config.T1_RANGE
    else:
        t_min, t_max = float(getattr(config, "H", 0.1)) * 0.5, float(getattr(config, "H", 0.1)) * 0.5
        if t_min == t_max:
            t_max = t_min + 0.01
    return float(t_min), float(t_max)


def _get_t2_range():
    if hasattr(config, "T2_RANGE"):
        t_min, t_max = config.T2_RANGE
    else:
        t_min, t_max = float(getattr(config, "H", 0.1)) * 0.5, float(getattr(config, "H", 0.1)) * 0.5
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


def _sample_e_pairs(n):
    e_min, e_max = _get_e_range()
    e1 = torch.rand(n, 1) * (e_max - e_min) + e_min
    e2 = torch.rand(n, 1) * (e_max - e_min) + e_min
    return e1, e2


def _sample_t1(n):
    t_min, t_max = _get_t1_range()
    return torch.rand(n, 1) * (t_max - t_min) + t_min


def _sample_t2(n):
    t_min, t_max = _get_t2_range()
    return torch.rand(n, 1) * (t_max - t_min) + t_min


def _sample_param_columns(n):
    e1, e2 = _sample_e_pairs(n)
    t1 = _sample_t1(n)
    t2 = _sample_t2(n)
    r_min, r_max = _get_restitution_range()
    restitution = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu_min, mu_max = _get_friction_range()
    friction = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0_min, v0_max = _get_impact_velocity_range()
    impact_velocity = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    return e1, e2, t1, t2, restitution, friction, impact_velocity

# Import FEM solver for generating supervision data
import sys
FEA_SOLVER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fea-workflow", "solver")
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)


def load_fem_supervision_data(n_points_per_e=None, e_values=None, t1_values=None, t2_values=None):
    import fem_solver
    
    if e_values is None:
        if hasattr(config, "DATA_E_VALUES"):
            e_values = config.DATA_E_VALUES
        else:
            e_min, e_max = _get_e_range()
            e_values = [e_min, 0.5 * (e_min + e_max), e_max]
    if t1_values is None:
        if hasattr(config, "DATA_T1_VALUES"):
            t1_values = config.DATA_T1_VALUES
        else:
            t1_values = [float(getattr(config, "H", 0.1)) * 0.5]
    if t2_values is None:
        if hasattr(config, "DATA_T2_VALUES"):
            t2_values = config.DATA_T2_VALUES
        else:
            t2_values = [float(getattr(config, "H", 0.1)) * 0.5]
    
    if n_points_per_e is None:
        if hasattr(config, "N_DATA_POINTS"):
            n_points_per_e = config.N_DATA_POINTS // max(1, len(e_values) ** 2 * len(t1_values) * len(t2_values))
        else:
            n_points_per_e = 0
    
    x_data_list = []
    u_data_list = []
    
    print("  FEM supervision uses two-layer solver across E1/E2 and t1/t2 grids.")
    for t1 in t1_values:
        for t2 in t2_values:
            thickness = float(t1) + float(t2)
            for E1_val in e_values:
                for E2_val in e_values:
                    print(f"  Generating FEM supervision for E1={E1_val}, E2={E2_val}, t1={t1}, t2={t2}...")

                    cfg = {
                        'geometry': {
                            'Lx': config.Lx,
                            'Ly': config.Ly,
                            'H': thickness,
                            'ne_x': int(getattr(config, "FEM_NE_X", 30)),
                            'ne_y': int(getattr(config, "FEM_NE_Y", 30)),
                            'ne_z': int(getattr(config, "FEM_NE_Z", 10)),
                        },
                        'material': {
                            'E_layers': [E1_val, E2_val],
                            't_layers': [float(t1), float(t2)],
                            'nu': config.nu_vals[0],
                        },
                        'load_patch': {
                            'pressure': config.p0,
                            'x_start': config.LOAD_PATCH_X[0] / config.Lx,
                            'x_end': config.LOAD_PATCH_X[1] / config.Lx,
                            'y_start': config.LOAD_PATCH_Y[0] / config.Ly,
                            'y_end': config.LOAD_PATCH_Y[1] / config.Ly
                        }
                    }
                    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_two_layer_fem(cfg)

                    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
                    x_flat = X.flatten()
                    y_flat = Y.flatten()
                    z_flat = Z.flatten()
                    u_flat = u_grid.reshape(-1, 3)

                    total_points = len(x_flat)
                    indices = np.random.choice(total_points, size=min(n_points_per_e, total_points), replace=False)

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
                        np.ones(len(indices)) * E1_val,
                        np.ones(len(indices)) * float(t1),
                        np.ones(len(indices)) * E2_val,
                        np.ones(len(indices)) * float(t2),
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
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    e1, e2, t1, t2, restitution, friction, impact_velocity = _sample_param_columns(n)
    t_total = t1 + t2
    z = torch.rand(n, 1) * t_total
    return torch.cat([x, y, z, e1, t1, e2, t2, restitution, friction, impact_velocity], dim=1)

def sample_domain_under_patch(n, z_min, z_max):
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    bias_frac = float(getattr(config, "PATCH_CENTER_BIAS_FRACTION", 0.0))
    bias_frac = max(0.0, min(bias_frac, 1.0))
    n_bias = int(n * bias_frac)
    n_uniform = n - n_bias
    if n_bias > 0:
        shape = float(getattr(config, "PATCH_CENTER_BIAS_SHAPE", 2.0))
        alpha = torch.full((n_bias, 1), shape)
        beta = torch.full((n_bias, 1), shape)
        center_dist = torch.distributions.Beta(alpha, beta)
        x_bias = center_dist.sample() * (x_max - x_min) + x_min
        y_bias = center_dist.sample() * (y_max - y_min) + y_min
        x_uniform = torch.rand(n_uniform, 1) * (x_max - x_min) + x_min
        y_uniform = torch.rand(n_uniform, 1) * (y_max - y_min) + y_min
        x = torch.cat([x_bias, x_uniform], dim=0)
        y = torch.cat([y_bias, y_uniform], dim=0)
    else:
        x = torch.rand(n, 1) * (x_max - x_min) + x_min
        y = torch.rand(n, 1) * (y_max - y_min) + y_min
    e1, e2, t1, t2, restitution, friction, impact_velocity = _sample_param_columns(n)
    t_total = t1 + t2
    z = torch.rand(n, 1) * t_total
    return torch.cat([x, y, z, e1, t1, e2, t2, restitution, friction, impact_velocity], dim=1)

def sample_domain_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    if n <= 0:
        return prev_pts.new_empty((0, prev_pts.shape[1]))
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_domain(n, z_min, z_max)
    
    # Normalize residuals to probabilities
    residual_probs = prev_residuals
    if residual_probs.dim() > 1:
        residual_probs = residual_probs.mean(dim=1)
    residual_probs = residual_probs.reshape(-1)
    residual_probs = residual_probs / residual_probs.sum()
    residual_probs = residual_probs + 1e-10  # Add small epsilon for numerical stability
    residual_probs = residual_probs / residual_probs.sum()  # Renormalize
    
    # Sample indices based on residual weights
    if n <= 0:
        return prev_pts.new_empty((0, prev_pts.shape[1]))
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    # Add noise to create new points nearby
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    e_min, e_max = _get_e_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    z_span = max(z_max - z_min, 1e-6)
    noise_z = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * z_span
    t1_min, t1_max = _get_t1_range()
    t2_min, t2_max = _get_t2_range()
    noise_e1 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_e2 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_t1 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t1_max - t1_min)
    noise_t2 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t2_max - t2_min)
    noise_r = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (r_max - r_min)
    noise_mu = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    noise_v0 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (v0_max - v0_min)

    noise = torch.cat([noise_x, noise_y, noise_z, noise_e1, noise_t1, noise_e2, noise_t2, noise_r, noise_mu, noise_v0], dim=1)
    
    new_pts = sampled_pts + noise
    
    # Clamp to domain bounds
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], e_min, e_max)
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t1_min, t1_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], t2_min, t2_max)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], min=0.0)
    new_pts[:, 2] = torch.minimum(new_pts[:, 2], new_pts[:, 4] + new_pts[:, 6])
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], r_min, r_max)
    new_pts[:, 8] = torch.clamp(new_pts[:, 8], mu_min, mu_max)
    new_pts[:, 9] = torch.clamp(new_pts[:, 9], v0_min, v0_max)
    
    return new_pts

def sample_boundaries(n, z_min, z_max):
    # 4 Side faces: x=0, x=Lx, y=0, y=Ly
    # Split n among 4 faces
    n_face = n // 4
    
    # x=0
    y1 = torch.rand(n_face, 1) * config.Ly
    x1 = torch.zeros(n_face, 1)
    e11, e12, t11, t12, r1, mu1, v01 = _sample_param_columns(n_face)
    z1 = torch.rand(n_face, 1) * (t11 + t12)
    p1 = torch.cat([x1, y1, z1, e11, t11, e12, t12, r1, mu1, v01], dim=1)
    
    # x=Lx
    y2 = torch.rand(n_face, 1) * config.Ly
    x2 = torch.ones(n_face, 1) * config.Lx
    e21, e22, t21, t22, r2, mu2, v02 = _sample_param_columns(n_face)
    z2 = torch.rand(n_face, 1) * (t21 + t22)
    p2 = torch.cat([x2, y2, z2, e21, t21, e22, t22, r2, mu2, v02], dim=1)
    
    # y=0
    x3 = torch.rand(n_face, 1) * config.Lx
    y3 = torch.zeros(n_face, 1)
    e31, e32, t31, t32, r3, mu3, v03 = _sample_param_columns(n_face)
    z3 = torch.rand(n_face, 1) * (t31 + t32)
    p3 = torch.cat([x3, y3, z3, e31, t31, e32, t32, r3, mu3, v03], dim=1)
    
    # y=Ly
    x4 = torch.rand(n_face, 1) * config.Lx
    y4 = torch.ones(n_face, 1) * config.Ly
    e41, e42, t41, t42, r4, mu4, v04 = _sample_param_columns(n_face)
    z4 = torch.rand(n_face, 1) * (t41 + t42)
    p4 = torch.cat([x4, y4, z4, e41, t41, e42, t42, r4, mu4, v04], dim=1)
    
    return torch.cat([p1, p2, p3, p4], dim=0)

def sample_boundaries_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    if n <= 0:
        return prev_pts.new_empty((0, prev_pts.shape[1]))
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
    t1_min, t1_max = _get_t1_range()
    t2_min, t2_max = _get_t2_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    noise_e1 = (torch.rand(n) - 0.5) * 2 * noise_scale * (e_max - e_min)
    new_pts[:, 3] += noise_e1
    noise_e2 = (torch.rand(n) - 0.5) * 2 * noise_scale * (e_max - e_min)
    new_pts[:, 5] += noise_e2
    noise_t1 = (torch.rand(n) - 0.5) * 2 * noise_scale * (t1_max - t1_min)
    noise_t2 = (torch.rand(n) - 0.5) * 2 * noise_scale * (t2_max - t2_min)
    new_pts[:, 4] += noise_t1
    new_pts[:, 6] += noise_t2
    noise_r = (torch.rand(n) - 0.5) * 2 * noise_scale * (r_max - r_min)
    new_pts[:, 7] += noise_r
    noise_mu = (torch.rand(n) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    new_pts[:, 8] += noise_mu
    noise_v0 = (torch.rand(n) - 0.5) * 2 * noise_scale * (v0_max - v0_min)
    new_pts[:, 9] += noise_v0
    
    # For each face, perturb only the non-fixed coordinates
    z_span = max(z_max - z_min, 1e-6)
    for i in range(n):
        pt = new_pts[i]
        if torch.abs(pt[0]) < 1e-6:  # x=0 face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * z_span
            new_pts[i, 0] = 0.0
        elif torch.abs(pt[0] - config.Lx) < 1e-6:  # x=Lx face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * z_span
            new_pts[i, 0] = config.Lx
        elif torch.abs(pt[1]) < 1e-6:  # y=0 face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * z_span
            new_pts[i, 1] = 0.0
        elif torch.abs(pt[1] - config.Ly) < 1e-6:  # y=Ly face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * z_span
            new_pts[i, 1] = config.Ly
    
    # Clamp
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], e_min, e_max)
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t1_min, t1_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], t2_min, t2_max)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], min=0.0)
    new_pts[:, 2] = torch.minimum(new_pts[:, 2], new_pts[:, 4] + new_pts[:, 6])
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], r_min, r_max)
    new_pts[:, 8] = torch.clamp(new_pts[:, 8], mu_min, mu_max)
    new_pts[:, 9] = torch.clamp(new_pts[:, 9], v0_min, v0_max)
    
    return new_pts

def sample_top_load(n):
    # Loaded Patch: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    xl = torch.rand(n, 1) * (config.Lx/3) + config.Lx/3
    yl = torch.rand(n, 1) * (config.Ly/3) + config.Ly/3
    bias_frac = float(getattr(config, "PATCH_CENTER_BIAS_FRACTION", 0.0))
    bias_frac = max(0.0, min(bias_frac, 1.0))
    n_bias = int(n * bias_frac)
    if n_bias > 0:
        shape = float(getattr(config, "PATCH_CENTER_BIAS_SHAPE", 2.0))
        alpha = torch.full((n_bias, 1), shape)
        beta = torch.full((n_bias, 1), shape)
        center_dist = torch.distributions.Beta(alpha, beta)
        xl[:n_bias] = center_dist.sample() * (config.Lx/3) + config.Lx/3
        yl[:n_bias] = center_dist.sample() * (config.Ly/3) + config.Ly/3
    e1l, e2l, t1l, t2l, rl, mul, v0l = _sample_param_columns(n)
    zl = (t1l + t2l).clone()
    return torch.cat([xl, yl, zl, e1l, t1l, e2l, t2l, rl, mul, v0l], dim=1)

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
            e1f, e2f, t1f, t2f, rf, muf, v0f = _sample_param_columns(len(xf))
            zf = (t1f + t2f).clone()
            batch_pts = torch.cat([xf, yf, zf, e1f, t1f, e2f, t2f, rf, muf, v0f], dim=1)
            pts_free_list.append(batch_pts)
            count += len(xf)
    
    pts_free = torch.cat(pts_free_list, dim=0)[:n]
    return pts_free

def sample_surface_residual_based(n, z_val, prev_pts, prev_residuals, constrain_load_patch=False, is_load_patch=False):
    if n <= 0:
        return prev_pts.new_empty((0, prev_pts.shape[1]))
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
            e1, e2, t1, t2, r, mu, v0 = _sample_param_columns(n)
            t_total = t1 + t2
            if z_val == 0.0:
                z = torch.zeros(n, 1)
            else:
                z = t_total.clone()
            return torch.cat([x, y, z, e1, t1, e2, t2, r, mu, v0], dim=1)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    e_min, e_max = _get_e_range()
    t1_min, t1_max = _get_t1_range()
    t2_min, t2_max = _get_t2_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()
    noise_e1 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_e2 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_t1 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t1_max - t1_min)
    noise_t2 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (t2_max - t2_min)
    noise_r = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (r_max - r_min)
    noise_mu = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (mu_max - mu_min)
    noise_v0 = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (v0_max - v0_min)
    noise = torch.cat([noise_x, noise_y, torch.zeros(n, 1), noise_e1, noise_t1, noise_e2, noise_t2, noise_r, noise_mu, noise_v0], dim=1)
    
    new_pts = sampled_pts + noise
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t1_min, t1_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], t2_min, t2_max)
    if z_val == 0.0:
        new_pts[:, 2] = 0.0
    else:
        new_pts[:, 2] = new_pts[:, 4] + new_pts[:, 6]
    
    # Clamp to domain
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], e_min, e_max)
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], r_min, r_max)
    new_pts[:, 8] = torch.clamp(new_pts[:, 8], mu_min, mu_max)
    new_pts[:, 9] = torch.clamp(new_pts[:, 9], v0_min, v0_max)
    
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
    e1, e2, t1, t2, r, mu, v0 = _sample_param_columns(n)
    z = t1
    return torch.cat([x, y, z, e1, t1, e2, t2, r, mu, v0], dim=1)

def sample_interface_band(n, z_center, band, z_min=None, z_max=None):
    if z_min is None:
        z_min = 0.0
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    e1, e2, t1, t2, r, mu, v0 = _sample_param_columns(n)
    t_total = t1 + t2
    z_center_local = t1
    z_low_local = torch.clamp(z_center_local - band, min=0.0)
    z_high_local = torch.clamp(z_center_local + band, max=t_total)
    z = torch.rand(n, 1) * torch.clamp(z_high_local - z_low_local, min=1e-6) + z_low_local
    return torch.cat([x, y, z, e1, t1, e2, t2, r, mu, v0], dim=1)

def sample_bottom(n):
    x_bot = torch.rand(n, 1) * config.Lx
    y_bot = torch.rand(n, 1) * config.Ly
    z_bot = torch.zeros(n, 1)
    e1_bot, e2_bot, t1_bot, t2_bot, r_bot, mu_bot, v0_bot = _sample_param_columns(n)
    return torch.cat([x_bot, y_bot, z_bot, e1_bot, t1_bot, e2_bot, t2_bot, r_bot, mu_bot, v0_bot], dim=1)

def get_data(prev_data=None, residuals=None):
    t1_min, t1_max = _get_t1_range()
    t2_min, t2_max = _get_t2_range()
    z_min, z_max = 0.0, float(t1_max + t2_max)
    
    # Decide whether to use residual-based sampling (50% uniform, 50% residual-based)
    use_residual = (prev_data is not None and residuals is not None)
    
    interface_frac = float(getattr(config, "INTERFACE_SAMPLE_FRACTION", 0.0))
    interface_band = float(getattr(config, "INTERFACE_BAND", 0.0))
    interface_frac = max(0.0, min(interface_frac, 0.9))
    n_interface = int(config.N_INTERIOR * interface_frac) if interface_frac > 0.0 else 0
    n_patch = int(config.N_INTERIOR * config.UNDER_PATCH_FRACTION)
    if n_patch < 0:
        n_patch = 0
    if n_patch > (config.N_INTERIOR - n_interface):
        n_patch = max(0, config.N_INTERIOR - n_interface)
    n_interior = config.N_INTERIOR - n_patch - n_interface
    
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
        if n_interface > 0 and interface_band > 0.0:
            interface_pts = sample_interface_band(n_interface, None, interface_band, z_min=z_min, z_max=z_max)
            interior = torch.cat([interior, interface_pts], dim=0)
        
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
            n_residual_load,
            z_max,
            prev_data['top_load'], residuals['top_load'],
            constrain_load_patch=True, is_load_patch=True
        )
        top_load = torch.cat([load_uniform, load_residual], dim=0)
        
        # Top Free: half uniform, half residual-based
        n_uniform_free = config.N_TOP_FREE // 2
        n_residual_free = config.N_TOP_FREE - n_uniform_free
        free_uniform = sample_top_free(n_uniform_free)
        free_residual = sample_surface_residual_based(
            n_residual_free,
            z_max,
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
        if n_interface > 0 and interface_band > 0.0:
            interface_pts = sample_interface_band(n_interface, None, interface_band, z_min=z_min, z_max=z_max)
            interior = torch.cat([interior, interface_pts], dim=0)
        bc_sides = sample_boundaries(config.N_SIDES, z_min, z_max)
        top_load = sample_top_load(config.N_TOP_LOAD)
        top_free = sample_top_free(config.N_TOP_FREE)
        bot_free = sample_bottom(config.N_BOTTOM)

    interface = sample_interface(config.N_INTERFACE, 0.0)
    
    return {
        'interior': [interior],
        'sides': [bc_sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bot_free,
        'interface': interface
    }
