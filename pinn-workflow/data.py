
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
    # Stratified sampling for 3-Layer Variable Geometry
    if hasattr(config, "LAYER_Z_RANGES") and hasattr(config, "LAYER_E_VALS"):
        layer_ranges = config.LAYER_Z_RANGES
        layer_evals = config.LAYER_E_VALS
        n_per_layer = n // len(layer_ranges)
        
        parts = []
        for i, (z_start_ref, z_end_ref) in enumerate(layer_ranges):
            # Coordinates
            x = torch.rand(n_per_layer, 1) * config.Lx
            y = torch.rand(n_per_layer, 1) * config.Ly
            
            # Calculate local z_top based on geometry
            # We map the reference z_end (max 0.1) to the actual z_top (max 0.1 - dent)
            # Simple linear scaling: z_actual = z_ref * (z_top(x,y) / H_ref)
            z_top_local = config.get_domain_height(x, y)
            scale = z_top_local / config.H
            
            z_start = z_start_ref * scale
            z_end = z_end_ref * scale
            
            z = torch.rand(n_per_layer, 1) * (z_end - z_start) + z_start
            
            # Material & Geometric Params (Fixed for Phase 5)
            # E is assigned by layer index
            e = torch.ones(n_per_layer, 1) * layer_evals[i]
            # Thickness is variable now
            t = z_top_local
            
            r_min, r_max = _get_restitution_range()
            restitution = torch.rand(n_per_layer, 1) * (r_max - r_min) + r_min
            mu_min, mu_max = _get_friction_range()
            friction = torch.rand(n_per_layer, 1) * (mu_max - mu_min) + mu_min
            v0_min, v0_max = _get_impact_velocity_range()
            impact_velocity = torch.rand(n_per_layer, 1) * (v0_max - v0_min) + v0_min
            
            parts.append(torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1))
        
        # Remainder (Core)
        n_rem = n - (n_per_layer * len(layer_ranges))
        if n_rem > 0:
            x = torch.rand(n_rem, 1) * config.Lx
            y = torch.rand(n_rem, 1) * config.Ly
            z_top_local = config.get_domain_height(x, y)
            scale = z_top_local / config.H
            
            z_start = layer_ranges[1][0] * scale
            z_end = layer_ranges[1][1] * scale
            
            z = torch.rand(n_rem, 1) * (z_end - z_start) + z_start
            e = torch.ones(n_rem, 1) * layer_evals[1]
            t = z_top_local
            parts.append(torch.cat([x, y, z, e, t, 
                                    torch.rand(n_rem, 1)*(r_max-r_min)+r_min, 
                                    torch.rand(n_rem, 1)*(mu_max-mu_min)+mu_min,
                                    torch.rand(n_rem, 1)*(v0_max-v0_min)+v0_min], dim=1))
            
        return torch.cat(parts, dim=0)

    # Standard Uniform sampling (fallback)
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z_max_local = config.get_domain_height(x, y)
    t_min, t_max = _get_thickness_range()
    
    # If parametric thickness is enabled, we sample it. Otherwise use geometry.
    # Logic pivot: In Geometry Phase, 't' is the actual thickness at (x,y)
    t = z_max_local
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

def sample_domain_under_patch(n, z_min, z_max):
    # Stratified sampling for 3-Layer Geometry under patch
    if hasattr(config, "LAYER_Z_RANGES") and hasattr(config, "LAYER_E_VALS"):
        layer_ranges = config.LAYER_Z_RANGES
        layer_evals = config.LAYER_E_VALS
        n_per_layer = n // len(layer_ranges)
        
        parts = []
        x_min, x_max = config.LOAD_PATCH_X
        y_min, y_max = config.LOAD_PATCH_Y
        
        for i, (z_start_ref, z_end_ref) in enumerate(layer_ranges):
            x = torch.rand(n_per_layer, 1) * (x_max - x_min) + x_min
            y = torch.rand(n_per_layer, 1) * (y_max - y_min) + y_min
            
            z_top_local = config.get_domain_height(x, y)
            scale = z_top_local / config.H
            z_start = z_start_ref * scale
            z_end = z_end_ref * scale
            
            z = torch.rand(n_per_layer, 1) * (z_end - z_start) + z_start
            
            e = torch.ones(n_per_layer, 1) * layer_evals[i]
            t = z_top_local
            
            r_min, r_max = _get_restitution_range()
            restitution = torch.rand(n_per_layer, 1) * (r_max - r_min) + r_min
            mu_min, mu_max = _get_friction_range()
            friction = torch.rand(n_per_layer, 1) * (mu_max - mu_min) + mu_min
            v0_min, v0_max = _get_impact_velocity_range()
            impact_velocity = torch.rand(n_per_layer, 1) * (v0_max - v0_min) + v0_min
            
            parts.append(torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1))
        
        # Remainder
        n_rem = n - (n_per_layer * len(layer_ranges))
        if n_rem > 0:
            x = torch.rand(n_rem, 1) * (x_max - x_min) + x_min
            y = torch.rand(n_rem, 1) * (y_max - y_min) + y_min
            
            z_top_local = config.get_domain_height(x, y)
            scale = z_top_local / config.H
            z_start = layer_ranges[1][0] * scale # Core
            z_end = layer_ranges[1][1] * scale
            
            z = torch.rand(n_rem, 1) * (z_end - z_start) + z_start
            e = torch.ones(n_rem, 1) * layer_evals[1]
            t = z_top_local
            parts.append(torch.cat([x, y, z, e, t, 
                                    torch.rand(n_rem, 1)*(r_max-r_min)+r_min, 
                                    torch.rand(n_rem, 1)*(mu_max-mu_min)+mu_min,
                                    torch.rand(n_rem, 1)*(v0_max-v0_min)+v0_min], dim=1))
            
        return torch.cat(parts, dim=0)

    # Standard
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    x = torch.rand(n, 1) * (x_max - x_min) + x_min
    y = torch.rand(n, 1) * (y_max - y_min) + y_min
    z_max_local = config.get_domain_height(x, y)
    t = z_max_local
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
    # Stratified sampling for 3-Layer Geometry
    if hasattr(config, "LAYER_Z_RANGES") and hasattr(config, "LAYER_E_VALS"):
        layer_ranges = config.LAYER_Z_RANGES
        layer_evals = config.LAYER_E_VALS
        
        # Split n among 4 faces
        n_face = n // 4
        parts = []
        
        # Helper to generating stratified points for a generic face
        # face_type: 0 (x=0), 1 (x=Lx), 2 (y=0), 3 (y=Ly)
        def sample_face_stratified(n_f, face_type):
            n_layer = n_f // len(layer_ranges)
            face_parts = []
            
            for i, (z_start_ref, z_end_ref) in enumerate(layer_ranges):
                # Common coordinates
                if face_type == 0 or face_type == 1: # x-faces
                    dim1 = torch.rand(n_layer, 1) * config.Ly # y
                    # x is fixed
                else: # y-faces
                    dim1 = torch.rand(n_layer, 1) * config.Lx # x
                    # y is fixed
                
                # Z-scaling logic
                # We need x,y to compute height.
                if face_type == 0: # x=0
                    x = torch.zeros(n_layer, 1)
                    y = dim1
                elif face_type == 1: # x=Lx
                    x = torch.ones(n_layer, 1) * config.Lx
                    y = dim1
                elif face_type == 2: # y=0
                    x = dim1
                    y = torch.zeros(n_layer, 1)
                elif face_type == 3: # y=Ly
                    x = dim1
                    y = torch.ones(n_layer, 1) * config.Ly
                
                z_top_local = config.get_domain_height(x, y)
                scale = z_top_local / config.H
                z_start = z_start_ref * scale
                z_end = z_end_ref * scale
                
                z = torch.rand(n_layer, 1) * (z_end - z_start) + z_start
                e = torch.ones(n_layer, 1) * layer_evals[i]
                t = z_top_local
                
                r_min, r_max = _get_restitution_range()
                restitution = torch.rand(n_layer, 1) * (r_max - r_min) + r_min
                mu_min, mu_max = _get_friction_range()
                friction = torch.rand(n_layer, 1) * (mu_max - mu_min) + mu_min
                v0_min, v0_max = _get_impact_velocity_range()
                impact_velocity = torch.rand(n_layer, 1) * (v0_max - v0_min) + v0_min
                
                face_parts.append(torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1))
                    
            return torch.cat(face_parts, dim=0)

        parts.append(sample_face_stratified(n_face, 0))
        parts.append(sample_face_stratified(n_face, 1))
        parts.append(sample_face_stratified(n_face, 2))
        parts.append(sample_face_stratified(n_face, 3))
        
        return torch.cat(parts, dim=0)

    # Standard
    n_face = n // 4
    
    # Helper for fallback
    def get_standard_face(face_type, n_f):
        if face_type == 0: # x=0
            x = torch.zeros(n_f, 1)
            y = torch.rand(n_f, 1) * config.Ly
        elif face_type == 1: # x=Lx
            x = torch.ones(n_f, 1) * config.Lx
            y = torch.rand(n_f, 1) * config.Ly
        elif face_type == 2: # y=0
            x = torch.rand(n_f, 1) * config.Lx
            y = torch.zeros(n_f, 1)
        elif face_type == 3: # y=Ly
            x = torch.rand(n_f, 1) * config.Lx
            y = torch.ones(n_f, 1) * config.Ly
            
        z_top_local = config.get_domain_height(x, y)
        t = z_top_local
        z = torch.rand(n_f, 1) * t
        
        e_min, e_max = _get_e_range()
        e = torch.rand(n_f, 1) * (e_max - e_min) + e_min
        r_min, r_max = _get_restitution_range()
        restitution = torch.rand(n_f, 1) * (r_max - r_min) + r_min
        mu_min, mu_max = _get_friction_range()
        friction = torch.rand(n_f, 1) * (mu_max - mu_min) + mu_min
        v0_min, v0_max = _get_impact_velocity_range()
        impact_velocity = torch.rand(n_f, 1) * (v0_max - v0_min) + v0_min
        return torch.cat([x, y, z, e, t, restitution, friction, impact_velocity], dim=1)

    p1 = get_standard_face(0, n_face)
    p2 = get_standard_face(1, n_face)
    p3 = get_standard_face(2, n_face)
    p4 = get_standard_face(3, n_face)
    
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
    z_min, z_max = 0.0, config.H
    
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
