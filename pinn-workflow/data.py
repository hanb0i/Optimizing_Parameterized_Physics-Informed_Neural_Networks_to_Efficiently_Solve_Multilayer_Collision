
import torch
import numpy as np
import pinn_config as config

def sample_domain(n, z_min, z_max):
    # Uniform sampling
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z = torch.rand(n, 1) * (z_max - z_min) + z_min
    return torch.cat([x, y, z], dim=1)

def sample_domain_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    """Sample points weighted by residual magnitude.
    
    Args:
        n: Number of points to sample
        z_min, z_max: Domain bounds in z
        prev_pts: Previous collocation points (N_prev, 3)
        prev_residuals: Residual magnitudes at previous points (N_prev,)
    """
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
    noise_scale = 0.05  # 5% perturbation
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    noise_z = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (z_max - z_min)
    noise = torch.cat([noise_x, noise_y, noise_z], dim=1)
    
    new_pts = sampled_pts + noise
    
    # Clamp to domain bounds
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], z_min, z_max)
    
    return new_pts

def sample_boundaries(n, z_min, z_max):
    # 4 Side faces: x=0, x=Lx, y=0, y=Ly
    # Split n among 4 faces
    n_face = n // 4
    
    # x=0
    y1 = torch.rand(n_face, 1) * config.Ly
    z1 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    x1 = torch.zeros(n_face, 1)
    p1 = torch.cat([x1, y1, z1], dim=1)
    
    # x=Lx
    y2 = torch.rand(n_face, 1) * config.Ly
    z2 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    x2 = torch.ones(n_face, 1) * config.Lx
    p2 = torch.cat([x2, y2, z2], dim=1)
    
    # y=0
    x3 = torch.rand(n_face, 1) * config.Lx
    z3 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    y3 = torch.zeros(n_face, 1)
    p3 = torch.cat([x3, y3, z3], dim=1)
    
    # y=Ly
    x4 = torch.rand(n_face, 1) * config.Lx
    z4 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    y4 = torch.ones(n_face, 1) * config.Ly
    p4 = torch.cat([x4, y4, z4], dim=1)
    
    return torch.cat([p1, p2, p3, p4], dim=0)

def sample_boundaries_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    """Sample boundary points weighted by BC residual."""
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_boundaries(n, z_min, z_max)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = 0.05
    # Keep boundary constraints while perturbing
    new_pts = sampled_pts.clone()
    
    # For each face, perturb only the non-fixed coordinates
    for i in range(n):
        pt = new_pts[i]
        if torch.abs(pt[0]) < 1e-6:  # x=0 face
            new_pts[i, 1] += (torch.rand(1) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand(1) - 0.5) * 2 * noise_scale * (z_max - z_min)
            new_pts[i, 0] = 0.0
        elif torch.abs(pt[0] - config.Lx) < 1e-6:  # x=Lx face
            new_pts[i, 1] += (torch.rand(1) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand(1) - 0.5) * 2 * noise_scale * (z_max - z_min)
            new_pts[i, 0] = config.Lx
        elif torch.abs(pt[1]) < 1e-6:  # y=0 face
            new_pts[i, 0] += (torch.rand(1) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand(1) - 0.5) * 2 * noise_scale * (z_max - z_min)
            new_pts[i, 1] = 0.0
        elif torch.abs(pt[1] - config.Ly) < 1e-6:  # y=Ly face
            new_pts[i, 0] += (torch.rand(1) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand(1) - 0.5) * 2 * noise_scale * (z_max - z_min)
            new_pts[i, 1] = config.Ly
    
    # Clamp
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], z_min, z_max)
    
    return new_pts

def sample_top_load(n):
    """Sample points on load patch only."""
    # Loaded Patch: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    xl = torch.rand(n, 1) * (config.Lx/3) + config.Lx/3
    yl = torch.rand(n, 1) * (config.Ly/3) + config.Ly/3
    zl = torch.ones(n, 1) * config.H
    return torch.cat([xl, yl, zl], dim=1)

def sample_top_free(n):
    """Sample points on free top surface (outside load patch)."""
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
            zf = torch.ones(len(xf), 1) * config.H
            batch_pts = torch.cat([xf, yf, zf], dim=1)
            pts_free_list.append(batch_pts)
            count += len(xf)
    
    pts_free = torch.cat(pts_free_list, dim=0)[:n]
    return pts_free

def sample_surface_residual_based(n, z_val, prev_pts, prev_residuals, constrain_load_patch=False, is_load_patch=False):
    """Sample surface points weighted by traction residual."""
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
            z = torch.ones(n, 1) * z_val
            return torch.cat([x, y, z], dim=1)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = 0.05
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    noise = torch.cat([noise_x, noise_y, torch.zeros(n, 1)], dim=1)
    
    new_pts = sampled_pts + noise
    new_pts[:, 2] = z_val  # Fix z coordinate
    
    # Clamp to domain
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    
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
    # z=H
    # Determine Loaded Patch vs Free
    # Loaded: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    
    # We'll sample uniform and classify, or explicit sample?
    # Better to explicitly sample both sets to ensure coverage
    
    n_load = n // 2
    n_free = n - n_load
    
    pts_load = sample_top_load(n_load)
    pts_free = sample_top_free(n_free)
    
    return pts_load, pts_free

def sample_interface(n, z_val):
    # z = z_val
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z = torch.ones(n, 1) * z_val
    return torch.cat([x, y, z], dim=1)

def sample_bottom(n):
    """Sample points on bottom surface (z=0)."""
    x_bot = torch.rand(n, 1) * config.Lx
    y_bot = torch.rand(n, 1) * config.Ly
    z_bot = torch.zeros(n, 1)
    return torch.cat([x_bot, y_bot, z_bot], dim=1)

def get_data(prev_data=None, residuals=None):
    """Generate collocation points with optional residual-based sampling.
    
    Args:
        prev_data: Previous training data dict (optional)
        residuals: Dict of residuals for each data type (optional)
                  Keys: 'interior', 'sides', 'top_load', 'top_free', 'bottom'
                  Values: Tensor of residual magnitudes
    
    Returns:
        Dictionary of training data
    """
    z_min, z_max = config.Layer_Interfaces[0], config.Layer_Interfaces[1]
    
    # Decide whether to use residual-based sampling (50% uniform, 50% residual-based)
    use_residual = (prev_data is not None and residuals is not None)
    
    if use_residual:
        n_uniform = config.N_INTERIOR // 2
        n_residual = config.N_INTERIOR - n_uniform
        
        # Interior: half uniform, half residual-based
        interior_uniform = sample_domain(n_uniform, z_min, z_max)
        interior_residual = sample_domain_residual_based(
            n_residual, z_min, z_max, 
            prev_data['interior'][0], residuals['interior']
        )
        interior = torch.cat([interior_uniform, interior_residual], dim=0)
        
        # BC Sides: half uniform, half residual-based
        n_uniform_bc = config.N_BOUNDARY // 2
        n_residual_bc = config.N_BOUNDARY - n_uniform_bc
        bc_uniform = sample_boundaries(n_uniform_bc, z_min, z_max)
        bc_residual = sample_boundaries_residual_based(
            n_residual_bc, z_min, z_max,
            prev_data['sides'][0], residuals['sides']
        )
        bc_sides = torch.cat([bc_uniform, bc_residual], dim=0)
        
        # Top Load: half uniform, half residual-based
        n_uniform_load = config.N_BOUNDARY // 2
        n_residual_load = config.N_BOUNDARY - n_uniform_load
        load_uniform = sample_top_load(n_uniform_load)
        load_residual = sample_surface_residual_based(
            n_residual_load, config.H,
            prev_data['top_load'], residuals['top_load'],
            constrain_load_patch=True, is_load_patch=True
        )
        top_load = torch.cat([load_uniform, load_residual], dim=0)
        
        # Top Free: half uniform, half residual-based
        n_uniform_free = config.N_BOUNDARY // 2
        n_residual_free = config.N_BOUNDARY - n_uniform_free
        free_uniform = sample_top_free(n_uniform_free)
        free_residual = sample_surface_residual_based(
            n_residual_free, config.H,
            prev_data['top_free'], residuals['top_free'],
            constrain_load_patch=True, is_load_patch=False
        )
        top_free = torch.cat([free_uniform, free_residual], dim=0)
        
        # Bottom: half uniform, half residual-based
        n_uniform_bot = config.N_BOUNDARY // 2
        n_residual_bot = config.N_BOUNDARY - n_uniform_bot
        bot_uniform = sample_bottom(n_uniform_bot)
        bot_residual = sample_surface_residual_based(
            n_residual_bot, 0.0,
            prev_data['bottom'], residuals['bottom']
        )
        bot_free = torch.cat([bot_uniform, bot_residual], dim=0)
        
    else:
        # Uniform sampling (initial or when no residuals provided)
        interior = sample_domain(config.N_INTERIOR, z_min, z_max)
        bc_sides = sample_boundaries(config.N_BOUNDARY, z_min, z_max)
        top_load = sample_top_load(config.N_BOUNDARY)
        top_free = sample_top_free(config.N_BOUNDARY)
        bot_free = sample_bottom(config.N_BOUNDARY)
    
    return {
        'interior': [interior],
        'sides': [bc_sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bot_free
    }
