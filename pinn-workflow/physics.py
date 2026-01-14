import torch
import torch.autograd as autograd
import pinn_config as config

# ==========================================
# HELPER FUNCTIONS (MASKS)
# ==========================================

def load_mask(x):
    """
    Soft edge mask for load patch using quadratic function.
    Similar to side BC mask: M(x,y) = 16*x(1-x)*y(1-y)
    but applied to normalized coordinates within the load patch.
    
    For x,y ∈ [1/3, 2/3], normalize to [0,1] and apply quadratic.
    Outside the patch, mask = 0.
    
    Args:
        x: (N, 3) tensor of coordinates
    Returns:
        mask: (N,) tensor of mask values ∈ [0, 1]
    """
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    
    # Normalize coordinates to [0, 1] within patch
    x_norm = (x_coord - x_min) / (x_max - x_min)
    y_norm = (y_coord - y_min) / (y_max - y_min)
    
    # Quadratic falloff: M(x,y) = 16*x(1-x)*y(1-y)
    # This gives M=1 at center (0.5, 0.5) and M=0 at edges
    mask = 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)
    
    # Clamp to ensure mask is only non-zero within patch
    mask = torch.where((x_coord >= x_min) & (x_coord <= x_max) & 
                       (y_coord >= y_min) & (y_coord <= y_max),
                       mask, torch.zeros_like(mask))
    
    return mask

# ==========================================
# PHYSICS EQUATIONS (LINEAR ELASTICITY)
# ==========================================

def gradient(u, x):
    """
    Computes the spatial gradient of displacement u using automatic differentiation.
    
    Args:
        u: Displacement field (N, 3)
        x: Spatial coordinates (N, 3)
        
    Returns:
        grad_u: Displacement gradient tensor (N, 3, 3)
                grad_u[i, j, k] = d(u_j)/d(x_k) for the i-th point
    """
    # u: (N, 3), x: (N, 3)
    # Returns du/dx: (N, 3, 3)
    # [ [dux/dx, dux/dy, dux/dz],
    #   [duy/dx, duy/dy, duy/dz],
    #   [duz/dx, duz/dy, duz/dz] ]
    
    grad_u = torch.zeros(x.shape[0], 3, 3, device=x.device)
    
    for i in range(3): # u_x, u_y, u_z
        u_i = u[:, i].unsqueeze(1)
        grad_i = autograd.grad(
            u_i, x, 
            grad_outputs=torch.ones_like(u_i),
            create_graph=True, 
            retain_graph=True
        )[0]
        grad_u[:, i, :] = grad_i
        
    return grad_u

def strain(grad_u):
    """
    Computes the infinitesimal strain tensor (linearized strain).
    epsilon = 0.5 * (grad_u + grad_u^T)
    """
    # epsilon = 0.5 * (grad_u + grad_u^T)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    """
    Computes the stress tensor using Hooke's Law for isotropic linear elasticity.
    sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
    """
    # sigma = lambda * tr(eps) * I + 2 * mu * eps
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    
    sigma = lm * trace_eps * eye + 2 * mu * eps
    return sigma

def divergence(sigma, x):
    """
    Computes the divergence of the stress tensor.
    div(sigma) represents the internal force density.
    Detailed expansion: d(sigma_ij)/dx_j
    """
    # sigma: (N, 3, 3), x: (N, 3)
    # div_sigma: (N, 3) vector
    # We need d(sigma_ij)/dx_j
    
    div = torch.zeros(x.shape[0], 3, device=x.device)
    
    # Row 0: d(sig_xx)/dx + d(sig_xy)/dy + d(sig_xz)/dz
    # etc.
    
    for i in range(3): # For each component of force equilibrium
        # We need d(sigma_i0)/dx + d(sigma_i1)/dy + d(sigma_i2)/dz
        div_i = 0
        for j in range(3):
            sig_ij = sigma[:, i, j].unsqueeze(1)
            grad_sig_ij = autograd.grad(
                sig_ij, x,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=True,
                retain_graph=True
            )[0]
            div_i += grad_sig_ij[:, j]
        div[:, i] = div_i
        
    return div

# ==========================================
# LOSS FUNCTION
# ==========================================

def compute_loss(model, data, device, weights=None):
    """
    Computes the total PINN loss, which is a weighted sum of:
    1. PDE Residuals (Equilibrium equations)
    2. Boundary Conditions (Hard constraints handle sides, this handles traction)
    3. Loading Conditions
    """
    if weights is None:
        weights = config.WEIGHTS
    pde_weight = weights.get('pde', config.WEIGHTS['pde'])
    bc_weight = weights.get('bc', config.WEIGHTS.get('bc', 1.0))
    load_weight = weights.get('load', config.WEIGHTS.get('load', 1.0))

    total_loss = 0
    losses = {}
    
    # --- 1. PDE Residuals (Interior) ---
    # Governing Equation: div(sigma) + BodyForce = 0
    # Assuming no body force -> div(sigma) = 0
    x_int = data['interior'][0].to(device)
    x_int.requires_grad = True
    
    lm, mu = config.Lame_Params[0]
    
    u = model(x_int, 0)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    # Equilibrium: -div(sigma) = 0
    residual = -div_sigma
    
    pde_loss = torch.mean(residual**2)
    losses['pde'] = pde_loss
    total_loss += pde_weight * pde_loss
    
    # --- 2. Dirichlet BCs (Clamped Sides) ---
    # While hard constraints handle this, we often include a soft loss 
    # as a "sanity check" or to reinforce zero values, though it should be zero by construction.
    x_side = data['sides'][0].to(device)
    u_side = model(x_side, 0)
    # u = 0
    bc_loss = torch.mean(u_side**2)
        
    losses['bc_sides'] = bc_loss
    total_loss += bc_weight * bc_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded Surface
    x_top_load = data['top_load'].to(device)
    x_top_load.requires_grad = True
    
    lm, mu = config.Lame_Params[0] # Single layer
    u_top = model(x_top_load, 0)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    
    # Traction T = sigma · n
    # For Top surface, n = (0, 0, 1)
    # T_z = sigma_33, T_x = sigma_13 ...
    T = sig_top[:, :, 2] 
    
    # Apply soft edge mask to target load
    mask = load_mask(x_top_load).unsqueeze(1)  # (N, 1)
    # Target: (0, 0, -p0 * mask) - load smoothly transitions to zero at edges
    target_load = -config.p0 * mask
    target = torch.cat([torch.zeros_like(target_load), 
                       torch.zeros_like(target_load), 
                       target_load], dim=1)
    
    loss_load = torch.mean((T - target)**2)
    losses['load'] = loss_load
    total_loss += load_weight * loss_load
    
    # Top Free Surface (No Load)
    # Traction should be zero
    x_top_free = data['top_free'].to(device)
    x_top_free.requires_grad = True
    u_top_free = model(x_top_free, 0)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm, mu)
    T_free = sig_top_free[:, :, 2]
    # Traction on top free surface (n = [0,0,1])
    
    loss_free = torch.mean(T_free**2)
    losses['free_top'] = loss_free
    total_loss += bc_weight * loss_free # Use BC weight
    
    # Bottom Free Surface
    # Traction should be zero. n = (0, 0, -1)
    x_bot = data['bottom'].to(device)
    x_bot.requires_grad = True
    
    u_bot = model(x_bot, 0)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm, mu)
    
    # T = sigma · n = -sigma[:,:,2]
    T_bot = -sig_bot[:, :, 2]
    
    loss_bot = torch.mean(T_bot**2)
    losses['free_bot'] = loss_bot
    total_loss += bc_weight * loss_bot
    
    # No interface continuity for single layer
    
    losses['total'] = total_loss
    return total_loss, losses

def compute_residuals(model, data, device):
    """Compute residual magnitudes for adaptive sampling.
    
    Returns:
        Dictionary of residual magnitudes for each data type
    """
    residuals = {}

    # --- PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device)
    x_int.requires_grad = True
    
    lm, mu = config.Lame_Params[0]
    
    u = model(x_int, 0)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    residuals['interior'] = residual_mag.cpu()
    
    # --- BC Sides Residuals ---
    x_side = data['sides'][0].to(device)
    u_side = model(x_side, 0)
    bc_residual = torch.sqrt(torch.sum(u_side**2, dim=1))
    residuals['sides'] = bc_residual.cpu()
    
    # --- Top Load Residuals ---
    x_top_load = data['top_load'].to(device)
    x_top_load.requires_grad = True
    u_top = model(x_top_load, 0)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    T = sig_top[:, :, 2]
    target = torch.tensor([0.0, 0.0, -config.p0], device=device).repeat(x_top_load.shape[0], 1)
    load_residual = torch.sqrt(torch.sum((T - target)**2, dim=1))
    residuals['top_load'] = load_residual.cpu()
    
    # --- Top Free Residuals ---
    x_top_free = data['top_free'].to(device)
    x_top_free.requires_grad = True
    u_top_free = model(x_top_free, 0)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm, mu)
    T_free = sig_top_free[:, :, 2]
    free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
    residuals['top_free'] = free_residual.cpu()
    
    # --- Bottom Residuals ---
    x_bot = data['bottom'].to(device)
    x_bot.requires_grad = True
    u_bot = model(x_bot, 0)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm, mu)
    T_bot = -sig_bot[:, :, 2]
    bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
    residuals['bottom'] = bot_residual.cpu()

    return residuals
