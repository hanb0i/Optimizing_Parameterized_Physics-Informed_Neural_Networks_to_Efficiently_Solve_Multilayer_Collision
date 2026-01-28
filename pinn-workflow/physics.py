
import torch
import torch.autograd as autograd
import pinn_config as config

def load_mask(x):
    """Binary mask for load patch (1 inside, 0 outside)."""
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    
    in_patch = (
        (x_coord >= x_min)
        & (x_coord <= x_max)
        & (y_coord >= y_min)
        & (y_coord <= y_max)
    )
    mask = torch.where(in_patch, torch.ones_like(x_coord), torch.zeros_like(x_coord))
    
    return mask

def gradient(u, x):
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
        # Extract only spatial gradients (first 3 columns: dx, dy, dz)
        grad_u[:, i, :] = grad_i[:, :3]
        
    return grad_u

def strain(grad_u):
    # epsilon = 0.5 * (grad_u + grad_u^T)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    # sigma = lambda * tr(eps) * I + 2 * mu * eps
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    
    sigma = lm * trace_eps * eye + 2 * mu * eps
    return sigma

def divergence(sigma, x):
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

def compute_loss(model, data, device, weights=None):
    total_loss = 0
    losses = {}
    if weights is None:
        weights = config.WEIGHTS
    
    # --- 1. PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device)
    x_int.requires_grad = True
    
    # Dynamic material properties
    E_local = x_int[:, 3:4]
    nu = config.nu_vals[0]
    lm = (E_local * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    u = model(x_int, 0)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    # Equilibrium: -div(sigma) = 0 (scale to stress units)
    residual = -div_sigma * config.PDE_LENGTH_SCALE
    
    pde_loss = torch.mean(residual**2)
    losses['pde'] = pde_loss
    total_loss += weights['pde'] * pde_loss
    
    # Internal strain energy (volume integral, approximated by mean * volume)
    energy_density = 0.5 * torch.einsum('bij,bij->b', eps, sig)
    internal_energy = energy_density.mean() * (config.Lx * config.Ly * config.H)
    
    # --- 2. Dirichlet BCs (Clamped Sides) ---
    x_side = data['sides'][0].to(device)
    u_side = model(x_side, 0)
    bc_loss = torch.mean(u_side**2)
    losses['bc_sides'] = bc_loss
    total_loss += weights['bc'] * bc_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded
    x_top_load = data['top_load'].to(device)
    x_top_load.requires_grad = True
    
    E_local_load = x_top_load[:, 3:4]
    lm = (E_local_load * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local_load / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    u_top = model(x_top_load, 0)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    
    T = sig_top[:, :, 2]
    
    mask = load_mask(x_top_load).unsqueeze(1)  # (N, 1)
    target_load = -config.p0 * mask
    target = torch.cat([
        torch.zeros_like(target_load),
        torch.zeros_like(target_load),
        target_load
    ], dim=1)
    
    loss_load = torch.mean((T - target)**2)
    losses['load'] = loss_load
    total_loss += weights['load'] * loss_load
    
    patch_area = (config.LOAD_PATCH_X[1] - config.LOAD_PATCH_X[0]) * (
        config.LOAD_PATCH_Y[1] - config.LOAD_PATCH_Y[0]
    )
    external_work = (-config.p0 * u_top[:, 2:3] * mask).mean() * patch_area
    energy_loss = internal_energy - external_work
    losses['energy'] = energy_loss
    total_loss += weights['energy'] * energy_loss
    
    # Top Free
    x_top_free = data['top_free'].to(device)
    x_top_free.requires_grad = True
    
    E_local_free = x_top_free[:, 3:4]
    lm_free = (E_local_free * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_free = E_local_free / (2 * (1 + nu))
    lm_free = lm_free.unsqueeze(2)
    mu_free = mu_free.unsqueeze(2)
    
    u_top_free = model(x_top_free, 0)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    
    loss_free = torch.mean(T_free**2)
    losses['free_top'] = loss_free
    total_loss += weights['bc'] * loss_free
    
    # Bottom Free
    x_bot = data['bottom'].to(device)
    x_bot.requires_grad = True
    
    E_local_bot = x_bot[:, 3:4]
    lm_bot = (E_local_bot * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_bot = E_local_bot / (2 * (1 + nu))
    lm_bot = lm_bot.unsqueeze(2)
    mu_bot = mu_bot.unsqueeze(2)
    
    u_bot = model(x_bot, 0)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    
    T_bot = -sig_bot[:, :, 2]
    loss_bot = torch.mean(T_bot**2)
    losses['free_bot'] = loss_bot
    total_loss += weights['bc'] * loss_bot
    
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
    
    E_local = x_int[:, 3:4]
    nu = config.nu_vals[0]
    lm = (E_local * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    u = model(x_int, 0)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma * config.PDE_LENGTH_SCALE
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
    
    E_local_load = x_top_load[:, 3:4]
    lm = (E_local_load * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local_load / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    u_top = model(x_top_load, 0)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    T = sig_top[:, :, 2]
    mask = load_mask(x_top_load).unsqueeze(1)
    target_load = -config.p0 * mask
    target = torch.cat([
        torch.zeros_like(target_load),
        torch.zeros_like(target_load),
        target_load,
    ], dim=1)
    load_residual = torch.sqrt(torch.sum((T - target) ** 2, dim=1))
    residuals['top_load'] = load_residual.cpu()
    
    # --- Top Free Residuals ---
    x_top_free = data['top_free'].to(device)
    x_top_free.requires_grad = True
    
    E_local_free = x_top_free[:, 3:4]
    lm_free = (E_local_free * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_free = E_local_free / (2 * (1 + nu))
    lm_free = lm_free.unsqueeze(2)
    mu_free = mu_free.unsqueeze(2)
    
    u_top_free = model(x_top_free, 0)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
    residuals['top_free'] = free_residual.cpu()
    
    # --- Bottom Residuals ---
    x_bot = data['bottom'].to(device)
    x_bot.requires_grad = True
    
    E_local_bot = x_bot[:, 3:4]
    lm_bot = (E_local_bot * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_bot = E_local_bot / (2 * (1 + nu))
    lm_bot = lm_bot.unsqueeze(2)
    mu_bot = mu_bot.unsqueeze(2)
    
    u_bot = model(x_bot, 0)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    T_bot = -sig_bot[:, :, 2]
    bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
    residuals['bottom'] = bot_residual.cpu()

    return residuals
