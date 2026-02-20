
import torch
import torch.autograd as autograd
import pinn_config as config

def compliance_scale(E, t):
    e_safe = torch.clamp(E, min=1e-8)
    t_safe = torch.clamp(t, min=1e-8)
    h_ref = float(getattr(config, "H", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    if alpha == 0.0:
        return 1.0 / (e_safe ** e_pow)
    return (1.0 / (e_safe ** e_pow)) * (h_ref / t_safe) ** alpha

def v_to_u(v, E, t):
    return v * compliance_scale(E, t)

def load_mask(x):
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

def get_material_properties(x):
    z = x[:, 2:3]
    if hasattr(config, "LAYER_Z_RATIOS"):
        # Geometry-aware Layer Assignment
        x_coords = x[:, 0:1]
        y_coords = x[:, 1:2]
        z_top = config.get_domain_height(x_coords, y_coords)
        
        # Avoid division by zero (shouldn't happen with H=0.1)
        z_rel = z / (z_top + 1e-8)
        
        # Initialize with Core properties
        dims = z.shape
        E = torch.ones(dims, device=x.device) * config.LAYER_E_VALS[1]
        nu = torch.ones(dims, device=x.device) * config.LAYER_NU_VALS[1]
        
        # Bottom Layer (0.0 to ratio[0])
        mask_bot = (z_rel <= config.LAYER_Z_RATIOS[0] + 1e-4) # Be generous at interfaces
        E[mask_bot] = config.LAYER_E_VALS[0]
        nu[mask_bot] = config.LAYER_NU_VALS[0]
        
        # Top Layer (ratio[1] to 1.0)
        mask_top = (z_rel >= config.LAYER_Z_RATIOS[1] - 1e-4)
        E[mask_top] = config.LAYER_E_VALS[2]
        nu[mask_top] = config.LAYER_NU_VALS[2]
        
        return E, nu

    elif hasattr(config, "LAYER_Z_RANGES"):
        # Fallback to Fixed Geometry (Phase 5)
        dims = z.shape
        E = torch.ones(dims, device=x.device) * config.LAYER_E_VALS[1]
        nu = torch.ones(dims, device=x.device) * config.LAYER_NU_VALS[1]
        
        for i, (z_start, z_end) in enumerate(config.LAYER_Z_RANGES):
            mask = (z >= z_start - 1e-6) & (z <= z_end + 1e-6)
            E[mask] = config.LAYER_E_VALS[i]
            nu[mask] = config.LAYER_NU_VALS[i]
        return E, nu
    
    # Fallback to parametric input
    E = x[:, 3:4]
    nu = torch.full_like(E, config.nu_vals[0])
    return E, nu

def compute_loss(model, data, device, weights=None):
    total_loss = 0
    losses = {}
    if weights is None:
        weights = config.WEIGHTS
    
    # --- 1. PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    
    # Dynamic material properties
    E_local, nu_local = get_material_properties(x_int)
    t_local = x_int[:, 4:5]
    
    lm = (E_local * nu_local) / ((1 + nu_local) * (1 - 2 * nu_local))
    mu = E_local / (2 * (1 + nu_local))
    # lm, mu are already (N, 1) from the get_material_properties return
    lm = lm.unsqueeze(2) # (N, 1, 1) for stress
    mu = mu.unsqueeze(2)
    
    v_int = model(x_int, 0)

    if getattr(config, "ENFORCE_IMPACT_INVARIANCE", False):
        # Neutral-parameter mode: keep restitution/friction effect suppressed.
        x_int_variant = x_int.clone()
        r_min, r_max = getattr(config, "RESTITUTION_RANGE", (0.5, 0.5))
        mu_min, mu_max = getattr(config, "FRICTION_RANGE", (0.3, 0.3))
        v0_min, v0_max = getattr(config, "IMPACT_VELOCITY_RANGE", (1.0, 1.0))
        if r_max > r_min:
            x_int_variant[:, 5:6] = torch.rand_like(x_int_variant[:, 5:6]) * (r_max - r_min) + r_min
        if mu_max > mu_min:
            x_int_variant[:, 6:7] = torch.rand_like(x_int_variant[:, 6:7]) * (mu_max - mu_min) + mu_min
        if v0_max > v0_min:
            x_int_variant[:, 7:8] = torch.rand_like(x_int_variant[:, 7:8]) * (v0_max - v0_min) + v0_min
        v_int_variant = model(x_int_variant, 0)
        impact_invariance_loss = torch.mean((v_int - v_int_variant) ** 2)
    else:
        impact_invariance_loss = torch.zeros((), device=x_int.device)
    losses['impact_invariance'] = impact_invariance_loss
    total_loss += weights.get('impact_invariance', 0.0) * impact_invariance_loss

    # Predict displacement u = v / E to handle parameter range.
    # NOTE: Thickness compliance scaling (H/t)^alpha is applied at evaluation/plot time,
    # not inside the PDE/traction losses, because the traction BC can otherwise cancel it.
    u = v_int
    
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    # Equilibrium: -div(sigma) = 0 (scale to stress units)
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    
    pde_loss = torch.mean(residual**2)
    losses['pde'] = pde_loss
    total_loss += weights['pde'] * pde_loss
    
    # Internal strain energy (volume integral, approximated by mean * volume)
    energy_density = 0.5 * torch.einsum('bij,bij->b', eps, sig)
    internal_energy = energy_density.mean() * (config.Lx * config.Ly * t_local.mean())
    
    # --- 2. Dirichlet BCs (Clamped Sides) ---
    x_side = data['sides'][0].to(device)
    E_side, _ = get_material_properties(x_side)
    v_side = model(x_side, 0)
    u_side = v_side
    bc_loss = torch.mean(u_side**2)
    losses['bc_sides'] = bc_loss
    total_loss += weights['bc'] * bc_loss
    
    # --- Helper: Calculate Surface Normal ---
    def calc_normal(x_points):
        # n = (-dz/dx, -dz/dy, 1) / norm
        # We need gradients of z_surface w.r.t x,y.
        # Since z_surface is analytic in config, we can use autograd on config.get_domain_height
        
        # x_points has requires_grad=True
        x_coords = x_points[:, 0:1]
        y_coords = x_points[:, 1:2]
        
        # We need to compute gradients of z_top w.r.t x and y
        # But get_domain_height uses returns a tensor. 
        # To use autograd, we need to ensure the graph is connected.
        # Re-implement small graph-friendly version or rely on finite diff/analytic if needed.
        # Ideally, config.get_domain_height is written in torch, so we can just differentiate it.
        
        z_top = config.get_domain_height(x_coords, y_coords)
        
        grad_z = torch.autograd.grad(z_top, [x_coords, y_coords], 
                                     grad_outputs=torch.ones_like(z_top),
                                     create_graph=True, retain_graph=True)
        dz_dx = grad_z[0]
        dz_dy = grad_z[1]
        
        # Normal vector n = (-dz/dx, -dz/dy, 1)
        n = torch.cat([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
        n = n / torch.norm(n, dim=1, keepdim=True)
        return n

    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load, nu_local_load = get_material_properties(x_top_load)
    lm = (E_local_load * nu_local_load) / ((1 + nu_local_load) * (1 - 2 * nu_local_load))
    mu = E_local_load / (2 * (1 + nu_local_load))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_top = model(x_top_load, 0)
    u_top = v_top
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    
    # Traction vector T = sigma . n
    n_top = calc_normal(x_top_load)
    T = torch.einsum('bij,bj->bi', sig_top, n_top)
    
    mask = load_mask(x_top_load).unsqueeze(1)  # (N, 1)
    # Target Load vector is -p0 * n (pressure acts normal to surface)
    target_load = -config.p0 * mask * n_top
    
    # ... (Impact/Friction logic omitted for brevity in this phase, assuming simple pressure first) ...
    # Standard Load Loss
    loss_load = torch.mean((T - target_load)**2)
    losses['load'] = loss_load
    total_loss += weights['load'] * loss_load
    
    losses['energy'] = torch.tensor(0.0).to(device) # Placeholder
    
    # Top Free
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    
    E_local_free, nu_local_free = get_material_properties(x_top_free)
    lm_free = (E_local_free * nu_local_free) / ((1 + nu_local_free) * (1 - 2 * nu_local_free))
    mu_free = E_local_free / (2 * (1 + nu_local_free))
    lm_free = lm_free.unsqueeze(2)
    mu_free = mu_free.unsqueeze(2)
    
    v_top_free = model(x_top_free, 0)
    u_top_free = v_top_free
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    
    n_free = calc_normal(x_top_free)
    T_free = torch.einsum('bij,bj->bi', sig_top_free, n_free)
    
    loss_free = torch.mean(T_free**2)
    losses['free_top'] = loss_free
    total_loss += weights['bc'] * loss_free
    
    # Bottom Free (Flat at z=0, n=[0,0,-1])
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    
    E_local_bot, nu_local_bot = get_material_properties(x_bot)
    lm_bot = (E_local_bot * nu_local_bot) / ((1 + nu_local_bot) * (1 - 2 * nu_local_bot))
    mu_bot = E_local_bot / (2 * (1 + nu_local_bot))
    lm_bot = lm_bot.unsqueeze(2)
    mu_bot = mu_bot.unsqueeze(2)
    
    v_bot = model(x_bot, 0)
    u_bot = v_bot
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    
    # Normal is [0, 0, -1]
    T_bot = -sig_bot[:, :, 2] 
    loss_bot = torch.mean(T_bot**2)
    losses['free_bot'] = loss_bot
    total_loss += weights['bc'] * loss_bot
    
    # --- 4. Supervised Data Loss (Hybrid/Parametric) ---
    if 'x_data' in data and 'u_data' in data:
        x_data = data['x_data'].to(device)
        u_data = data['u_data'].to(device)
        
        # Predict v directly (Stress Potential)
        v_pred = model(x_data, 0)
        E_data = x_data[:, 3:4]
        
        # Ground truth mapping: v_target = u_data * E.
        v_target = u_data * E_data
        
        loss_data = torch.mean((v_pred - v_target)**2)
        losses['data'] = loss_data
        
        # specific weight for data or default high weight
        w_data = weights.get('data', 1.0) 
        total_loss += w_data * loss_data
    
    losses['total'] = total_loss
    return total_loss, losses

def compute_residuals(model, data, device):
    residuals = {}

    # --- PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    
    E_local, nu_local = get_material_properties(x_int)
    t_local = x_int[:, 4:5]
    
    lm = (E_local * nu_local) / ((1 + nu_local) * (1 - 2 * nu_local))
    mu = E_local / (2 * (1 + nu_local))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_int = model(x_int, 0)
    u = v_int
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    residuals['interior'] = residual_mag.cpu()
    
    # --- BC Sides Residuals ---
    x_side = data['sides'][0].to(device)
    E_side, _ = get_material_properties(x_side)
    v_side = model(x_side, 0)
    u_side = v_side
    bc_residual = torch.sqrt(torch.sum(u_side**2, dim=1))
    residuals['sides'] = bc_residual.cpu()
    
    # --- Top Load Residuals ---
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load, nu_local_load = get_material_properties(x_top_load)
    lm = (E_local_load * nu_local_load) / ((1 + nu_local_load) * (1 - 2 * nu_local_load))
    mu = E_local_load / (2 * (1 + nu_local_load))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_top = model(x_top_load, 0)
    u_top = v_top
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
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    
    E_local_free, nu_local_free = get_material_properties(x_top_free)
    lm_free = (E_local_free * nu_local_free) / ((1 + nu_local_free) * (1 - 2 * nu_local_free))
    mu_free = E_local_free / (2 * (1 + nu_local_free))
    lm_free = lm_free.unsqueeze(2)
    mu_free = mu_free.unsqueeze(2)
    
    v_top_free = model(x_top_free, 0)
    u_top_free = v_top_free
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
    residuals['top_free'] = free_residual.cpu()
    
    # --- Bottom Residuals ---
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    
    E_local_bot, nu_local_bot = get_material_properties(x_bot)
    lm_bot = (E_local_bot * nu_local_bot) / ((1 + nu_local_bot) * (1 - 2 * nu_local_bot))
    mu_bot = E_local_bot / (2 * (1 + nu_local_bot))
    lm_bot = lm_bot.unsqueeze(2)
    mu_bot = mu_bot.unsqueeze(2)
    
    v_bot = model(x_bot, 0)
    u_bot = v_bot
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    T_bot = -sig_bot[:, :, 2]
    bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
    residuals['bottom'] = bot_residual.cpu()

    return residuals
