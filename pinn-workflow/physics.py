
import torch
import torch.autograd as autograd
import pinn_config as config

def compliance_scale(E, t=None):
    e_safe = torch.clamp(E, min=1e-8)
    if t is None:
        t = torch.ones_like(e_safe) * float(getattr(config, "H", 1.0))
    t_safe = torch.clamp(t, min=1e-8)
    h_ref = float(getattr(config, "H", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    disp_scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    if alpha == 0.0:
        return disp_scale / (e_safe ** e_pow)
    return disp_scale * (1.0 / (e_safe ** e_pow)) * (h_ref / t_safe) ** alpha

def v_to_u(v, E, t=None):
    return v * compliance_scale(E, t)

def _global_E_scale(x):
    return 0.5 * (x[:, 3:4] + x[:, 5:6])

def _global_thickness(x):
    return x[:, 4:5] + x[:, 6:7]

def _select_E_local(x):
    z_coord = x[:, 2:3]
    e1 = x[:, 3:4]
    e2 = x[:, 5:6]
    z_interface = x[:, 4:5]
    return torch.where(z_coord <= z_interface, e1, e2)

def _lame_from_E(E):
    nu = config.nu_vals[0]
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm.unsqueeze(2), mu.unsqueeze(2)

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

def compute_loss(model, data, device, weights=None):
    total_loss = 0
    losses = {}
    if weights is None:
        weights = config.WEIGHTS
    
    # --- 1. PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    
    # Dynamic material properties
    E_local = _select_E_local(x_int)
    E_scale = _global_E_scale(x_int)
    t_scale = _global_thickness(x_int)
    lm, mu = _lame_from_E(E_local)
    
    v_int = model(x_int, 0)

    if getattr(config, "ENFORCE_IMPACT_INVARIANCE", False):
        # Neutral-parameter mode: keep restitution/friction effect suppressed.
        x_int_variant = x_int.clone()
        r_min, r_max = getattr(config, "RESTITUTION_RANGE", (0.5, 0.5))
        mu_min, mu_max = getattr(config, "FRICTION_RANGE", (0.3, 0.3))
        v0_min, v0_max = getattr(config, "IMPACT_VELOCITY_RANGE", (1.0, 1.0))
        if r_max > r_min:
            x_int_variant[:, 7:8] = torch.rand_like(x_int_variant[:, 7:8]) * (r_max - r_min) + r_min
        if mu_max > mu_min:
            x_int_variant[:, 8:9] = torch.rand_like(x_int_variant[:, 8:9]) * (mu_max - mu_min) + mu_min
        if v0_max > v0_min:
            x_int_variant[:, 9:10] = torch.rand_like(x_int_variant[:, 9:10]) * (v0_max - v0_min) + v0_min
        v_int_variant = model(x_int_variant, 0)
        impact_invariance_loss = torch.mean((v_int - v_int_variant) ** 2)
    else:
        impact_invariance_loss = torch.zeros((), device=x_int.device)
    losses['impact_invariance'] = impact_invariance_loss
    total_loss += weights.get('impact_invariance', 0.0) * impact_invariance_loss

    # Predict displacement with a global scale so the displacement field stays single-valued
    # across the interface while constitutive response still uses local material properties.
    u = v_to_u(v_int, E_scale, t_scale)
    
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
    internal_energy = (energy_density * t_scale.squeeze(1)).mean() * (config.Lx * config.Ly)
    
    # --- 2. Dirichlet BCs (Clamped Sides) ---
    x_side = data['sides'][0].to(device)
    E_side = _global_E_scale(x_side)
    t_side = _global_thickness(x_side)
    v_side = model(x_side, 0)
    u_side = v_to_u(v_side, E_side, t_side)
    bc_loss = torch.mean(u_side**2)
    losses['bc_sides'] = bc_loss
    total_loss += weights['bc'] * bc_loss

    interface_loss = torch.zeros((), device=device)
    if 'interface' in data:
        x_if = data['interface'].to(device).detach().clone().requires_grad_(True)
        v_if = model(x_if, 0)
        u_if = v_to_u(v_if, _global_E_scale(x_if), _global_thickness(x_if))
        grad_u_if = gradient(u_if, x_if)
        eps_if = strain(grad_u_if)
        sig_if_1 = stress(eps_if, *_lame_from_E(x_if[:, 3:4]))
        sig_if_2 = stress(eps_if, *_lame_from_E(x_if[:, 5:6]))
        traction_1 = sig_if_1[:, :, 2]
        traction_2 = sig_if_2[:, :, 2]
        interface_loss = torch.mean((traction_1 - traction_2) ** 2)
    losses['interface_u'] = interface_loss
    total_loss += weights.get('interface_u', 0.0) * interface_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load = _select_E_local(x_top_load)
    E_scale_load = _global_E_scale(x_top_load)
    t_load = _global_thickness(x_top_load)
    lm, mu = _lame_from_E(E_local_load)
    
    v_top = model(x_top_load, 0)
    u_top = v_to_u(v_top, E_scale_load, t_load)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    
    T = sig_top[:, :, 2]
    
    mask = load_mask(x_top_load).unsqueeze(1)  # (N, 1)
    target_load = -config.p0 * mask
    impact_contact_loss = torch.zeros((), device=x_top_load.device)
    friction_coulomb_loss = torch.zeros((), device=x_top_load.device)
    friction_stick_loss = torch.zeros((), device=x_top_load.device)
    if getattr(config, "USE_EXPLICIT_IMPACT_PHYSICS", False):
        # Restitution-aware normal traction:
        # lower restitution -> stronger dissipative/impact contact response.
        restitution_local = torch.clamp(x_top_load[:, 7:8], 0.0, 1.0)
        impact_velocity_local = torch.clamp(x_top_load[:, 9:10], min=0.0)
        thickness_local = torch.clamp(t_load, min=1e-8)
        compression = torch.relu(-u_top[:, 2:3]) / thickness_local
        gain = float(getattr(config, "IMPACT_RESTITUTION_GAIN", 0.75))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        v_gain = float(getattr(config, "IMPACT_VELOCITY_GAIN", 0.20))
        v_ratio = impact_velocity_local / max(v0_ref, 1e-8)
        dynamic_scale = 1.0 + v_gain * (v_ratio ** 2)
        restitution_scale = 1.0 + gain * (1.0 - restitution_local) * compression
        target_load_contact = -config.p0 * mask * restitution_scale * dynamic_scale
        impact_contact_loss = torch.mean((T[:, 2:3] - target_load_contact) ** 2)
        losses['impact_contact'] = impact_contact_loss
        total_loss += weights.get('impact_contact', 0.0) * impact_contact_loss

        # Coulomb limit: ||T_t|| <= mu * |T_n| on loaded patch.
        mu_local = torch.clamp(x_top_load[:, 8:9], min=0.0)
        tangential_mag = torch.norm(T[:, :2], dim=1, keepdim=True)
        normal_mag = torch.abs(T[:, 2:3])
        friction_limit = mu_local * normal_mag
        # Slightly stricter friction limit at larger impact velocity.
        friction_limit = friction_limit / torch.sqrt(1.0 + v_gain * (v_ratio ** 2))
        friction_violation = torch.relu(tangential_mag - friction_limit) * mask
        friction_coulomb_loss = torch.mean(friction_violation ** 2)
        losses['friction_coulomb'] = friction_coulomb_loss
        total_loss += weights.get('friction_coulomb', 0.0) * friction_coulomb_loss

        # Stick-style regularization: higher mu discourages tangential slip.
        friction_stick_loss = torch.mean((mu_local * mask * u_top[:, :2]) ** 2)
        losses['friction_stick'] = friction_stick_loss
        total_loss += weights.get('friction_stick', 0.0) * friction_stick_loss

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
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    
    E_local_free = _select_E_local(x_top_free)
    E_scale_free = _global_E_scale(x_top_free)
    t_free = _global_thickness(x_top_free)
    lm_free, mu_free = _lame_from_E(E_local_free)
    
    v_top_free = model(x_top_free, 0)
    u_top_free = v_to_u(v_top_free, E_scale_free, t_free)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    
    loss_free = torch.mean(T_free**2)
    losses['free_top'] = loss_free
    total_loss += weights['bc'] * loss_free
    
    # Bottom Free
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    
    E_local_bot = _select_E_local(x_bot)
    E_scale_bot = _global_E_scale(x_bot)
    t_bot = _global_thickness(x_bot)
    lm_bot, mu_bot = _lame_from_E(E_local_bot)
    
    v_bot = model(x_bot, 0)
    u_bot = v_to_u(v_bot, E_scale_bot, t_bot)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    
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
        E_data = _global_E_scale(x_data)
        t_data = _global_thickness(x_data)
        
        # Ground truth mapping: v_target = u_data * E.
        scale_data = compliance_scale(E_data, t_data)
        v_target = u_data / torch.clamp(scale_data, min=1e-8)
        
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
    
    E_local = _select_E_local(x_int)
    E_scale = _global_E_scale(x_int)
    t_scale = _global_thickness(x_int)
    lm, mu = _lame_from_E(E_local)
    
    v_int = model(x_int, 0)
    u = v_to_u(v_int, E_scale, t_scale)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    residuals['interior'] = residual_mag.cpu()
    
    # --- BC Sides Residuals ---
    x_side = data['sides'][0].to(device)
    E_side = _global_E_scale(x_side)
    t_side = _global_thickness(x_side)
    v_side = model(x_side, 0)
    u_side = v_to_u(v_side, E_side, t_side)
    bc_residual = torch.sqrt(torch.sum(u_side**2, dim=1))
    residuals['sides'] = bc_residual.cpu()
    
    # --- Top Load Residuals ---
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load = _select_E_local(x_top_load)
    E_scale_load = _global_E_scale(x_top_load)
    t_load = _global_thickness(x_top_load)
    lm, mu = _lame_from_E(E_local_load)
    
    v_top = model(x_top_load, 0)
    u_top = v_to_u(v_top, E_scale_load, t_load)
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
    
    E_local_free = _select_E_local(x_top_free)
    E_scale_free = _global_E_scale(x_top_free)
    t_free = _global_thickness(x_top_free)
    lm_free, mu_free = _lame_from_E(E_local_free)
    
    v_top_free = model(x_top_free, 0)
    u_top_free = v_to_u(v_top_free, E_scale_free, t_free)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
    residuals['top_free'] = free_residual.cpu()
    
    # --- Bottom Residuals ---
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    
    E_local_bot = _select_E_local(x_bot)
    E_scale_bot = _global_E_scale(x_bot)
    t_bot = _global_thickness(x_bot)
    lm_bot, mu_bot = _lame_from_E(E_local_bot)
    
    v_bot = model(x_bot, 0)
    u_bot = v_to_u(v_bot, E_scale_bot, t_bot)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    T_bot = -sig_bot[:, :, 2]
    bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
    residuals['bottom'] = bot_residual.cpu()

    return residuals
