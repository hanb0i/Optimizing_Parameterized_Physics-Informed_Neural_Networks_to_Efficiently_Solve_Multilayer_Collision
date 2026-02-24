
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def solve_fem(cfg):
    print("Initializing FEA Solver...")
    Lx, Ly, H = cfg['geometry']['Lx'], cfg['geometry']['Ly'], cfg['geometry']['H']
    Lx, Ly, H = cfg['geometry']['Lx'], cfg['geometry']['Ly'], cfg['geometry']['H']
    ne_x = cfg.get('mesh', {}).get('ne_x', 30)
    ne_y = cfg.get('mesh', {}).get('ne_y', 30)
    ne_z = cfg.get('mesh', {}).get('ne_z', 30)  # Increased default to 30 for multi-layer support
    
    dx, dy, dz = Lx/ne_x, Ly/ne_y, H/ne_z
    nx, ny, nz = ne_x+1, ne_y+1, ne_z+1
    n_dof = nx * ny * nz * 3
    
    # --- Materials / Layering ---
    # Backward compatible:
    # - cfg['material'] can be a single dict: {'E':..., 'nu':...}
    # - cfg['material'] can be a list of dicts (multi-material), and we split z uniformly across them.
    #
    # New (preferred for PINN parity):
    # - cfg['layers'] = [{'t': t1, 'E': E1, 'nu': nu1}, ...] where sum(t) ~= H.
    #   Element material is chosen by z-centroid vs cumulative thickness boundaries.
    layers = cfg.get('layers', None)
    if layers is not None:
        materials = [{'E': float(l['E']), 'nu': float(l['nu'])} for l in layers]
        layer_thicknesses = np.array([float(l.get('t', l.get('thickness'))) for l in layers], dtype=float)
        if np.any(layer_thicknesses <= 0):
            raise ValueError(f"All layer thicknesses must be > 0. Got: {layer_thicknesses}")
        t_sum = float(layer_thicknesses.sum())
        if t_sum <= 0:
            raise ValueError("Sum of layer thicknesses must be > 0.")
        # Normalize thicknesses to match H exactly (avoid drift if user passes fractions or slightly-off sums).
        layer_thicknesses = layer_thicknesses * (float(H) / t_sum)
    else:
        # Pre-compute material matrices for each layer
        # Expect cfg['material'] to be a list of dicts, or a single dict (for backward compatibility)
        materials = cfg['material']
        if isinstance(materials, dict):
            materials = [materials]  # Single layer case
        layer_thicknesses = None
        
    C_matrices = []
    lam_vals = []
    mu_vals = []
    
    for mat in materials:
        E_val, nu_val = mat['E'], mat['nu']
        lam = (E_val * nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
        mu = E_val / (2 * (1 + nu_val))
        lam_vals.append(lam)
        mu_vals.append(mu)
        
        C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
        C = np.zeros((6, 6))
        C[0:3, 0:3] = lam
        np.fill_diagonal(C, C_diag)
        C_matrices.append(C)
        
    # Element integration points and shape function derivatives (constant for Hex8)
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    invJ = np.diag([2/dx, 2/dy, 2/dz])
    detJ = dx * dy * dz / 8.0
    
    # Pre-calculate B matrix part that depends on geometry (same for all elements if regular grid)
    # Actually, B depends on r,s,t (integration points), so we compute Ke_base with C=Identity first?
    # No, easier to just compute Ke for each material type once.
    
    Ke_by_material = []
    for mat_idx in range(len(materials)):
        Ke = np.zeros((24, 24))
        C = C_matrices[mat_idx]
        
        for r in gp:
            for s in gp:
                for t in gp:
                    B = np.zeros((6, 24))
                    node_signs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
                    for i in range(8):
                        xi, eta, zeta = node_signs[i]
                        dN_dxi = 0.125 * xi * (1 + eta * s) * (1 + zeta * t)
                        dN_deta = 0.125 * eta * (1 + xi * r) * (1 + zeta * t)
                        dN_dzeta = 0.125 * zeta * (1 + xi * r) * (1 + eta * s)
                        d_global = invJ @ np.array([dN_dxi, dN_deta, dN_dzeta])
                        nx_val, ny_val, nz_val = d_global
                        col = 3 * i
                        B[0, col] = nx_val
                        B[1, col+1] = ny_val
                        B[2, col+2] = nz_val
                        B[3, col+1] = nz_val; B[3, col+2] = ny_val
                        B[4, col] = nz_val; B[4, col+2] = nx_val
                        B[5, col] = ny_val; B[5, col+1] = nx_val
                    Ke += B.T @ C @ B * detJ
        Ke_by_material.append(Ke)

    # Assembly
    print("Assembling...")
    
    n_layers = len(materials)
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    
    # ek is the element index in Z direction (0 to ne_z-1)
    if layer_thicknesses is None:
        # Old behavior: uniform split in element counts.
        layers_per_mat = max(1, ne_z // n_layers)
        if ne_z % n_layers != 0:
            print(f"Warning: ne_z={ne_z} not divisible by n_layers={n_layers}. Layer boundaries may be approximate.")
        mat_indices = np.minimum(ek // layers_per_mat, n_layers - 1)
    else:
        # New behavior: choose by z-centroid vs cumulative thickness boundaries.
        zc = (ek.astype(float) + 0.5) * float(dz)  # centroid in [0,H]
        bounds = np.concatenate([[0.0], np.cumsum(layer_thicknesses)])
        bounds[-1] = float(H)
        mat_indices = np.searchsorted(bounds[1:], zc, side='right')
        mat_indices = np.clip(mat_indices, 0, n_layers - 1).astype(int)
    
    n0 = (ei) + (ej)*nx + (ek)*nx*ny
    n1 = (ei+1) + (ej)*nx + (ek)*nx*ny
    n2 = (ei+1) + (ej+1)*nx + (ek)*nx*ny
    n3 = (ei) + (ej+1)*nx + (ek)*nx*ny
    n4 = (ei) + (ej)*nx + (ek+1)*nx*ny
    n5 = (ei+1) + (ej)*nx + (ek+1)*nx*ny
    n6 = (ei+1) + (ej+1)*nx + (ek+1)*nx*ny
    n7 = (ei) + (ej+1)*nx + (ek+1)*nx*ny
    
    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)
    
    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i*3:i*3+3] = conn[:, i:i+1] * 3 + np.array([0,1,2])
        
    dof_rows = np.repeat(dof_indices, 24, axis=1).ravel()
    dof_cols = np.tile(dof_indices, (1, 24)).ravel()
    
    # Vectorized assignment of Ke values
    # We need to construct the 'vals' array differently because Ke varies per element
    # But wait, Ke only has n_layers distinct values.
    # We can iterate over layers to build the sparse matrix components? 
    # Or just flatten everything.
    
    # Let's do a loop over unique material indices to avoid massive broadcasting complexity
    vals_list = []
    rows_list = []
    cols_list = []
    
    for m in range(n_layers):
        mask = (mat_indices == m)
        n_elem_layer = np.sum(mask)
        if n_elem_layer == 0: continue
        
        # Ke for this material
        Ke_flat = Ke_by_material[m].ravel()
        
        # Repeat Ke for all elements in this layer
        vals_layer = np.tile(Ke_flat, n_elem_layer)
        
        # Extract DOF indices for these elements
        # dof_indices is (n_elem_total, 24)
        # We need rows/cols for just the masked elements
        dof_subset = dof_indices[mask] # (n_elem_layer, 24)
        
        rows_layer = np.repeat(dof_subset, 24, axis=1).ravel()
        cols_layer = np.tile(dof_subset, (1, 24)).ravel()
        
        vals_list.append(vals_layer)
        rows_list.append(rows_layer)
        cols_list.append(cols_layer)
        
    vals = np.concatenate(vals_list)
    dof_rows = np.concatenate(rows_list)
    dof_cols = np.concatenate(cols_list)
    
    K = sp.coo_matrix((vals, (dof_rows, dof_cols)), shape=(n_dof, n_dof)).tocsr()
    
    # Load
    F = np.zeros(n_dof)
    p0 = cfg['load_patch']['pressure']
    
    # Identify Surface Nodes
    k = nz-1
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)
    
    patch_x_min = cfg['load_patch']['x_start'] * Lx
    patch_x_max = cfg['load_patch']['x_end'] * Lx
    patch_y_min = cfg['load_patch']['y_start'] * Ly
    patch_y_max = cfg['load_patch']['y_end'] * Ly
    
    # Load mask function
    def load_mask(x, y):
        """
        Supports both Quadratic falloff and Hard rectangular mask.
        """
        if x < patch_x_min or x > patch_x_max or y < patch_y_min or y > patch_y_max:
            return 0.0
            
        if not cfg.get('use_soft_mask', True):
            return 1.0
        
        # Normalize to [0, 1] within patch for soft mask
        x_norm = (x - patch_x_min) / (patch_x_max - patch_x_min)
        y_norm = (y - patch_y_min) / (patch_y_max - patch_y_min)
        
        # Quadratic falloff
        return 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)
    
    for j in range(ny):
        if y_nodes[j] >= patch_y_min and y_nodes[j] <= patch_y_max:
            for i in range(nx):
                if x_nodes[i] >= patch_x_min and x_nodes[i] <= patch_x_max:
                    # Apply soft edge mask to load
                    mask = load_mask(x_nodes[i], y_nodes[j])
                    n_idx = i + j*nx + k*nx*ny
                    F[3*n_idx + 2] -= p0 * mask * dx * dy
                    
    # BCs
    fixed_dofs = []
    # x=0, x=Lx
    for j in range(ny):
        for k in range(nz):
            n_start = 0 + j*nx + k*nx*ny
            n_end = (nx-1) + j*nx + k*nx*ny
            fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
            fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])
            
    # y=0, y=Ly
    for i in range(nx):
        for k in range(nz):
            n_start = i + 0*nx + k*nx*ny
            n_end = i + (ny-1)*nx + k*nx*ny
            fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
            fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])
            
    fixed_dofs = np.unique(fixed_dofs)
    
    penalty = 1e12
    K = K + sp.coo_matrix((np.ones(len(fixed_dofs))*penalty, (fixed_dofs, fixed_dofs)), shape=(n_dof, n_dof)).tocsr()
    
    print("Solving...")
    u = spla.spsolve(K, F)
    
    # Reshape
    u_grid = np.zeros((nx, ny, nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + j*nx + k*nx*ny
                u_grid[i,j,k] = u[3*idx:3*idx+3]
                
    return x_nodes, y_nodes, np.linspace(0, H, nz), u_grid
