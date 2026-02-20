
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_fem_complex(cfg, material_grid):
    """
    Solves FEA for a voxel grid with arbitrary per-element materials.
    
    Args:
        cfg: Configuration dict.
        material_grid: Int array (ne_x, ne_y, ne_z) with material indices.
                       Index -1 indicates VOID (inactive element).
    """
    print("Initializing Complex FEA Solver...")
    Lx, Ly, H = cfg['geometry']['Lx'], cfg['geometry']['Ly'], cfg['geometry']['H']
    ne_x = cfg.get('mesh', {}).get('ne_x', 30)
    ne_y = cfg.get('mesh', {}).get('ne_y', 30)
    ne_z = cfg.get('mesh', {}).get('ne_z', 30)
    
    dx, dy, dz = Lx/ne_x, Ly/ne_y, H/ne_z
    nx, ny, nz = ne_x+1, ne_y+1, ne_z+1
    n_dof = nx * ny * nz * 3
    
    # Pre-compute material matrices
    materials = cfg['material']
    C_matrices = []
    
    for mat in materials:
        E_val, nu_val = mat['E'], mat['nu']
        lam = (E_val * nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
        mu = E_val / (2 * (1 + nu_val))
        
        C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
        C = np.zeros((6, 6))
        C[0:3, 0:3] = lam
        np.fill_diagonal(C, C_diag)
        C_matrices.append(C)
        
    # Element integration points
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    invJ = np.diag([2/dx, 2/dy, 2/dz])
    detJ = dx * dy * dz / 8.0
    
    # Pre-compute Ke for each material
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

    print("Assembling...")
    
    # Flatten material grid to coordinate with element indices
    # material_grid is (ne_x, ne_y, ne_z).
    # Element index k (linear) maps to (ek, ej, ei) via unravel (if using C order).
    # Our linear index loop constructs:
    # el_indices = np.arange(...)
    # ek, ej, ei = np.unravel_index(..., (ne_z, ne_y, ne_x))
    # So we need to index material_grid as [ei, ej, ek]
    
    # Let's align indices carefully
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    
    # Get material index for each element
    # material_grid access: material_grid[ei, ej, ek]
    mat_indices_flat = material_grid[ei, ej, ek]
    
    # Filter out VOID elements (-1)
    # active_mask = (mat_indices_flat >= 0)
    # We will just skip them in loop
    
    # Node connections
    nx_pts = ne_x + 1
    ny_pts = ne_y + 1 # Correction: variable name override
    
    n0 = (ei) + (ej)*nx + (ek)*nx*ny
    n1 = (ei+1) + (ej)*nx + (ek)*nx*ny
    n2 = (ei+1) + (ej+1)*nx + (ek)*nx*ny
    n3 = (ei) + (ej+1)*nx + (ek)*nx*ny
    n4 = (ei) + (ej)*nx + (ek+1)*nx*ny
    n5 = (ei+1) + (ej)*nx + (ek+1)*nx*ny
    n6 = (ei+1) + (ej+1)*nx + (ek+1)*nx*ny
    n7 = (ei) + (ej+1)*nx + (ek+1)*nx*ny
    
    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1) # (n_elem, 8)
    
    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i*3:i*3+3] = conn[:, i:i+1] * 3 + np.array([0,1,2])
        
    vals_list = []
    rows_list = []
    cols_list = []
    
    # Iterate over active materials
    n_materials = len(materials)
    for m in range(n_materials):
        mask = (mat_indices_flat == m)
        n_elem_mat = np.sum(mask)
        if n_elem_mat == 0: continue
        
        Ke_flat = Ke_by_material[m].ravel()
        vals_layer = np.tile(Ke_flat, n_elem_mat)
        
        dof_subset = dof_indices[mask]
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
    
    # Load patch logic
    # We apply load to surface nodes. 
    # For complex geometry, "Top Surface" is defined by the highest *active* element k for each (i,j).
    # This is "Voxel Surface Loading".
    
    # Find top-most active nodes
    # Iterate x, y
    patch_x_min = cfg['load_patch']['x_start'] * Lx
    patch_x_max = cfg['load_patch']['x_end'] * Lx
    patch_y_min = cfg['load_patch']['y_start'] * Ly
    patch_y_max = cfg['load_patch']['y_end'] * Ly
    
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)
    
    for j in range(ny):
        if y_nodes[j] < patch_y_min or y_nodes[j] > patch_y_max: continue
        for i in range(nx):
            if x_nodes[i] < patch_x_min or x_nodes[i] > patch_x_max: continue
            
            # Find top active z index
            # Look at material_grid[i, j, :]? No, material grid is ELEMENT based (0..ne_x-1).
            # Node (i,j) is shared by elements (i-1, j-1), (i, j-1), (i-1, j), (i, j).
            # Simplified: Look at element column (i, j) (clamped).
            
            i_elem = min(i, ne_x-1)
            j_elem = min(j, ne_y-1)
            
            # Search downwards from top
            k_top_node = 0
            found = False
            for k in range(ne_z-1, -1, -1):
                if material_grid[i_elem, j_elem, k] != -1:
                    # Found top active element. Top node is k+1.
                    k_top_node = k + 1
                    found = True
                    break
            
            if found:
                # Apply load to node (i, j, k_top_node)
                n_idx = i + j*nx + k_top_node*nx*ny
                # Area factor? roughly dx*dy.
                # Normal vector? For now assume vertical load (-z).
                # Refine later if needed.
                F[3*n_idx + 2] -= p0 * dx * dy

    # BCs
    fixed_dofs = []
    # x=0, x=Lx (Clamped sides)
    # y=0, y=Ly (Clamped sides)
    # Bottom (z=0) is FREE? Or Clamped?
    # In PINN config: Sides are Clamped, Bottom is Free.
    
    # We only fix nodes that differ from VOID?
    # Or just fix boundary nodes if they are part of active elements.
    # For simplicity, fix all boundary nodes. The void ones don't matter (disconnected or 0 stiffness).
    # Wait, disconnected nodes result in singular matrix.
    # We should add a small stiffness to void nodes or fix them all.
    # Adding small stiffness to void elements is easier: E_void = 1e-6 * E_max.
    # Let's assume the user passes a separate material index for 'void' if they want weak stiffness,
    # OR we post-process K to identity for void DOFs.
    
    # Better approach: Fix all DOFs for inactive nodes.
    # Active nodes are those belonging to at least one active element.
    active_nodes = np.zeros(n_dof // 3, dtype=bool)
    active_mask_flat = (mat_indices_flat >= 0)
    # Mark nodes of active elements
    for c in range(8):
        active_nodes[conn[active_mask_flat, c]] = True
        
    inactive_dofs = []
    for n_idx in range(len(active_nodes)):
        if not active_nodes[n_idx]:
            inactive_dofs.extend([3*n_idx, 3*n_idx+1, 3*n_idx+2])
            
    # Add inactive DOFs to fixed set (ground them)
    fixed_dofs.extend(inactive_dofs)
    
    # Real BCs on active boundary nodes
    for j in range(ny):
        for k in range(nz):
            # x=0
            n_start = 0 + j*nx + k*nx*ny
            if active_nodes[n_start]:
                fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
            # x=Lx
            n_end = (nx-1) + j*nx + k*nx*ny
            if active_nodes[n_end]:
                fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])
                
    for i in range(nx):
        for k in range(nz):
            # y=0
            n_start = i + 0*nx + k*nx*ny
            if active_nodes[n_start]:
                fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
            # y=Ly
            n_end = i + (ny-1)*nx + k*nx*ny
            if active_nodes[n_end]:
                fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])

    fixed_dofs = np.unique(fixed_dofs)
    
    # Apply BCs
    penalty = 1e12
    # Ensure fixed_dofs is int array
    fixed_dofs = fixed_dofs.astype(int)
    
    K = K + sp.coo_matrix((np.ones(len(fixed_dofs))*penalty, (fixed_dofs, fixed_dofs)), shape=(n_dof, n_dof)).tocsr()
    
    print("Solving...")
    u = spla.spsolve(K, F)
    
    # Reshape
    u_grid = np.zeros((nx, ny, nz, 3))
    # We can reshape directly if order is correct
    # u is (n_dof,) -> (nz, ny, nx, 3) 
    # But our grid is (nx, ny, nz, 3) usually?
    # fem_solver returns (nx, ny, nz, 3).
    # Nodes are indexed i + j*nx + k*nx*ny.
    # So k is slowest, then j, then i.
    # This corresponds to C-order with shape (nz, ny, nx).
    
    u_reshaped = u.reshape((nz, ny, nx, 3))
    # Transpose to (nx, ny, nz, 3)
    u_grid = np.transpose(u_reshaped, (2, 1, 0, 3))
    
    return x_nodes, y_nodes, np.linspace(0, H, nz), u_grid
