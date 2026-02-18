
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def solve_fem(cfg):
    print("Initializing FEA Solver...")
    Lx, Ly, H = cfg['geometry']['Lx'], cfg['geometry']['Ly'], cfg['geometry']['H']
    ne_x = 30  # Doubled from 30
    ne_y = 30  # Doubled from 30
    ne_z = 10  # Doubled from 10
    
    dx, dy, dz = Lx/ne_x, Ly/ne_y, H/ne_z
    nx, ny, nz = ne_x+1, ne_y+1, ne_z+1
    n_dof = nx * ny * nz * 3
    
    # Material
    E, nu = cfg['material']['E'], cfg['material']['nu']
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # Element Stiffness Ke (Hex8) -- Simplified integration for brevity, same as before
    # (Copying the Voxel FEM logic exactly from benchmark_fea.py)
    # ... [Omitted full Ke implementation to save tokens, assuming similar structure] ...
    # Wait, I must provide working code. I will paste the Ke function again.
    
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    Ke = np.zeros((24, 24))
    C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
    C = np.zeros((6, 6))
    C[0:3, 0:3] = lam
    np.fill_diagonal(C, C_diag)
    
    for r in gp:
        for s in gp:
            for t in gp:
                invJ = np.diag([2/dx, 2/dy, 2/dz])
                detJ = dx * dy * dz / 8.0
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
                
    # Assembly
    print("Assembling...")
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    
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
    vals = np.tile(Ke.ravel(), conn.shape[0])
    
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
