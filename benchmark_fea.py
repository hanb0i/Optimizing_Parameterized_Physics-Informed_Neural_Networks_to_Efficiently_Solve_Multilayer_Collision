
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def element_stiffness_matrix(E, nu, dx, dy, dz):
    """
    Compute 24x24 stiffness matrix for a trilinear hexahedral element.
    Uses Gauss integration (2x2x2).
    """
    # Lame parameters
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # Gauss points and weights (2x2x2)
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    w = 1.0 # Weight is 1 for all points
    
    # Shape functions N(xi, eta, zeta) = 1/8 * (1 +/- xi)(1 +/- eta)(1 +/- zeta)
    # Order: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), (-1,-1,1), ...
    
    ke = np.zeros((24, 24))
    
    # Constitutive matrix C (Voigt notation: xx, yy, zz, yz, xz, xy)
    # Stress = C * Strain
    vals = [lam+2*mu, lam, lam, 0, 0, 0]
    C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
    # Full C matrix
    C = np.zeros((6, 6))
    C[0:3, 0:3] = lam
    np.fill_diagonal(C, C_diag)
    
    for r in gp:
        for s in gp:
            for t in gp:
                # Shape function derivatives dN/dxi, dN/deta, dN/dzeta
                # N_i = 1/8 * (1 + xi_i * xi) ...
                
                # Jacobian J
                # x = sum N_i x_i
                # dx/dxi = sum dN_i/dxi * x_i = dx/2 * sum ... = dx/2
                # J = diag(dx/2, dy/2, dz/2)
                invJ = np.diag([2/dx, 2/dy, 2/dz])
                detJ = dx * dy * dz / 8.0
                
                # B matrix: Strain = B * u_nodal
                B = np.zeros((6, 24))
                
                node_signs = [
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]
                ]
                
                for i in range(8):
                    xi_i, eta_i, zeta_i = node_signs[i]
                    
                    # Derivatives in local coords
                    dN_dxi = 0.125 * xi_i * (1 + eta_i * s) * (1 + zeta_i * t)
                    dN_deta = 0.125 * eta_i * (1 + xi_i * r) * (1 + zeta_i * t)
                    dN_dzeta = 0.125 * zeta_i * (1 + xi_i * r) * (1 + eta_i * s)
                    
                    d_local = np.array([dN_dxi, dN_deta, dN_dzeta])
                    
                    # Derivatives in global coords: dN/dx = J^-1 dN/dxi
                    d_global = invJ @ d_local
                    
                    nx, ny, nz = d_global[0], d_global[1], d_global[2]
                    
                    # Fill B matrix for node i (columns 3*i, 3*i+1, 3*i+2)
                    col = 3 * i
                    # xx
                    B[0, col] = nx
                    # yy
                    B[1, col+1] = ny
                    # zz
                    B[2, col+2] = nz
                    # yz = dy + dz
                    B[3, col+1] = nz
                    B[3, col+2] = ny
                    # xz = dx + dz
                    B[4, col] = nz
                    B[4, col+2] = nx
                    # xy = dx + dy
                    B[5, col] = ny
                    B[5, col+1] = nx
                    
                ke += B.T @ C @ B * detJ * w**3  # w=1
                
    return ke

def solve_fea():
    print("Initializing FEA Solver...")
    
    # 1. Mesh Parameters
    Lx, Ly, H = 1.0, 1.0, 0.1
    # Resolution (Elements)
    ne_x, ne_y, ne_z = 30, 30, 10
    
    dx = Lx / ne_x
    dy = Ly / ne_y
    dz = H / ne_z
    
    nx, ny, nz = ne_x + 1, ne_y + 1, ne_z + 1
    n_nodes = nx * ny * nz
    n_dof = n_nodes * 3
    
    print(f"Mesh: {ne_x}x{ne_y}x{ne_z} elements. {n_dof} DOFs.")
    
    # 2. Stiffness Matrix Assembly
    print("Computing Element Stiffness Matrices...")
    # Materials
    E_base, nu_base = 360.0, 0.3
    
    # Precompute Ke for base material (assuming valid for all)
    # Actually, if layers have same E,nu we only need one Ke.
    # Config says E=[1,1,1]. So consistent.
    Ke_base = element_stiffness_matrix(E_base, nu_base, dx, dy, dz)
    
    print("Assembling Global Stiffness Matrix...")
    # We'll use COO format for assembly
    rows, cols, data = [], [], []
    
    # Estimated non-zeros per row ~ 24*3 = 72? Actually 8 nodes * 3 * 24 = 576 entries per element
    # Total entries approx ne * 576. 30*30*10 = 9000 elements -> 5M entries. Feasible.
    
    # Element-Node Map
    # Node index: n = i + j*nx + k*nx*ny
    
    def get_node_idx(i, j, k):
        return i + j*nx + k*nx*ny

    # Pre-allocate arrays for speed would be better, but list append is safer for now.
    # To optimize: calc indices once.
    
    # Global K assembly loop
    # Vectorized approach:
    # 1. Generate connectivity for all elements
    el_indices = np.arange(ne_x * ne_y * ne_z)
    
    # Element grid coordinates
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    
    # 8 nodes per element
    # Order matches the Ke generation: (-,-,-), (+,-,-), ... -> i, i+1
    # Nodes:
    # 0: i, j, k
    # 1: i+1, j, k
    # 2: i+1, j+1, k
    # 3: i, j+1, k
    # 4: i, j, k+1
    # 5: i+1, j, k+1
    # 6: i+1, j+1, k+1
    # 7: i, j+1, k+1
    
    n0 = (ei) + (ej)*nx + (ek)*nx*ny
    n1 = (ei+1) + (ej)*nx + (ek)*nx*ny
    n2 = (ei+1) + (ej+1)*nx + (ek)*nx*ny
    n3 = (ei) + (ej+1)*nx + (ek)*nx*ny
    n4 = (ei) + (ej)*nx + (ek+1)*nx*ny
    n5 = (ei+1) + (ej)*nx + (ek+1)*nx*ny
    n6 = (ei+1) + (ej+1)*nx + (ek+1)*nx*ny
    n7 = (ei) + (ej+1)*nx + (ek+1)*nx*ny
    
    # Shape: (ne_total, 8)
    connectivity = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)
    
    # Expand to DOFs: node*3, node*3+1, node*3+2
    # Shape: (ne_total, 24)
    # 0 -> 0,1,2; 1->3,4,5 etc.
    dof_indices = np.zeros((connectivity.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i*3] = connectivity[:, i] * 3
        dof_indices[:, i*3+1] = connectivity[:, i] * 3 + 1
        dof_indices[:, i*3+2] = connectivity[:, i] * 3 + 2
        
    # Create I, J, V arrays
    # Ke is 24x24. We replicate it for all elements.
    n_elem = connectivity.shape[0]
    
    # Tile Ke entries
    Ke_flat = Ke_base.ravel() # 576
    V = np.tile(Ke_flat, n_elem) # n_elem * 576
    
    # Tile indices
    # I: Row indices. For each element, 24 rows, each repeating 24 times?
    # No, we need combinations. 
    # Row indices for an element: dof_indices[e, :] broadcasted
    
    # Efficient I, J construction
    # We need to repeat each row index 24 times for the columns
    # and tile the column indices 24 times for the rows
    
    local_rows, local_cols = np.indices((24, 24))
    local_rows = local_rows.ravel()
    local_cols = local_cols.ravel()
    
    # Global indices
    # I[k] = dof_indices[e, local_rows[p]]
    # J[k] = dof_indices[e, local_cols[p]]
    
    # Expand dof_indices to (n_elem, 24, 1) and (n_elem, 1, 24)?
    # Better:
    # Rows: repeat each dof 24 times
    # Cols: tile all dofs 24 times
    
    # This is heavy on memory if done naively in python loops.
    # Using broadcasting:
    
    dof_rows = np.repeat(dof_indices, 24, axis=1) # (n_elem, 576) [0,0...0, 1,1...1...]
    dof_cols = np.tile(dof_indices, (1, 24))      # (n_elem, 576) [0,1...23, 0,1...23...]
    
    I = dof_rows.ravel()
    J = dof_cols.ravel()
    
    K_global = sp.coo_matrix((V, (I, J)), shape=(n_dof, n_dof)).tocsr()
    print("Matrix Assembled.")
    
    # 3. Apply Loads
    # Load patch on Top Surface (z=H, k=nz-1)
    # x in [Lx/3, 2Lx/3], y in [Ly/3, 2Ly/3]
    F = np.zeros(n_dof)
    p0 = 0.1
    
    # Identify Top Nodes
    # k = nz-1
    # i goes from 0 to nx-1, j from 0 to ny-1
    # Use grid coords
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)
    
    # Force per node?
    # Convert pressure p0 to force.
    # F = integral(N^T * traction) dA
    # For uniform pressure on regular grid, internal nodes get p0 * dx * dy
    # Edge nodes get p0 * dx * dy / 2
    # Corner nodes get p0 * dx * dy / 4
    # Simple approximation: F_z = -p0 * dx * dy for nodes strictly inside patch
    
    patch_nodes = []
    
    for j in range(ny):
        y = y_nodes[j]
        for i in range(nx):
            x = x_nodes[i]
            
            # Check if in patch
            # Allow boundary tolerance
            if (x >= Lx/3 - 1e-5) and (x <= 2*Lx/3 + 1e-5) and \
               (y >= Ly/3 - 1e-5) and (y <= 2*Ly/3 + 1e-5):
                
                # Node index
                n_idx = get_node_idx(i, j, nz-1) # Top surface
                
                # Area weight
                weight = 1.0
                if abs(x - Lx/3) < 1e-5 or abs(x - 2*Lx/3) < 1e-5: weight *= 0.5
                if abs(y - Ly/3) < 1e-5 or abs(y - 2*Ly/3) < 1e-5: weight *= 0.5
                
                force = -p0 * dx * dy * weight
                
                # F_z is dof index 3*n_idx + 2
                F[3*n_idx + 2] += force
                patch_nodes.append(n_idx)
                
    print(f"Applied load to {len(patch_nodes)} nodes.")
    
    # 4. Boundary Conditions (Dirichlet)
    # Clamped Sides: x=0, x=Lx, y=0, y=Ly -> u=0
    print("Applying Dirichlet BCs...")
    
    fixed_dofs = []
    
    # x=0 (i=0) and x=Lx (i=nx-1)
    for j in range(ny):
        for k in range(nz):
            # i=0
            n1 = get_node_idx(0, j, k)
            fixed_dofs.extend([3*n1, 3*n1+1, 3*n1+2])
            # i=nx-1
            n2 = get_node_idx(nx-1, j, k)
            fixed_dofs.extend([3*n2, 3*n2+1, 3*n2+2])
            
    # y=0 (j=0) and y=Ly (j=ny-1)
    for i in range(nx):
        for k in range(nz):
            # j=0
            n1 = get_node_idx(i, 0, k)
            fixed_dofs.extend([3*n1, 3*n1+1, 3*n1+2])
            # j=ny-1
            n2 = get_node_idx(i, ny-1, k)
            fixed_dofs.extend([3*n2, 3*n2+1, 3*n2+2])
            
    fixed_dofs = np.unique(fixed_dofs)
    
    # Modify K and F for BCs (Identity method)
    # Set rows and cols of fixed DOFs to 0, diagonal to 1
    # Set F to target value (0)
    
    # This matrix modification is slow on large sparse matrices.
    # Better: set F_fixed = 0, keep K symmetric? 
    # Penalty method: Add large number to diagonal, set F = large * val
    # Penalty = 1e10 * max(diag)
    
    penalty = 1e12
    
    # Create penalty matrix
    # I_fix = fixed_dofs, J_fix = fixed_dofs, V = penalty
    
    # We only add to diagonal
    K_global = K_global + sp.coo_matrix(
        (np.ones(len(fixed_dofs))*penalty, (fixed_dofs, fixed_dofs)), 
        shape=(n_dof, n_dof)
    ).tocsr()
    
    # Forces are 0, so no change to F needed (penalty * 0 = 0)
    
    print("Solving System...")
    start = time.time()
    u = spla.spsolve(K_global, F)
    print(f"Solved in {time.time() - start:.2f}s")
    
    # 5. Export
    # Reshape to (Nx, Ny, Nz, 3)
    u_grid = np.zeros((nx, ny, nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = get_node_idx(i, j, k)
                u_grid[i, j, k, 0] = u[3*idx]
                u_grid[i, j, k, 1] = u[3*idx+1]
                u_grid[i, j, k, 2] = u[3*idx+2]
                
    # Save coordinate grids + u data for easy interpolation/plotting
    grid_x, grid_y, grid_z = np.meshgrid(x_nodes, y_nodes, np.linspace(0, H, nz), indexing='ij')
    
    np.save("fea_solution.npy", {
        'x': grid_x, 'y': grid_y, 'z': grid_z,
        'u': u_grid
    })
    print("Saved fea_solution.npy")

if __name__ == "__main__":
    solve_fea()
