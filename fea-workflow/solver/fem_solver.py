
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _hex8_stiffness(dx, dy, dz, E, nu):
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    ke = np.zeros((24, 24))
    c_diag = [lam + 2 * mu, lam + 2 * mu, lam + 2 * mu, mu, mu, mu]
    c_mat = np.zeros((6, 6))
    c_mat[0:3, 0:3] = lam
    np.fill_diagonal(c_mat, c_diag)

    for r_val in gp:
        for s_val in gp:
            for t_val in gp:
                inv_j = np.diag([2 / dx, 2 / dy, 2 / dz])
                det_j = dx * dy * dz / 8.0
                b_mat = np.zeros((6, 24))
                node_signs = [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
                for node_idx, (xi, eta, zeta) in enumerate(node_signs):
                    dN_dxi = 0.125 * xi * (1 + eta * s_val) * (1 + zeta * t_val)
                    dN_deta = 0.125 * eta * (1 + xi * r_val) * (1 + zeta * t_val)
                    dN_dzeta = 0.125 * zeta * (1 + xi * r_val) * (1 + eta * s_val)
                    d_global = inv_j @ np.array([dN_dxi, dN_deta, dN_dzeta])
                    nx_val, ny_val, nz_val = d_global
                    col = 3 * node_idx
                    b_mat[0, col] = nx_val
                    b_mat[1, col + 1] = ny_val
                    b_mat[2, col + 2] = nz_val
                    b_mat[3, col + 1] = nz_val
                    b_mat[3, col + 2] = ny_val
                    b_mat[4, col] = nz_val
                    b_mat[4, col + 2] = nx_val
                    b_mat[5, col] = ny_val
                    b_mat[5, col + 1] = nx_val
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    ke += b_mat.T @ c_mat @ b_mat * det_j
    return ke


def _assemble_and_solve(cfg, ke_per_element):
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))

    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z
    nx, ny, nz = ne_x + 1, ne_y + 1, ne_z + 1
    n_dof = nx * ny * nz * 3

    print("Assembling...")
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))

    n0 = (ei) + (ej) * nx + (ek) * nx * ny
    n1 = (ei + 1) + (ej) * nx + (ek) * nx * ny
    n2 = (ei + 1) + (ej + 1) * nx + (ek) * nx * ny
    n3 = (ei) + (ej + 1) * nx + (ek) * nx * ny
    n4 = (ei) + (ej) * nx + (ek + 1) * nx * ny
    n5 = (ei + 1) + (ej) * nx + (ek + 1) * nx * ny
    n6 = (ei + 1) + (ej + 1) * nx + (ek + 1) * nx * ny
    n7 = (ei) + (ej + 1) * nx + (ek + 1) * nx * ny

    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)

    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i * 3 : i * 3 + 3] = conn[:, i : i + 1] * 3 + np.array([0, 1, 2])

    dof_rows = np.repeat(dof_indices, 24, axis=1).ravel()
    dof_cols = np.tile(dof_indices, (1, 24)).ravel()

    vals = ke_per_element.reshape(-1)
    K = sp.coo_matrix((vals, (dof_rows, dof_cols)), shape=(n_dof, n_dof)).tocsr()

    F = np.zeros(n_dof)
    p0 = cfg["load_patch"]["pressure"]

    k = nz - 1
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)

    patch_x_min = cfg["load_patch"]["x_start"] * Lx
    patch_x_max = cfg["load_patch"]["x_end"] * Lx
    patch_y_min = cfg["load_patch"]["y_start"] * Ly
    patch_y_max = cfg["load_patch"]["y_end"] * Ly

    def load_mask(x, y):
        if x < patch_x_min or x > patch_x_max or y < patch_y_min or y > patch_y_max:
            return 0.0

        x_norm = (x - patch_x_min) / (patch_x_max - patch_x_min)
        y_norm = (y - patch_y_min) / (patch_y_max - patch_y_min)
        return 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)

    for j in range(ny):
        if y_nodes[j] >= patch_y_min and y_nodes[j] <= patch_y_max:
            for i in range(nx):
                if x_nodes[i] >= patch_x_min and x_nodes[i] <= patch_x_max:
                    mask = load_mask(x_nodes[i], y_nodes[j])
                    n_idx = i + j * nx + k * nx * ny
                    F[3 * n_idx + 2] -= p0 * mask * dx * dy

    fixed_dofs = []
    for j in range(ny):
        for k in range(nz):
            n_start = 0 + j * nx + k * nx * ny
            n_end = (nx - 1) + j * nx + k * nx * ny
            fixed_dofs.extend([3 * n_start, 3 * n_start + 1, 3 * n_start + 2])
            fixed_dofs.extend([3 * n_end, 3 * n_end + 1, 3 * n_end + 2])

    for i in range(nx):
        for k in range(nz):
            n_start = i + 0 * nx + k * nx * ny
            n_end = i + (ny - 1) * nx + k * nx * ny
            fixed_dofs.extend([3 * n_start, 3 * n_start + 1, 3 * n_start + 2])
            fixed_dofs.extend([3 * n_end, 3 * n_end + 1, 3 * n_end + 2])

    fixed_dofs = np.unique(fixed_dofs)

    penalty = 1e12
    K = K + sp.coo_matrix(
        (np.ones(len(fixed_dofs)) * penalty, (fixed_dofs, fixed_dofs)),
        shape=(n_dof, n_dof),
    ).tocsr()

    print("Solving...")
    u = spla.spsolve(K, F)

    u_grid = np.zeros((nx, ny, nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx + k * nx * ny
                u_grid[i, j, k] = u[3 * idx : 3 * idx + 3]

    return x_nodes, y_nodes, np.linspace(0, H, nz), u_grid


def solve_fem(cfg):
    print("Initializing FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    ke = _hex8_stiffness(dx, dy, dz, material["E"], material["nu"])
    ke_per_element = np.tile(ke.ravel(), ne_x * ne_y * ne_z).reshape(-1, 24 * 24)
    return _assemble_and_solve(cfg, ke_per_element)


def solve_two_layer_fem(cfg):
    print("Initializing Two-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [0.5 * H, 0.5 * H])
    if len(e_layers) != 2 or len(t_layers) != 2:
        raise ValueError("solve_two_layer_fem expects exactly two layers.")

    interface_z = float(t_layers[0])
    ke_bottom = _hex8_stiffness(dx, dy, dz, float(e_layers[0]), nu)
    ke_top = _hex8_stiffness(dx, dy, dz, float(e_layers[1]), nu)

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    is_bottom = z_centers <= interface_z
    ke_flat_bottom = ke_bottom.ravel()
    ke_flat_top = ke_top.ravel()
    ke_per_element = np.where(is_bottom[:, None], ke_flat_bottom[None, :], ke_flat_top[None, :])

    return _assemble_and_solve(cfg, ke_per_element)


def solve_three_layer_fem(cfg):
    print("Initializing Three-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [H / 3.0, H / 3.0, H / 3.0])
    if len(e_layers) != 3 or len(t_layers) != 3:
        raise ValueError("solve_three_layer_fem expects exactly three layers.")

    interfaces = np.cumsum(np.array(t_layers, dtype=float))
    ke_by_layer = np.stack(
        [_hex8_stiffness(dx, dy, dz, float(e_layers[i]), nu).ravel() for i in range(3)],
        axis=0,
    )

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    layer_ids = np.searchsorted(interfaces, z_centers, side="right")
    layer_ids = np.clip(layer_ids, 0, 2)
    ke_per_element = ke_by_layer[layer_ids]

    return _assemble_and_solve(cfg, ke_per_element)
