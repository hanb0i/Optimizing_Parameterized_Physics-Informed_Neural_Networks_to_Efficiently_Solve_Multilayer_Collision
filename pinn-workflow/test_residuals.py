# Compute PDE residual on x-z plane at mid-y
import physics
import numpy as np

# Sample points on x-z plane
nx_res, nz_res = 50, 20
x_res = np.linspace(0, 1.0, nx_res)
z_res = np.linspace(0, 0.1, nz_res)
y_mid = 0.5

X_res, Z_res = np.meshgrid(x_res, z_res, indexing='ij')
Y_res = np.ones_like(X_res) * y_mid

pts_res = np.stack([X_res.ravel(), Y_res.ravel(), Z_res.ravel()], axis=1)
pts_res_tensor = torch.tensor(pts_res, dtype=torch.float32, requires_grad=True).to(device)

# Get Lame parameters
lm, mu = config.Lame_Params[0]

# Compute displacement
u = pinn(pts_res_tensor, 0)

# Compute gradients and stress
grad_u = physics.gradient(u, pts_res_tensor)
eps = physics.strain(grad_u)
sig = physics.stress(eps, lm, mu)

# Compute divergence (PDE residual)
div_sigma = physics.divergence(sig, pts_res_tensor)
residual = -div_sigma  # Equilibrium: -div(sigma) = 0

# Convert to numpy and compute magnitude
residual_np = residual.detach().cpu().numpy()
residual_mag = np.sqrt(np.sum(residual_np**2, axis=1))
residual_mag_grid = residual_mag.reshape(X_res.shape)

print(f"PDE Residual Statistics (x-z plane):")
print(f"  Mean: {residual_mag.mean():.6e}")
print(f"  Max: {residual_mag.max():.6e}")
print(f"  Min: {residual_mag.min():.6e}")
# Optionally, visualize the residual magnitude
# Plot PDE residual magnitude
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

c = ax.contourf(X_res, Z_res, residual_mag_grid, levels=50, cmap='viridis')
ax.set_title("PDE Residual Magnitude |−∇·σ| (x-z plane at y=0.5)", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("z")
cbar = plt.colorbar(c, ax=ax)
cbar.set_label("Residual Magnitude")

plt.tight_layout()
plt.savefig("pde_residual_xz.png", dpi=150)
print("Saved pde_residual_xz.png")
plt.show()
# Plot individual components of PDE residual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

residual_x = residual_np[:, 0].reshape(X_res.shape)
residual_y = residual_np[:, 1].reshape(X_res.shape)
residual_z = residual_np[:, 2].reshape(X_res.shape)

# X-component
c1 = axes[0].contourf(X_res, Z_res, residual_x, levels=50, cmap='RdBu_r')
axes[0].set_title("PDE Residual: X-component", fontsize=12)
axes[0].set_xlabel("x")
axes[0].set_ylabel("z")
plt.colorbar(c1, ax=axes[0])

# Y-component
c2 = axes[1].contourf(X_res, Z_res, residual_y, levels=50, cmap='RdBu_r')
axes[1].set_title("PDE Residual: Y-component", fontsize=12)
axes[1].set_xlabel("x")
axes[1].set_ylabel("z")
plt.colorbar(c2, ax=axes[1])

# Z-component
c3 = axes[2].contourf(X_res, Z_res, residual_z, levels=50, cmap='RdBu_r')
axes[2].set_title("PDE Residual: Z-component", fontsize=12)
axes[2].set_xlabel("x")
axes[2].set_ylabel("z")
plt.colorbar(c3, ax=axes[2])

plt.tight_layout()
plt.savefig("pde_residual_components_xz.png", dpi=150)
print("Saved pde_residual_components_xz.png")
plt.show()