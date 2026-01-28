

import torch
import numpy as np
import matplotlib.pyplot as plt
import pinn_config as config
import model

def plot_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pinn = model.MultiLayerPINN().to(device)
    try:
        pinn.load_state_dict(torch.load("pinn_model.pth"))
    except FileNotFoundError:
        print("Model not found, cannot plot.")
        return
    pinn.eval()
    
    # 1. Loss History
    try:
        loss_hist = np.load("loss_history.npy")
        plt.figure(figsize=(8, 5))
        plt.semilogy(loss_hist)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig("loss_curve.png")
        print("Saved loss_curve.png")
    except FileNotFoundError:
        print("Loss history not found.")

    # 2. Displacement Field u_z at Top Surface (z=H)
    # Grid
    n_plot = 100
    x = np.linspace(0, config.Lx, n_plot)
    y = np.linspace(0, config.Ly, n_plot)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * config.H
    
    # Flatten
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    e_ones = np.ones((pts.shape[0], 1)) * config.E_vals[0]
    pts = np.hstack([pts, e_ones])
    pts_t = torch.tensor(pts, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Top is Layer 3 (index 2)
        u = pinn(pts_t, 2)
        u_z = u[:, 2].cpu().numpy().reshape(X.shape)
        
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u_z, levels=50, cmap='jet')
    plt.colorbar(label="Vertical Displacement u_z")
    plt.title(f"Top Surface Displacement (z={config.H})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("displacement_top.png")
    print("Saved displacement_top.png")
    
    # 3. Cross Section Slice at y = Ly/2
    # We need to query from all 3 layers
    y_slice = config.Ly / 2
    z = np.linspace(0, config.H, 100)
    x = np.linspace(0, config.Lx, 100)
    X_slice, Z_slice = np.meshgrid(x, z)
    Y_slice = np.ones_like(X_slice) * y_slice
    
    pts_slice = np.stack([X_slice.ravel(), Y_slice.ravel(), Z_slice.ravel()], axis=1)
    e_ones = np.ones((pts_slice.shape[0], 1)) * config.E_vals[0]
    pts_slice = np.hstack([pts_slice, e_ones])
    pts_slice_t = torch.tensor(pts_slice, dtype=torch.float32).to(device)
    
    # We need to manually assign points to layers to query correctly
    # Only matters for output.
    # Actually, simpler: just sample dedicated grids for each layer
    
    u_z_combined = np.zeros_like(X_slice)
    
    # Layer 1 Grid
    z1 = np.linspace(config.Layer_Interfaces[0], config.Layer_Interfaces[1], 33)
    X1, Z1 = np.meshgrid(x, z1)
    Y1 = np.ones_like(X1) * y_slice
    p1 = np.stack([X1.ravel(), Y1.ravel(), Z1.ravel()], axis=1)
    e1 = np.ones((p1.shape[0], 1)) * config.E_vals[0]
    p1 = np.hstack([p1, e1])
    p1_t = torch.tensor(p1, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out1 = pinn(p1_t, 0)
        uz1 = out1[:, 2].cpu().numpy()
        
    # Layer 2 Grid
    z2 = np.linspace(config.Layer_Interfaces[1], config.Layer_Interfaces[2], 33)
    X2, Z2 = np.meshgrid(x, z2)
    Y2 = np.ones_like(X2) * y_slice
    p2 = np.stack([X2.ravel(), Y2.ravel(), Z2.ravel()], axis=1)
    e2 = np.ones((p2.shape[0], 1)) * config.E_vals[0]
    p2 = np.hstack([p2, e2])
    p2_t = torch.tensor(p2, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out2 = pinn(p2_t, 1)
        uz2 = out2[:, 2].cpu().numpy()
        
    # Layer 3 Grid
    z3 = np.linspace(config.Layer_Interfaces[2], config.Layer_Interfaces[3], 34)
    X3, Z3 = np.meshgrid(x, z3)
    Y3 = np.ones_like(X3) * y_slice
    p3 = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)
    e3 = np.ones((p3.shape[0], 1)) * config.E_vals[0]
    p3 = np.hstack([p3, e3])
    p3_t = torch.tensor(p3, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out3 = pinn(p3_t, 2)
        uz3 = out3[:, 2].cpu().numpy()
        
    # Combine for Scatter Plot (easiest for irregular grid concatenation)
    plt.figure(figsize=(10, 5))
    plt.scatter(X1.ravel(), Z1.ravel(), c=uz1, cmap='jet', s=5, vmin=min(uz1.min(), uz2.min(), uz3.min()), vmax=max(uz1.max(), uz2.max(), uz3.max()))
    plt.scatter(X2.ravel(), Z2.ravel(), c=uz2, cmap='jet', s=5, vmin=min(uz1.min(), uz2.min(), uz3.min()), vmax=max(uz1.max(), uz2.max(), uz3.max()))
    plt.scatter(X3.ravel(), Z3.ravel(), c=uz3, cmap='jet', s=5, vmin=min(uz1.min(), uz2.min(), uz3.min()), vmax=max(uz1.max(), uz2.max(), uz3.max()))
    
    plt.colorbar(label="u_z")
    plt.title("Cross Section Displacement u_z (y=Ly/2)")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axhline(config.Layer_Interfaces[1], color='k', linestyle='--', linewidth=0.5)
    plt.axhline(config.Layer_Interfaces[2], color='k', linestyle='--', linewidth=0.5)
    
    plt.savefig("cross_section.png")
    print("Saved cross_section.png")

if __name__ == "__main__":
    plot_results()
