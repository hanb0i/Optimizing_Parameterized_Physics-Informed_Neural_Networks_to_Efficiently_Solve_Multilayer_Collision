
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss_history(loss_history, adam_epochs=None, save_path="."):
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, label='Total Loss')
    if adam_epochs is not None and adam_epochs < len(loss_history):
        plt.axvline(x=adam_epochs, color='r', linestyle='--', label='L-BFGS Start')
        plt.legend()
    plt.title("Training Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{save_path}/loss_history_plot.png")
    plt.close()
    print(f"Saved loss_history_plot.png to {save_path}")

    # Separate L-BFGS Plot
    if adam_epochs is not None and adam_epochs < len(loss_history):
        plt.figure(figsize=(10, 6))
        lbfgs_hist = loss_history[adam_epochs:]
        plt.plot(lbfgs_hist, 'b-o', markersize=3)
        plt.title("L-BFGS Training Loss Phase")
        plt.xlabel("L-BFGS Steps (x20 iterations)")
        plt.ylabel("Loss")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(f"{save_path}/lbfgs_loss_plot.png")
        plt.close()
        print(f"Saved lbfgs_loss_plot.png to {save_path}")

def plot_pinn_results(model, config, device, save_path="."):
    model.eval()
    Lx, Ly, H = config['geometry']['Lx'], config['geometry']['Ly'], config['geometry']['H']
    
    # Top surface
    n_plot = 100
    x = np.linspace(0, Lx, n_plot)
    y = np.linspace(0, Ly, n_plot)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * H
    
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    e_ones = np.ones((pts.shape[0], 1)) * config['material']['E']
    pts = np.hstack([pts, e_ones])
    pts_t = torch.tensor(pts, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u = model(pts_t, 2) # Layer 3
        u_z = u[:, 2].cpu().numpy().reshape(X.shape)
        
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u_z, levels=50, cmap='jet')
    plt.colorbar(label="u_z")
    plt.title("PINN Top Surface Displacement")
    plt.savefig(f"{save_path}/pinn_top.png")
    plt.close()

def plot_comparison(u_fea, params_fea, pinn_model, config, device, save_path="."):
    x_nodes, y_nodes, z_nodes = params_fea
    u_grid_fea = u_fea
    
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    X_fea, Y_fea = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    
    # Top surface FEA
    uz_fea_top = u_grid_fea[:, :, -1, 2]
    
    # PINN Prediction on FEA grid
    pts_top = np.stack(
        [X_fea.ravel(), Y_fea.ravel(), np.ones_like(X_fea.ravel()) * config['geometry']['H']],
        axis=1,
    )
    e_ones = np.ones((pts_top.shape[0], 1)) * config['material']['E']
    pts_top = np.hstack([pts_top, e_ones])
    pts_top_t = torch.tensor(pts_top, dtype=torch.float32).to(device)
    
    pinn_model.eval()
    with torch.no_grad():
        u_pinn = pinn_model(pts_top_t, 2).cpu().numpy()
        uz_pinn_top = u_pinn[:, 2].reshape(nx, ny)
        
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    c1 = axes[0].contourf(X_fea, Y_fea, uz_fea_top, levels=50, cmap='jet')
    axes[0].set_title("FEA u_z")
    plt.colorbar(c1, ax=axes[0])
    
    c2 = axes[1].contourf(X_fea, Y_fea, uz_pinn_top, levels=50, cmap='jet')
    axes[1].set_title("PINN u_z")
    plt.colorbar(c2, ax=axes[1])
    
    diff = np.abs(uz_fea_top - uz_pinn_top)
    c3 = axes[2].contourf(X_fea, Y_fea, diff, levels=50, cmap='magma')
    axes[2].set_title("Diff |FEA - PINN|")
    plt.colorbar(c3, ax=axes[2])
    
    plt.savefig(f"{save_path}/comparison.png")
    plt.close()
    
    print(f"Comparison: Max FEA {uz_fea_top.min():.4f}, Max PINN {uz_pinn_top.min():.4f}")
