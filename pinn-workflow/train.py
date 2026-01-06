
import torch
import torch.optim as optim
import numpy as np
import time
import os

import soap

import pinn_config as config
import data
import model
import physics
import matplotlib.pyplot as plt

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    
    # Initialize Optimizers
    # SOAP improves conditioning for stiff, multi-term PINN losses; prefer it when Adam/AdamW stagnates.
    # precondition_frequency controls how often curvature stats are refreshed: lower is more stable, higher is cheaper.
    optimizer_adam = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=1e-2,
        precondition_frequency=config.SOAP_PRECONDITION_FREQUENCY,
    )
    
    # Learning rate scheduler: reduce by 0.3 every epochs_adam//5 steps
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=config.EPOCHS_ADAM//5, gamma=0.3)
    
    # Load FEM data for comparison
    print("Loading FEM solution for comparison...")
    try:
        fem_data = np.load("fea_solution.npy", allow_pickle=True).item()
        X_fea = fem_data['x']
        Y_fea = fem_data['y']
        Z_fea = fem_data['z']
        U_fea = fem_data['u']
        
        # Prepare FEM evaluation grid
        pts_fea = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1)
        pts_fea_tensor = torch.tensor(pts_fea, dtype=torch.float32).to(device)
        u_fea_flat = U_fea.reshape(-1, 3)
        
        fem_available = True
        print(f"FEM data loaded: {X_fea.shape}")
    except FileNotFoundError:
        print("FEM solution not found. Training without FEM comparison.")
        fem_available = False
    
    # Data Container
    training_data = data.get_data()
    
    # History - store all loss components separately for each optimizer
    adam_history = {
        'total': [],
        'pde': [],
        'bc_sides': [],
        'free_top': [],
        'free_bot': [],
        'load': [],
        'fem_mae': [],
        'fem_max_err': [],
        'epochs': []
    }
    
    lbfgs_history = {
        'total': [],
        'pde': [],
        'bc_sides': [],
        'free_top': [],
        'free_bot': [],
        'load': [],
        'fem_mae': [],
        'fem_max_err': [],
        'steps': []
    }
    
    print("Starting SOAP Training...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_ADAM):
        optimizer_adam.zero_grad()
        
        # Periodic data refresh with residual-based adaptive sampling
        if epoch % 500 == 0 and epoch > 0:
            # Compute residuals for adaptive sampling
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_adam.step()
        scheduler.step()  # Update learning rate
        
        adam_history['total'].append(loss_val.item())
        adam_history['pde'].append(losses['pde'].item())
        adam_history['bc_sides'].append(losses['bc_sides'].item())
        adam_history['free_top'].append(losses['free_top'].item())
        adam_history['free_bot'].append(losses['free_bot'].item())
        adam_history['load'].append(losses['load'].item())
        
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            
            # Compute FEM error every 100 epochs
            if fem_available:
                with torch.no_grad():
                    u_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                    diff = np.abs(u_pinn_flat - u_fea_flat)
                    mae = np.mean(diff)
                    max_err = np.max(diff)
                    adam_history['fem_mae'].append(mae)
                    adam_history['fem_max_err'].append(max_err)
                    adam_history['epochs'].append(epoch)
                    
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} | Time: {step_duration:.4f}s")
            else:
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"SOAP Training Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # SOAP Fine-Tuning
    print("Starting SOAP Fine-Tuning...")
    optimizer_lbfgs = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=1e-2,
        precondition_frequency=config.SOAP_PRECONDITION_FREQUENCY,
    )
        
    num_lbfgs_steps = config.EPOCHS_LBFGS
    print(f"Running {num_lbfgs_steps} SOAP outer steps.")
    print("Resampling with residual-based adaptive sampling each outer step.")
    
    for i in range(num_lbfgs_steps):
        # Resample collocation points with residual-based adaptive sampling
        residuals = physics.compute_residuals(pinn, training_data, device)
        training_data = data.get_data(prev_data=training_data, residuals=residuals)
        
        step_start = time.time()
        optimizer_lbfgs.zero_grad()
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_lbfgs.step()
        step_end = time.time()
        
        # Compute losses for logging
        _, losses = physics.compute_loss(pinn, training_data, device)
        lbfgs_history['total'].append(loss_val.item())
        lbfgs_history['pde'].append(losses['pde'].item())
        lbfgs_history['bc_sides'].append(losses['bc_sides'].item())
        lbfgs_history['free_top'].append(losses['free_top'].item())
        lbfgs_history['free_bot'].append(losses['free_bot'].item())
        lbfgs_history['load'].append(losses['load'].item())
        
        # Compute FEM error and print
        if fem_available:
            with torch.no_grad():
                u_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                lbfgs_history['fem_mae'].append(mae)
                lbfgs_history['fem_max_err'].append(max_err)
                lbfgs_history['steps'].append(i)
            print(f"SOAP Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"SOAP Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Time: {step_end - step_start:.4f}s")
        
        # Save model at every L-BFGS step
        torch.save(pinn.state_dict(), "pinn_model.pth")
            
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), "pinn_model.pth")
    loss_history = {'adam': adam_history, 'lbfgs': lbfgs_history}
    np.save("loss_history.npy", loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
