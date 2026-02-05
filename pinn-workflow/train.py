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

def get_loss_weights(epoch, use_hard_bc=True):
    weights = dict(config.WEIGHTS)
    if config.WEIGHT_RAMP_EPOCHS > 0 and epoch < config.WEIGHT_RAMP_EPOCHS:
        ramp = max(1, config.WEIGHT_RAMP_EPOCHS)
        t = epoch / ramp
        weights['load'] = config.LOAD_WEIGHT_START + t * (config.WEIGHTS['load'] - config.LOAD_WEIGHT_START)
        weights['pde'] = config.PDE_WEIGHT_START + t * (config.WEIGHTS['pde'] - config.PDE_WEIGHT_START)
        weights['energy'] = config.ENERGY_WEIGHT_START + t * (config.WEIGHTS['energy'] - config.ENERGY_WEIGHT_START)
    if not use_hard_bc:
        weights['pde'] *= config.SOFT_MODE_PDE_WEIGHT_SCALE
        weights['load'] *= config.SOFT_MODE_LOAD_WEIGHT_SCALE
    return weights

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Using p0 = {config.p0}")
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    if config.FORCE_SOFT_SIDE_BC_FROM_START:
        config.USE_HARD_SIDE_BC = False
        pinn.set_hard_bc(False)
    
    # Initialize Optimizers
    # SOAP improves conditioning for stiff, multi-term PINN losses; prefer it when Adam/AdamW stagnates.
    # precondition_frequency controls how often curvature stats are refreshed: lower is more stable, higher is cheaper.
    optimizer_adam = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=0.0, #was 1e-2
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
        
        # Prepare FEM evaluation grid (append base E for parametric PINN)
        pts_fea = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1)
        e_ones = np.ones((pts_fea.shape[0], 1)) * config.E_vals[0]
        pts_fea = np.hstack([pts_fea, e_ones])
        pts_fea_tensor = torch.tensor(pts_fea, dtype=torch.float32).to(device)
        u_fea_flat = U_fea.reshape(-1, 3)
        
        fem_available = True
        print(f"FEM data loaded: {X_fea.shape}")
    except FileNotFoundError:
        print("FEM solution not found. Training without FEM comparison.")
        fem_available = False
    
    # Data Container
    training_data = data.get_data()

    # Load and attach Parametric/Hybrid Supervision Data
    if hasattr(config, "N_DATA_POINTS") and hasattr(config, "DATA_E_VALUES"):
        print(f"Loading hybrid supervision data (N={config.N_DATA_POINTS}, E={config.DATA_E_VALUES})...")
        x_data, u_data = data.load_fem_supervision_data()
        training_data['x_data'] = x_data
        training_data['u_data'] = u_data
        print(f"Attached {len(x_data)} supervision points to training data.")
    else:
        print("Parametric data config not found, skipping hybrid data loading.")
 
    # History - store all loss components separately for each optimizer
    adam_history = {
        'total': [],
        'pde': [],
        'bc_sides': [],
        'free_top': [],
        'free_bot': [],
        'load': [],
        'energy': [],
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
        'energy': [],
        'fem_mae': [],
        'fem_max_err': [],
        'steps': []
    }
    
    print("Starting SOAP Pretraining...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_ADAM):
        optimizer_adam.zero_grad()

        if config.FORCE_SOFT_SIDE_BC_FROM_START:
            use_hard_bc = False
        else:
            use_hard_bc = epoch < config.HARD_BC_EPOCHS
            if config.USE_HARD_SIDE_BC != use_hard_bc:
                config.USE_HARD_SIDE_BC = use_hard_bc
                pinn.set_hard_bc(use_hard_bc)
                if not use_hard_bc:
                    print("Switching to soft side BCs (mask off) to lift deflection.")
        
        if epoch % 500 == 0 and epoch > 0:
            # Compute residuals for adaptive sampling
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        weights = get_loss_weights(epoch, use_hard_bc)
        loss_val, losses = physics.compute_loss(pinn, training_data, device, weights=weights)
        loss_val.backward()
        optimizer_adam.step()
        scheduler.step()  # Update learning rate
        
        adam_history['total'].append(loss_val.item())
        adam_history['pde'].append(losses['pde'].item())
        adam_history['bc_sides'].append(losses['bc_sides'].item())
        adam_history['free_top'].append(losses['free_top'].item())
        adam_history['free_bot'].append(losses['free_bot'].item())
        adam_history['load'].append(losses['load'].item())
        adam_history['energy'].append(losses['energy'].item())
        
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            
            # Compute FEM error every 100 epochs
            if fem_available:
                with torch.no_grad():
                    v_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                    # Apply compliance scaling u = v / E
                    # E is the 4th column of input
                    E_vals = pts_fea[:, 3:4] # numpy array from line 71
                    u_pinn_flat = v_pinn_flat / (E_vals)
                     
                    diff = np.abs(u_pinn_flat - u_fea_flat)
                    mae = np.mean(diff)
                    max_err = np.max(diff)
                    adam_history['fem_mae'].append(mae)
                    adam_history['fem_max_err'].append(max_err)
                    adam_history['epochs'].append(epoch)
                    
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | Energy: {losses['energy']:.6f} | LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} | Time: {step_duration:.4f}s")
            else:
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | Energy: {losses['energy']:.6f} | "
                      f"LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"SOAP Pretraining Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # L-BFGS Fine-Tuning
    print("Starting L-BFGS Fine-Tuning...")
    config.USE_HARD_SIDE_BC = False
    pinn.set_hard_bc(False)
    optimizer_lbfgs = optim.LBFGS(
        pinn.parameters(),
        lr=1.0,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )
        
    num_lbfgs_steps = config.EPOCHS_LBFGS
    print(f"Running {num_lbfgs_steps} L-BFGS outer steps.")
    print("Resampling with residual-based adaptive sampling each outer step.")
    
    for i in range(num_lbfgs_steps):
        # Resample collocation points with residual-based adaptive sampling
        residuals = physics.compute_residuals(pinn, training_data, device)
        training_data = data.get_data(prev_data=training_data, residuals=residuals)
        
        step_start = time.time()
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_val, _ = physics.compute_loss(pinn, training_data, device, weights=config.WEIGHTS)
            loss_val.backward()
            return loss_val

        loss_val = optimizer_lbfgs.step(closure)
        step_end = time.time()
        
        # Compute losses for logging
        _, losses = physics.compute_loss(pinn, training_data, device, weights=config.WEIGHTS)
        lbfgs_history['total'].append(loss_val.item())
        lbfgs_history['pde'].append(losses['pde'].item())
        lbfgs_history['bc_sides'].append(losses['bc_sides'].item())
        lbfgs_history['free_top'].append(losses['free_top'].item())
        lbfgs_history['free_bot'].append(losses['free_bot'].item())
        lbfgs_history['load'].append(losses['load'].item())
        lbfgs_history['energy'].append(losses['energy'].item())
        
        # Compute FEM error and print
        if fem_available:
            with torch.no_grad():
                v_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                # Apply compliance scaling
                E_vals = pts_fea[:, 3:4]
                u_pinn_flat = v_pinn_flat / (E_vals)
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                lbfgs_history['fem_mae'].append(mae)
                lbfgs_history['fem_max_err'].append(max_err)
                lbfgs_history['steps'].append(i)
            print(f"L-BFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Energy: {losses['energy'].item():.6e} | "
                  f"FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"L-BFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Energy: {losses['energy'].item():.6e} | "
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
