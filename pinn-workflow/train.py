
import torch
import torch.optim as optim
import numpy as np
import time

from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import sys
import os
sys.path.append(os.path.dirname(__file__))

import scipy_patch
import pinn_config as config
import data
import model
import physics

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    
    # SOAP Optimizer with learning rate scheduler
    optimizer_soap = optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer_soap, step_size=400, gamma=0.3)
    
    # Initial Data
    training_data = data.get_data()
    
    # Load FEM data for comparison
    print("Loading FEM solution for comparison...")
    try:
        fea_path = "fea_solution.npy"
        if not os.path.exists(fea_path):
            fea_path = os.path.join(os.path.dirname(__file__), "..", "fea_solution.npy")
        fem_data = np.load(fea_path, allow_pickle=True).item()
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
    except Exception as e:
        print(f"FEM data not available: {e}")
        fem_available = False
    
    soap_history = {
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
    
    ssbfgs_history = {
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
    
    print("Starting SOAP Pretraining...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_SOAP):
        optimizer_soap.zero_grad()
        
        # Periodic data refresh with residual-based adaptive sampling
        if epoch % 500 == 0 and epoch > 0:
            # Compute residuals for adaptive sampling
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_soap.step()
        scheduler.step()  # Update learning rate
        
        soap_history['total'].append(loss_val.item())
        soap_history['pde'].append(losses['pde'].item())
        soap_history['bc_sides'].append(losses['bc_sides'].item())
        soap_history['free_top'].append(losses['free_top'].item())
        soap_history['free_bot'].append(losses['free_bot'].item())
        soap_history['load'].append(losses['load'].item())
        
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
                    soap_history['fem_mae'].append(mae)
                    soap_history['fem_max_err'].append(max_err)
                    soap_history['epochs'].append(epoch)
                    
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
            
    print(f"SOAP Pretraining Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # SciPy self-scaled BFGS fine-tuning
    print(f"Starting SciPy SSBFGS Fine-Tuning ({config.SS_BFGS_VARIANT})...")
    if scipy_patch.ensure_scipy_bfgs_patch():
        print("Applied local SciPy optimize patch for method_bfgs support.")

    param_device = next(pinn.parameters()).device
    param_dtype = next(pinn.parameters()).dtype

    def _set_params(flat_params):
        flat_tensor = torch.as_tensor(flat_params, dtype=param_dtype, device=param_device)
        with torch.no_grad():
            vector_to_parameters(flat_tensor, pinn.parameters())

    def loss_and_grad(flat_params):
        _set_params(flat_params)
        loss_val, _ = physics.compute_loss(pinn, training_data, device)
        grads = torch.autograd.grad(loss_val, pinn.parameters(), create_graph=False, retain_graph=False)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        return float(loss_val.item()), grad_flat.detach().cpu().numpy().astype(np.float64, copy=False)

    num_ssbfgs_steps = config.EPOCHS_SSBFGS
    print(f"Running {num_ssbfgs_steps} SSBFGS outer steps.")
    print("Resampling with residual-based adaptive sampling each outer step.")

    initial_weights = parameters_to_vector(pinn.parameters()).detach().cpu().numpy().astype(np.float64, copy=False)
    hess_inv0 = np.eye(initial_weights.size, dtype=np.float64)

    for i in range(num_ssbfgs_steps):
        # Resample collocation points with residual-based adaptive sampling
        residuals = physics.compute_residuals(pinn, training_data, device)
        training_data = data.get_data(prev_data=training_data, residuals=residuals)

        step_start = time.time()
        result = minimize(
            loss_and_grad,
            initial_weights,
            method=config.SS_BFGS_METHOD,
            jac=True,
            options={
                'maxiter': config.SS_BFGS_MAXITER,
                'gtol': config.SS_BFGS_GTOL,
                'hess_inv0': hess_inv0,
                'method_bfgs': config.SS_BFGS_VARIANT,
                'initial_scale': config.SS_BFGS_INITIAL_SCALE,
            },
            tol=0.0,
        )
        step_end = time.time()

        if not result.success:
            print(f"  SciPy minimize status {result.status}: {result.message}")

        initial_weights = result.x
        _set_params(initial_weights)

        hess_inv0 = getattr(result, "hess_inv", None)
        if isinstance(hess_inv0, np.ndarray):
            hess_inv0 = 0.5 * (hess_inv0 + hess_inv0.T)
            try:
                cholesky(hess_inv0)
            except LinAlgError:
                hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)
        else:
            hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)

        # Compute losses for logging
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        ssbfgs_history['total'].append(loss_val.item())
        ssbfgs_history['pde'].append(losses['pde'].item())
        ssbfgs_history['bc_sides'].append(losses['bc_sides'].item())
        ssbfgs_history['free_top'].append(losses['free_top'].item())
        ssbfgs_history['free_bot'].append(losses['free_bot'].item())
        ssbfgs_history['load'].append(losses['load'].item())

        # Compute FEM error and print
        if fem_available:
            with torch.no_grad():
                u_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                ssbfgs_history['fem_mae'].append(mae)
                ssbfgs_history['fem_max_err'].append(max_err)
                ssbfgs_history['steps'].append(i)
            print(f"SSBFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"SSBFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Time: {step_end - step_start:.4f}s")

        # Save model at every SSBFGS step
        torch.save(pinn.state_dict(), "pinn_model.pth")
            
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), "pinn_model.pth")
    loss_history = {'soap': soap_history, 'ssbfgs': ssbfgs_history}
    np.save("loss_history.npy", loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
