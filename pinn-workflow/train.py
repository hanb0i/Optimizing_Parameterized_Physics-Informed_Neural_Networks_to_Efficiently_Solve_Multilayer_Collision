
import torch
import torch.optim as optim
import numpy as np
import time

from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import soap
import scipy_patch

import pinn_config as config
import data
import model
import physics

def train():
    """
    Main training function for the PINN.
    Implements a two-stage training strategy:
    1. Pre-training with SOAP (Second-Order Adaptive Preconditioning) to find a good basin of attraction.
    2. Fine-tuning with SciPy's L-BFGS (Self-Scaled BFGS) for high-precision convergence.
    """
    
    # ---------------------------------------------------------
    # 1. SETUP & INITIALIZATION
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Neural Network Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    
    # Initialize SOAP Optimizer (Stage 1)
    # SOAP uses a preconditioning matrix to approximate second-order information,
    # which helps navigate the complex loss landscape of PINNs.
    optimizer_soap = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=0.0,
        precondition_frequency=config.SOAP_PRECONDITION_FREQUENCY,
    )
    
    # Learning rate scheduler: reduce by 0.3 every epochs_soap//5 steps
    scheduler = optim.lr_scheduler.StepLR(optimizer_soap, step_size=config.EPOCHS_SOAP//5, gamma=0.3)
    
    # Load FEM data for comparison (Ground Truth)
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
    
    # Generate Initial Training Data
    training_data = data.get_data()
    
    # History Containers - store loss components separately for analysis
    soap_history = {
        'total': [], 'pde': [], 'bc_sides': [], 'free_top': [], 'free_bot': [], 'load': [],
        'fem_mae': [], 'fem_max_err': [], 'epochs': []
    }
    
    ssbfgs_history = {
        'total': [], 'pde': [], 'bc_sides': [], 'free_top': [], 'free_bot': [], 'load': [],
        'fem_mae': [], 'fem_max_err': [], 'steps': []
    }
    
    # ---------------------------------------------------------
    # 2. STAGE 1: SOAP PRE-TRAINING
    # ---------------------------------------------------------
    print("Starting SOAP Pretraining...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_SOAP):
        optimizer_soap.zero_grad()
        
        # Adaptive Sampling (RAD)
        # Periodically re-sample points based on where the error (residual) is highest.
        # This focuses the training on "hard" regions.
        if epoch % 500 == 0 and epoch > 0:
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        # Compute Loss and Backpropagate
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_soap.step()
        scheduler.step()  # Update learning rate
        
        # Log History
        soap_history['total'].append(loss_val.item())
        soap_history['pde'].append(losses['pde'].item())
        soap_history['bc_sides'].append(losses['bc_sides'].item())
        soap_history['free_top'].append(losses['free_top'].item())
        soap_history['free_bot'].append(losses['free_bot'].item())
        soap_history['load'].append(losses['load'].item())
        
        # Print Status every 100 epochs
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            
            # Compute FEM error validation
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
                      f"PDE: {losses['pde']:.6f} | BC: {losses['bc_sides']:.6f} | "
                      f"Load: {losses['load']:.6f} | LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} | Time: {step_duration:.4f}s")
            else:
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | Load: {losses['load']:.6f} | "
                      f"LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"SOAP Pretraining Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # ---------------------------------------------------------
    # 3. STAGE 2: L-BFGS FINE-TUNING
    # ---------------------------------------------------------
    # We use a custom interface to SciPy's L-BFGS optimizer because PyTorch's
    # built-in L-BFGS implementation can be unstable or lack features like line search.
    print(f"Starting SciPy SSBFGS Fine-Tuning ({config.SS_BFGS_VARIANT})...")
    if scipy_patch.ensure_scipy_bfgs_patch():
        print("Applied local SciPy optimize patch for method_bfgs support.")

    param_device = next(pinn.parameters()).device
    param_dtype = next(pinn.parameters()).dtype

    # Helper functions to bridge PyTorch tensors and SciPy flat numpy arrays
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

    num_bfgs_steps = config.EPOCHS_SSBFGS
    print(f"Running {num_bfgs_steps} SSBFGS outer steps.")
    print("Resampling with residual-based adaptive sampling each outer step.")

    initial_weights = parameters_to_vector(pinn.parameters()).detach().cpu().numpy().astype(np.float64, copy=False)
    hess_inv0 = np.eye(initial_weights.size, dtype=np.float64)

    # L-BFGS Outer Loop
    # We step the L-BFGS optimizer manually in a loop to allow for periodic
    # resampling (RAD) and monitoring.
    for i in range(num_bfgs_steps):
        # 1. Resample Data (Adaptive Sampling)
        residuals = physics.compute_residuals(pinn, training_data, device)
        training_data = data.get_data(prev_data=training_data, residuals=residuals)

        step_start = time.time()
        
        # 2. Run Optimization Step (minimize calls the optimizer)
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

        # Update Inverse Hessian approximation for next step
        hess_inv0 = getattr(result, "hess_inv", None)
        if isinstance(hess_inv0, np.ndarray):
            hess_inv0 = 0.5 * (hess_inv0 + hess_inv0.T)
            try:
                cholesky(hess_inv0)
            except LinAlgError:
                hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)
        else:
            hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)

        # 3. Compute final losses for logging
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
                  f"Load: {losses['load'].item():.6e} | FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"SSBFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"Load: {losses['load'].item():.6e} | Time: {step_end - step_start:.4f}s")

        # Save model at every SSBFGS step
        torch.save(pinn.state_dict(), "pinn_model.pth")
            
    # ---------------------------------------------------------
    # 4. FINALIZE
    # ---------------------------------------------------------
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), "pinn_model.pth")
    loss_history = {'soap': soap_history, 'ssbfgs': ssbfgs_history}
    np.save("loss_history.npy", loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
