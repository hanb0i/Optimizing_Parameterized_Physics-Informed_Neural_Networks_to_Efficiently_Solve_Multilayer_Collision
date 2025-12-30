
import torch
import torch.optim as optim
import numpy as np
import time
import os

import pinn_config as config
import data
import model
import physics
import matplotlib.pyplot as plt

def train(callback=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    
    # Initialize Optimizers
    optimizer_adam = optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    
    # Data Container
    training_data = data.get_data()
    
    # History
    loss_history = []
    
    print("Starting Adam Training...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_ADAM):
        optimizer_adam.zero_grad()
        
        # Periodic data refresh (optional, computationally expensive to re-sample every Step)
        if epoch % 1000 == 0 and epoch > 0:
            training_data = data.get_data()
            
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_adam.step()
        
        loss_history.append(loss_val.item())
        
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                  f"PDE: {losses['pde']:.6f} | BC: {losses['bc_sides']:.6f} | "
                  f"Load: {losses['load']:.6f} | Interface: {losses['interface']:.6f} | "
                  f"Time: {step_duration:.4f}s")
            
        # Visualization Callback (Every 500 epochs)
        if callback and epoch % 500 == 0:
            callback(f"adam_epoch_{epoch}", pinn, device)
            
    print(f"Adam Training Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # L-BFGS Training
    print("Starting L-BFGS Training...")
    optimizer_lbfgs = optim.LBFGS(pinn.parameters(), 
                                  max_iter=100, 
                                  history_size=50, 
                                  line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_val, _ = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        return loss_val
        
    num_lbfgs_steps = config.EPOCHS_LBFGS // 20
    print(f"Running {num_lbfgs_steps} L-BFGS outer steps.")
    
    for i in range(num_lbfgs_steps): 
        step_start = time.time()
        loss_val = optimizer_lbfgs.step(closure)
        step_end = time.time()
        loss_history.append(loss_val.item())
        
        # Print every step to see progress since total steps is small (5)
        print(f"L-BFGS Step {i}: Loss: {loss_val.item():.6f} | Time: {step_end - step_start:.4f}s")
        
        # Visualization Callback (Every 10 steps)
        if callback and i % 10 == 0:
            callback(f"lbfgs_step_{i}", pinn, device)
            
    # Save Model
    torch.save(pinn.state_dict(), "pinn_model.pth")
    np.save("loss_history.npy", np.array(loss_history))
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
