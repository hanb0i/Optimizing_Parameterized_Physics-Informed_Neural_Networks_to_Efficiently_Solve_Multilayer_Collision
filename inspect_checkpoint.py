
import torch
import os

path = "pinn-workflow/pinn_model.pth"
if os.path.exists(path):
    state = torch.load(path, map_location='cpu')
    print("Checkpoint Keys:")
    for k, v in state.items():
        print(f"{k}: {v.shape}")
else:
    print(f"File not found: {path}")

import sys
sys.path.append('pinn-workflow')
import model
pinn = model.MultiLayerPINN()
print("\nInitialized Model Keys:")
for k, v in pinn.state_dict().items():
    print(f"{k}: {v.shape}")
