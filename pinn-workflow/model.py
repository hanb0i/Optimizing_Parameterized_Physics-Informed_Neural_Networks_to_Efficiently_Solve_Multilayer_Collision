import torch
import torch.nn as nn
import pinn_config as config

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================

class LayerNet(nn.Module):
    """
    A Fully Connected Neural Network (MLP) to approximate displacement u(x,y,z).
    Enforces Dirichlet boundary conditions on the side walls via a hard constraint mask.
    """
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Input: x, y, z (3 spatial coordinates)
        input_dim = 3
        current_dim = input_dim
        
        # Input Layer
        layers.append(nn.Linear(current_dim, hidden_units))
        layers.append(activation)
        
        # Hidden Layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        # Output Layer: u_x, u_y, u_z (3 displacement components)
        layers.append(nn.Linear(hidden_units, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Weight initialization (Xavier Normal for Tanh activation)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass with Hard Constraints.
        
        Args:
            x: Tensor of shape (N, 3) containing columns [x, y, z]
            
        Returns:
            Displacement tensor (N, 3)
        """
        # Feature Scaling
        # x, y are in [0, 1]. z is in [0, 0.1].
        # We scale z by 10.0 to bring it to [0, 1] range for better training stability.
        # x shape: (N, 3)
        x_scaled = torch.cat([x[:, 0:1], x[:, 1:2], x[:, 2:3] * 10.0], dim=1)
        
        # Raw Network Output
        u_raw = self.net(x_scaled)
        
        # ==========================================
        # HARD BOUNDARY CONSTRAINTS
        # ==========================================
        # We want u = 0 at x=0, x=1, y=0, y=1 (Clamped sides).
        # We achieve this by multiplying the output by a mask function M(x,y).
        # Mask M(x,y) = x * (1-x) * y * (1-y)
        # This function is exactly 0 at the boundaries.
        
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        
        # Normalize the mask so its maximum value in the domain is approx 1.0.
        # Max of x(1-x) is 0.25 at x=0.5. Max of y(1-y) is 0.25 at y=0.5.
        # Max of M is 0.25 * 0.25 = 0.0625.
        # Multiply by 16.0 to make the peak value 1.0.
        mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
        
        # Apply mask and Global Output Scaling
        # OUTPUT_SCALE (3.55) is crucial to match the physical displacement magnitude.
        return u_raw * mask * config.OUTPUT_SCALE

class MultiLayerPINN(nn.Module):
    """
    Wrapper class for the PINN. 
    Currently supports a single layer but designed to be extensible for multi-layer laminates.
    """
    def __init__(self):
        super().__init__()
        # Single network for homogeneous material (or first layer)
        self.layer = LayerNet()
        
    def forward(self, x, layer_idx=0):
        # layer_idx kept for compatibility/extensions
        return self.layer(x)

    def predict_all(self, x):
        # Direct prediction helper
        return self.layer(x)
