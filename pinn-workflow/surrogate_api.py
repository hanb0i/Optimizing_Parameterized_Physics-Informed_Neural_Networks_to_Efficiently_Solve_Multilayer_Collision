import os
import sys
import torch
import numpy as np

# Add parent directory to path to support package imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from surrogate_workflow import config
from surrogate_workflow import surrogate

class ParametricSurrogate:
    """
    High-level API for interacting with the 5D Parametric Surrogate model.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_path = model_path or config.MODEL_PATH
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Surrogate model not found at {self.model_path}")
            
        # Load payload
        payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.x_min = payload["x_min"]
        self.x_max = payload["x_max"]
        self.y_min = payload["y_min"]
        self.y_max = payload["y_max"]
        self.param_names = payload["param_names"]
        
        # Build model
        self.model = surrogate.MLPRegressor(
            input_dim=len(self.param_names),
            output_dim=1,
            hidden_layers=payload["config"]["hidden_layers"],
            hidden_units=payload["config"]["hidden_units"],
            activation=payload["config"]["activation"]
        ).to(self.device)
        
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        print(f"Surrogate API: Loaded model from {self.model_path} ({self.device})")

    def predict(self, params_dict):
        """
        Predict peak vertical displacement magnitude |Uz_max| for given parameters.
        params_dict: { "E": val, "thickness": val, "restitution": val, "friction": val, "impact_velocity": val }
        """
        # 1. Arrange inputs in correct order
        x_raw = np.array([[params_dict[p] for p in self.param_names]], dtype=np.float32)
        
        # 2. Normalize
        x_norm = (x_raw - self.x_min) / (self.x_max - self.x_min + 1e-8)
        
        # 3. Inference
        with torch.no_grad():
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)
            y_norm = self.model(x_tensor).cpu().numpy()
            
        # 4. Denormalize
        y_raw = y_norm * (self.y_max - self.y_min) + self.y_min
        return float(y_raw.flatten()[0])

if __name__ == "__main__":
    # Quick test
    ps = ParametricSurrogate()
    sample = {
        "E": 1.0, 
        "thickness": 0.1, 
        "restitution": 0.5, 
        "friction": 0.3, 
        "impact_velocity": 1.0
    }
    val = ps.predict(sample)
    print(f"Prediction for baseline: {val:.4f}")
