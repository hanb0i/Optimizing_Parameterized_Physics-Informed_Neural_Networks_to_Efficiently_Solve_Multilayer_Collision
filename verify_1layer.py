
import sys
import os
import torch
import numpy as np

# Ensure paths are correct
sys.path.append(os.path.join(os.getcwd(), 'pinn-workflow'))
import pinn_config as config
import model
from surrogate_api import ParametricSurrogate

def verify_single_layer():
    print("=== Single-Layer Accuracy Verification ===")
    
    # 1. Test Reference Case: E=5.0, t=0.1
    # We compare against the known FEA peak from walkthrough/verification_results
    # walkthrough says for t=0.1:
    # E=1.0: FEA -2.83, PINN -2.75
    # E=5.0: FEA -0.566, PINN -0.640
    
    ps = ParametricSurrogate()
    
    test_cases = [
        {"E": 1.0, "thickness": 0.1, "restitution": 0.5, "friction": 0.3, "impact_velocity": 1.0, "target": 2.833},
        {"E": 5.0, "thickness": 0.1, "restitution": 0.5, "friction": 0.3, "impact_velocity": 1.0, "target": 0.566},
        {"E": 10.0, "thickness": 0.1, "restitution": 0.5, "friction": 0.3, "impact_velocity": 1.0, "target": 0.283}
    ]
    
    for tc in test_cases:
        pred = ps.predict(tc)
        err = abs(pred - tc["target"])
        rel_err = (err / tc["target"]) * 100
        print(f"E={tc['E']}, t={tc['thickness']}: Pred={pred:.4f}, Target={tc['target']:.4f}, Error={rel_err:.2f}%")

if __name__ == "__main__":
    verify_single_layer()
