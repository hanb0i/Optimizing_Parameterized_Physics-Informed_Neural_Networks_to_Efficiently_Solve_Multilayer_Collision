import torch
import torch.nn as nn
import numpy as np

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, fourier_dim, fourier_scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * fourier_scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units, activation="tanh", fourier_dim=0, fourier_scale=1.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        if fourier_dim > 0:
            self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale)
            current_dim = fourier_dim * 2
        else:
            self.fourier = None
            current_dim = input_dim

        layers = []
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU()

        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_units))
            layers.append(act)
            current_dim = hidden_units

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier is not None:
            x = self.fourier(x)
        return self.net(x)

def train_model(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = config.PATIENCE
    history = {"train": [], "val": []}

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss + config.MIN_DELTA < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.PATIENCE
        else:
            patience_left -= 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: train={train_loss:.6e}, val={val_loss:.6e}")
        if patience_left <= 0:
            print("Early stopping.")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history

def predict(model, x_norm, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
        return model(x_t).cpu().numpy().reshape(-1)
