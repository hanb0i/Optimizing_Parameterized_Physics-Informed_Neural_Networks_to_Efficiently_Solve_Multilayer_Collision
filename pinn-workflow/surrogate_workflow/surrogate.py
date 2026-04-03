import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, fourier_dim: int, fourier_scale: float):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(input_dim, fourier_dim) * float(fourier_scale),
            requires_grad=False,
        )

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: int,
        activation: str = "tanh",
        fourier_dim: int = 0,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        self.fourier_dim = int(fourier_dim)
        if self.fourier_dim > 0:
            self.fourier = FourierFeatures(input_dim, self.fourier_dim, float(fourier_scale))
            current_dim = self.fourier_dim * 2
        else:
            self.fourier = None
            current_dim = input_dim

        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU()

        layers = []
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(current_dim, int(hidden_units)))
            layers.append(act)
            current_dim = int(hidden_units)
        layers.append(nn.Linear(current_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier is not None:
            x = self.fourier(x)
        return self.net(x)


def train_model(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.LEARNING_RATE))
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = int(config.PATIENCE)
    history = {"train": [], "val": []}
    has_val = hasattr(val_loader, "dataset") and len(val_loader.dataset) > 0
    loss_mode = str(getattr(config, "LOSS_MODE", "mse")).strip().lower()

    for epoch in range(int(config.MAX_EPOCHS)):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(device)
            else:
                xb, yb = batch
                wb = None
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            if loss_mode == "relative_mse" and wb is not None:
                loss = torch.mean(wb * (pred - yb) ** 2)
            else:
                loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        if has_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        xb, yb, wb = batch
                        wb = wb.to(device)
                    else:
                        xb, yb = batch
                        wb = None
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    if loss_mode == "relative_mse" and wb is not None:
                        loss = torch.mean(wb * (pred - yb) ** 2)
                    else:
                        loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = train_loss

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss + float(config.MIN_DELTA) < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(config.PATIENCE)
        else:
            patience_left -= 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: train={train_loss:.6e}, val={val_loss:.6e}")
        if patience_left <= 0:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, x_norm: np.ndarray, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
        return model(x_t).detach().cpu().numpy().reshape(-1)
