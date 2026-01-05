import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, activation):
        super().__init__()
        layers = []
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(act)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(act)
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
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
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss + config.MIN_DELTA < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.PATIENCE
        else:
            patience_left -= 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: train={train_loss:.6e}, val={val_loss:.6e}")

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict(model, x_norm, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
        y = model(x_t).cpu().numpy().reshape(-1)
    return y
