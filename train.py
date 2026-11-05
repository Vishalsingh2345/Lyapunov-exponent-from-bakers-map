import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)


# 1. Load train / val data

def load_split(name):
    d = np.load(os.path.join(HERE, f"bakers_map_{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32).unsqueeze(1)  # (N,1,1000)
    Y = torch.tensor(d["Y"], dtype=torch.float32)                # (N,2)
    return X, Y

Xtr, Ytr = load_split("train")
Xva, Yva = load_split("val")
print(f"Train: {Xtr.shape}, Val: {Xva.shape}")

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=100, shuffle=False)


# 2. CNN architecture

class LEsNet(nn.Module):
    def __init__(self, n_outputs=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 15, kernel_size=10, stride=1, dilation=2, padding="same")
        self.conv2 = nn.Conv1d(15, 30, kernel_size=5, stride=1, dilation=4, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(30, n_outputs)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

model = LEsNet(n_outputs=2).to(device)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")

# 3. Loss and optimizer

criterion = nn.HuberLoss(delta=0.6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-5)


# 4. Training loop with early stopping

N_EPOCHS = 300
PATIENCE = 30
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0
train_losses, val_losses = [], []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * xb.size(0)
    train_loss = train_loss_sum / len(train_loader.dataset)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss_sum += loss.item() * xb.size(0)
    val_loss = val_loss_sum / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(best_state)
print(f"\nBest validation loss: {best_val_loss:.4f}")


# 5. Plot the loss curve

plt.figure(figsize=(7, 4.5))
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Huber loss")
plt.title("Training progress")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "train_val_loss_curve.png"), dpi=150)
print("Saved train_val_loss_curve.png")

# 6. Save the trained model

weights_path = os.path.join(HERE, "bakers_lenet_weights.pt")
torch.save(model.state_dict(), weights_path)
print(f"Saved trained model weights to {weights_path}")
print("Next step: run 4_test_model.py")