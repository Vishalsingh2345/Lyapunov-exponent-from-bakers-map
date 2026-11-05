import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the test set

d = np.load(os.path.join(HERE, "bakers_map_test.npz"))
Xte = torch.tensor(d["X"], dtype=torch.float32).unsqueeze(1)
Yte = torch.tensor(d["Y"], dtype=torch.float32)
print(f"Test set: {Xte.shape}")

test_loader = DataLoader(TensorDataset(Xte, Yte), batch_size=100, shuffle=False)

# 2. Rebuild the SAME architecture used in training

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
weights_path = os.path.join(HERE, "bakers_lenet_weights.pt")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"Loaded weights from {weights_path}")


# 3. Run predictions on the test set

criterion = nn.HuberLoss(delta=0.6)
test_loss_sum = 0.0
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        test_loss_sum += loss.item() * xb.size(0)
        all_preds.append(pred.cpu())
        all_true.append(yb.cpu())

test_loss = test_loss_sum / len(test_loader.dataset)
all_preds = torch.cat(all_preds).numpy()
all_true  = torch.cat(all_true).numpy()

print(f"\nTest loss (Huber): {test_loss:.4f}")
mae = np.mean(np.abs(all_preds - all_true), axis=0)
print(f"Mean Absolute Error  -> lambda1: {mae[0]:.4f}   lambda2: {mae[1]:.4f}")

print("\nSample predictions (true -> predicted):")
for i in range(min(5, len(all_true))):
    print(f"  lambda1: {all_true[i,0]:.3f} -> {all_preds[i,0]:.3f}   "
          f"lambda2: {all_true[i,1]:.3f} -> {all_preds[i,1]:.3f}")


# 4. Plot true vs. predicted (scatter)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(all_true[:, 0], all_preds[:, 0], alpha=0.5, s=15)
lims = [all_true[:, 0].min(), all_true[:, 0].max()]
axes[0].plot(lims, lims, "r--", label="perfect prediction")
axes[0].set_xlabel("True lambda1"); axes[0].set_ylabel("Predicted lambda1")
axes[0].set_title(f"lambda1 (MAE = {mae[0]:.4f})"); axes[0].legend()

axes[1].scatter(all_true[:, 1], all_preds[:, 1], alpha=0.5, s=15, color="green")
lims = [all_true[:, 1].min(), all_true[:, 1].max()]
axes[1].plot(lims, lims, "r--", label="perfect prediction")
axes[1].set_xlabel("True lambda2"); axes[1].set_ylabel("Predicted lambda2")
axes[1].set_title(f"lambda2 (MAE = {mae[1]:.4f})"); axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(HERE, "test_predictions_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"\nSaved plot to {plot_path}")