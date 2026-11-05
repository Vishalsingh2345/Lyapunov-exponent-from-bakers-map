import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# 1. Settings

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # remaining 0.15 goes to test
SEED = 123                          # separate seed just for the shuffle/split


# 2. Load the raw dataset

raw_path = os.path.join(HERE, "bakers_map_raw.npz")
d = np.load(raw_path)
X, Y, alphas = d["X"], d["Y"], d["alpha"]
N = X.shape[0]
print(f"Loaded raw dataset: {N} samples")

# 3. Shuffle and split

rng = np.random.default_rng(SEED)
idx = rng.permutation(N)

n_train = int(TRAIN_FRAC * N)
n_val   = int(VAL_FRAC * N)

train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

def subset(ix):
    return X[ix], Y[ix], alphas[ix]

Xtr, Ytr, atr = subset(train_idx)
Xva, Yva, ava = subset(val_idx)
Xte, Yte, ate = subset(test_idx)

print(f"Train: {Xtr.shape[0]} samples")
print(f"Val:   {Xva.shape[0]} samples")
print(f"Test:  {Xte.shape[0]} samples")


# 4. Save the three splits

np.savez(os.path.join(HERE, "bakers_map_train.npz"), X=Xtr, Y=Ytr, alpha=atr)
np.savez(os.path.join(HERE, "bakers_map_val.npz"),   X=Xva, Y=Yva, alpha=ava)
np.savez(os.path.join(HERE, "bakers_map_test.npz"),  X=Xte, Y=Yte, alpha=ate)


# 5. Save human-readable metadata (no series, just labels)

meta_path = os.path.join(HERE, "bakers_map_metadata.csv")
with open(meta_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "alpha", "lambda1", "lambda2"])
    for a, y in zip(atr, Ytr):
        w.writerow(["train", a, y[0], y[1]])
    for a, y in zip(ava, Yva):
        w.writerow(["val", a, y[0], y[1]])
    for a, y in zip(ate, Yte):
        w.writerow(["test", a, y[0], y[1]])

print(f"\nSaved train/val/test .npz files and {meta_path}")
print("Next step: run 3_train_model.py")