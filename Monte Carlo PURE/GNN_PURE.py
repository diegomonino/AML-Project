"""
Graph Neural Network Training on CYGNO Images with Zero-Suppression.
Parallel to CNN_PURE.py but using dynamic graph convolutions.

Usage (local dev):
  python GNN_PURE.py

Usage (Leonardo):
  See sbatch_pure_gnn.txt
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import InMemoryDataset

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy.stats import ks_2samp

# Import custom modules
from preprocessing import batch_preprocess, estimate_energy_from_image
from graph_utils import batch_to_pyg_dataset
from gnn_model import DynamicGraphNN


# ===========================
# 1. CONFIGURATION
# ===========================
ER_DIR = "./ER_label_0"          # Standard / background-like
NR_DIR = "./NR_label_1"          # AmBe / signal-like
IMG_SIZE = (128, 128)
VALIDATION_SPLIT = 0.2
SEED = 42

# Zero-suppression params
K_SIGMA = 3.0                     # Threshold = pedestal + k_sigma * sigma
MIN_COMPONENT_SIZE = 2            # Min pixel cluster to keep

# Graph params
K_NEIGHBORS = 5                   # k-NN graph parameter
BATCH_SIZE = 32

# Training params
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10
HIDDEN_DIM = 64
DROPOUT = 0.3

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================
# 2. UTILITIES
# ===========================
def load_all_images(folder, img_size=(128, 128)):
    """Load grayscale images from folder, normalize to [0,1]."""
    exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("L")
            im = im.resize(img_size, Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        except Exception as e:
            print(f"[load_all_images] skipping {p}: {e}")
    if len(imgs) == 0:
        raise RuntimeError(f"No images loaded from {folder}")
    return np.stack(imgs, axis=0)  # (N, H, W)


class GraphDataset(InMemoryDataset):
    """PyG dataset wrapper for graphs."""
    def __init__(self, data_list):
        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


# ===========================
# 3. PIPELINE
# ===========================
print("[INFO] Loading images...")
er_images = load_all_images(ER_DIR, img_size=IMG_SIZE)   # label 0
nr_images = load_all_images(NR_DIR, img_size=IMG_SIZE)   # label 1

print(f"[INFO] Loaded {len(er_images)} ER and {len(nr_images)} NR images")

# Create labels
X_er = er_images
y_er = np.zeros(len(er_images), dtype=np.int64)
X_nr = nr_images
y_nr = np.ones(len(nr_images), dtype=np.int64)

X_all = np.concatenate([X_er, X_nr], axis=0)
y_all = np.concatenate([y_er, y_nr], axis=0)

# Shuffle
X_all, y_all = shuffle(X_all, y_all, random_state=SEED)

# Train/val split
n_total = len(X_all)
n_val = int(n_total * VALIDATION_SPLIT)
X_val, y_val = X_all[:n_val], y_all[:n_val]
X_train, y_train = X_all[n_val:], y_all[n_val:]

print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

# Zero-suppression preprocessing
print("[INFO] Applying zero-suppression...")
X_train_sup, ped_train, mask_train = batch_preprocess(
    X_train, k_sigma=K_SIGMA, min_component_size=MIN_COMPONENT_SIZE
)
X_val_sup, ped_val, mask_val = batch_preprocess(
    X_val, k_sigma=K_SIGMA, min_component_size=MIN_COMPONENT_SIZE
)

# Compute energy for later binning
energy_train = np.array([estimate_energy_from_image(img) for img in X_train_sup])
energy_val = np.array([estimate_energy_from_image(img) for img in X_val_sup])

print(f"[INFO] Train energy: mean={energy_train.mean():.2f}, std={energy_train.std():.2f}")
print(f"[INFO] Val energy: mean={energy_val.mean():.2f}, std={energy_val.std():.2f}")

# Build k-NN graphs
print("[INFO] Building k-NN graphs...")
train_graphs = batch_to_pyg_dataset(
    X_train_sup, y_train, signal_masks=mask_train, k=K_NEIGHBORS
)
val_graphs = batch_to_pyg_dataset(
    X_val_sup, y_val, signal_masks=mask_val, k=K_NEIGHBORS
)

# Wrap in PyG dataset
train_dataset = GraphDataset(train_graphs)
val_dataset = GraphDataset(val_graphs)

# Data loaders
train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ===========================
# 4. MODEL & TRAINING
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = DynamicGraphNN(
    input_features=3,  # [intensity, norm_y, norm_x]
    hidden_dim=HIDDEN_DIM,
    k=K_NEIGHBORS,
    dropout=DROPOUT,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

best_val_auc = 0
patience_counter = 0


def train_epoch():
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_dataset)


@torch.no_grad()
def validate():
    """Validation and compute AUC."""
    model.eval()
    y_true = []
    y_score = []
    val_loss = 0

    for batch in val_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        val_loss += loss.item() * batch.num_graphs

        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true.extend(batch.y.cpu().numpy())
        y_score.extend(probs)

    val_loss /= len(val_dataset)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    val_auc = auc(fpr, tpr)

    return val_loss, val_auc, y_true, y_score


print("[INFO] Starting training...")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val AUC':<12} {'Patience':<8}")
print("-" * 60)

for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss, val_auc, y_true_val, y_score_val = validate()

    print(
        f"{epoch+1:<6} {train_loss:<12.4f} {val_loss:<12.4f} {val_auc:<12.4f} {patience_counter:<8}"
    )

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), "final_model_GNN_PURE.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

print(f"[INFO] Best validation AUC: {best_val_auc:.4f}")


# ===========================
# 5. EVALUATION
# ===========================
print("\n[INFO] Loading best model for evaluation...")
model.load_state_dict(torch.load("final_model_GNN_PURE.pt"))
model.eval()

# Full validation predictions
val_loss, val_auc, y_true_val, y_score_val = validate()
print(f"Validation AUC: {val_auc:.4f}")

# Confusion matrix and report
y_pred_val = (y_score_val > 0.5).astype(int)
cm = confusion_matrix(y_true_val, y_pred_val)
print("\nConfusion Matrix (Val):")
print(cm)
print("\nClassification Report (Val):")
print(classification_report(y_true_val, y_pred_val, target_names=["ER", "NR"]))

# ===========================
# 6. ENERGY-BINNED EVALUATION
# ===========================
print("\n[INFO] Energy-binned evaluation...")

n_bins = 5
energy_bins = np.percentile(energy_val, np.linspace(0, 100, n_bins + 1))
bin_edges = [(energy_bins[i], energy_bins[i + 1]) for i in range(n_bins)]

print(f"\n{'Bin':<8} {'E range':<20} {'N':<6} {'AUC':<8}")
print("-" * 42)

for i, (e_low, e_high) in enumerate(bin_edges):
    mask = (energy_val >= e_low) & (energy_val <= e_high)
    if np.sum(mask) == 0:
        continue

    y_bin = y_true_val[mask]
    score_bin = y_score_val[mask]

    if len(np.unique(y_bin)) > 1:  # Need both classes to compute AUC
        fpr, tpr, _ = roc_curve(y_bin, score_bin)
        bin_auc = auc(fpr, tpr)
    else:
        bin_auc = np.nan

    print(f"Bin {i:<4} [{e_low:8.2f}, {e_high:8.2f}] {np.sum(mask):<6} {bin_auc:<8.4f}")

# ===========================
# 7. VISUALIZATION
# ===========================
print("\n[INFO] Saving plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ROC curve
fpr, tpr, _ = roc_curve(y_true_val, y_score_val)
axes[0].plot(fpr, tpr, 'b-', label=f'GNN (AUC={val_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (Validation)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Score distributions
axes[1].hist(y_score_val[y_true_val == 0], bins=30, alpha=0.6, label="ER (true 0)")
axes[1].hist(y_score_val[y_true_val == 1], bins=30, alpha=0.6, label="NR (true 1)")
axes[1].set_xlabel("Model Score")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Score Distributions (Validation)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gnn_evaluation.png", dpi=150, bbox_inches='tight')
print("[INFO] Saved gnn_evaluation.png")

print("\n[INFO] Training complete!")
print(f"Model saved: final_model_GNN_PURE.pt")
