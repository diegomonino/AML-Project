"""
Optional: Pre-process images and cache graphs to disk.
Speeds up training on Leonardo if preprocessing is a bottleneck.

Usage:
  python prepare_cached_graphs.py

This generates:
  - train_graphs.pkl
  - val_graphs.pkl

Then modify GNN_PURE.py to load from pickle instead of computing on-the-fly.
"""

import os
import glob
import numpy as np
import pickle
from PIL import Image
from sklearn.utils import shuffle

from preprocessing import batch_preprocess, estimate_energy_from_image
from graph_utils import batch_to_pyg_dataset


# ===========================
# CONFIGURATION
# ===========================
ER_DIR = "./ER_label_0"
NR_DIR = "./NR_label_1"
IMG_SIZE = (128, 128)
VALIDATION_SPLIT = 0.2
SEED = 42

K_SIGMA = 3.0
MIN_COMPONENT_SIZE = 2
K_NEIGHBORS = 5


def load_all_images(folder, img_size=(128, 128)):
    """Load grayscale images from folder."""
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
    return np.stack(imgs, axis=0)


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    print("[INFO] Loading images...")
    er_images = load_all_images(ER_DIR, img_size=IMG_SIZE)
    nr_images = load_all_images(NR_DIR, img_size=IMG_SIZE)

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

    # Zero-suppression
    print("[INFO] Preprocessing training set...")
    X_train_sup, ped_train, mask_train = batch_preprocess(
        X_train, k_sigma=K_SIGMA, min_component_size=MIN_COMPONENT_SIZE
    )

    print("[INFO] Preprocessing validation set...")
    X_val_sup, ped_val, mask_val = batch_preprocess(
        X_val, k_sigma=K_SIGMA, min_component_size=MIN_COMPONENT_SIZE
    )

    # Compute energies
    energy_train = np.array([estimate_energy_from_image(img) for img in X_train_sup])
    energy_val = np.array([estimate_energy_from_image(img) for img in X_val_sup])

    # Build graphs
    print("[INFO] Building training graphs...")
    train_graphs = batch_to_pyg_dataset(
        X_train_sup, y_train, signal_masks=mask_train, k=K_NEIGHBORS
    )

    print("[INFO] Building validation graphs...")
    val_graphs = batch_to_pyg_dataset(
        X_val_sup, y_val, signal_masks=mask_val, k=K_NEIGHBORS
    )

    # Save to disk
    print("[INFO] Saving cached graphs...")
    with open("train_graphs.pkl", "wb") as f:
        pickle.dump({
            "graphs": train_graphs,
            "energy": energy_train,
            "labels": y_train,
            "pedestals": ped_train,
        }, f)

    with open("val_graphs.pkl", "wb") as f:
        pickle.dump({
            "graphs": val_graphs,
            "energy": energy_val,
            "labels": y_val,
            "pedestals": ped_val,
        }, f)

    print(f"[INFO] Cached graphs saved!")
    print(f"  train_graphs.pkl: {len(train_graphs)} graphs")
    print(f"  val_graphs.pkl: {len(val_graphs)} graphs")
    print("[INFO] To use in GNN_PURE.py, uncomment the pickle loading section")
