# GNN Pipeline for CYGNO Classification

This setup provides a complete Graph Neural Network pipeline for distinguishing ER vs NR events in CYGNO detector images, with a focus on improving low-energy discrimination via zero-suppression and sparse graph representation.

## Overview

**Problem**: The CNN achieves poor metrics at low energies because tracks are sparse and drowned in 95% background noise.

**Solution**: 
1. **Zero-suppression**: Remove background pixels using `pedestal + k·σ` thresholding
2. **Sparse point cloud**: Surviving signal pixels → nodes in a graph
3. **k-NN graph**: Adaptive neighbor connectivity captures track topology
4. **DynamicEdgeConv GNN**: Recomputes neighborhoods in feature space at each layer

## Files

### Core Modules
- **`preprocessing.py`**: Zero-suppression pipeline
  - `estimate_pedestal()`: Compute noise floor per image
  - `zero_suppress()`: Apply threshold & pedestal subtraction
  - `connected_component_filter()`: Remove isolated noise spikes
  - `batch_preprocess()`: Batch processing wrapper

- **`graph_utils.py`**: Graph construction
  - `image_to_point_cloud()`: Extract signal pixels as nodes
  - `build_knn_graph()`: Build k-NN adjacency from point cloud
  - `image_to_pyg_data()`: Convert image → PyG Data object
  - `batch_to_pyg_dataset()`: Batch processing

- **`gnn_model.py`**: Neural network architectures
  - `DynamicGraphNN`: Main model (DynamicEdgeConv-based, recommended)
  - `SimpleGraphNN`: Fallback (GCN-based, faster but less adaptive)

- **`GNN_PURE.py`**: Main training script
  - Full pipeline from images → graphs → training → evaluation
  - Energy-binned ROC AUC evaluation

### Scripts
- **`sbatch_pure_gnn.txt`**: SLURM job script for Leonardo

---

## Installation

### Local Development
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric scikit-learn scipy pillow matplotlib
```

### Leonardo (in sbatch script)
The script automatically installs PyTorch Geometric. Ensure `python/3.10.20` is loaded.

---

## Usage

### Local Testing
```bash
cd "Monte Carlo PURE"
python GNN_PURE.py
```

Expected output:
- `final_model_GNN_PURE.pt` — trained model weights
- `gnn_evaluation.png` — ROC curve + score distributions

### Leonardo Submission
```bash
sbatch sbatch_pure_gnn.txt
```

The script will:
1. Copy dataset and code to SCRATCH (fast storage)
2. Preprocess images (zero-suppression)
3. Build k-NN graphs
4. Train the GNN on GPU
5. Save results back to WORK directory

Monitor with: `squeue -u $USER`

---

## Configuration Parameters

### Zero-Suppression (`GNN_PURE.py`)
```python
K_SIGMA = 3.0              # Threshold = pedestal + k_sigma * sigma
MIN_COMPONENT_SIZE = 2     # Min pixels in connected cluster
```

**Interpretation**:
- `k_sigma=3.0`: Keep pixels >3σ above noise floor (~99.7% noise exclusion)
- `k_sigma=5.0`: Stricter (only brightest signal)
- Adjust down if too much signal is suppressed; up if noise remains

### Graph Building
```python
K_NEIGHBORS = 5            # Number of nearest neighbors in k-NN
```

**Trade-offs**:
- `k=3-5`: Sparse, topology-sensitive (good for sparse ER tracks)
- `k=7-10`: Denser, captures broader neighborhood

### Training
```python
HIDDEN_DIM = 64            # Hidden layer dimension
DROPOUT = 0.3              # Dropout rate
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10              # Early stopping patience
```

---

## Understanding the Output

### Validation Metrics
- **Val AUC**: Overall discrimination power (target: >0.90 for low-E data)
- **Confusion Matrix**: TP, FP, TN, FN breakdown
- **Classification Report**: Per-class precision, recall, F1

### Energy-Binned Evaluation (Key Insight)
The script bins validation data by total signal intensity (energy proxy) and computes AUC per bin:

```
Bin 0     [E_low, E_mid1]  N_samples  AUC_bin_0
Bin 1     [E_mid1, E_mid2] N_samples  AUC_bin_1
...
```

**Why this matters**:
- **Low-E bins**: Where CNN struggles → GNN should improve
- **High-E bins**: Where CNN already works → GNN should maintain
- If GNN AUC drops in high-E bins, model may be overfit

### Plots
- **gnn_evaluation.png**:
  - Left: ROC curve (higher = better discrimination)
  - Right: Score distributions (ER and NR should separate well)

---

## Troubleshooting

### "No images loaded"
Check that directories `ER_label_0/` and `NR_label_1/` exist and contain PNG/JPG files.

### CUDA out of memory
Reduce `BATCH_SIZE` (e.g., 16 or 8) or `HIDDEN_DIM` (e.g., 32).

### Graphs are empty (all nodes filtered)
Increase `K_SIGMA` (e.g., 2.0 instead of 3.0) to keep more pixels. Check `energy_train` stats in output.

### Training very slow on Leonardo
- Use `SimpleGraphNN` instead of `DynamicGraphNN` (static edges, faster)
- Reduce `EPOCHS` or increase `PATIENCE` to exit early
- Pre-cache graphs before training (optional optimization)

---

## Next Steps: Comparison with CNN

To directly compare GNN vs CNN:

1. Run `CNN_PURE.py` and save predictions: `cnn_scores.npy`
2. Modify `GNN_PURE.py` to save: `gnn_scores.npy` + `energy_values.npy`
3. Run comparison script (to be added):
   ```python
   import numpy as np
   from sklearn.metrics import auc, roc_curve
   
   # Load predictions
   cnn_scores = np.load('cnn_scores.npy')
   gnn_scores = np.load('gnn_scores.npy')
   energy = np.load('energy_values.npy')
   y_true = np.load('y_true.npy')
   
   # Per-energy-bin AUC comparison
   for elow, ehigh in zip(energy_bins[:-1], energy_bins[1:]):
       mask = (energy >= elow) & (energy <= ehigh)
       cnn_auc = auc(*roc_curve(y_true[mask], cnn_scores[mask])[:2])
       gnn_auc = auc(*roc_curve(y_true[mask], gnn_scores[mask])[:2])
       print(f"E in [{elow:.1f}, {ehigh:.1f}]: CNN={cnn_auc:.3f}, GNN={gnn_auc:.3f}")
   ```

---

## Physics Intuition

**Why graphs help**:
- **NR (dense blobs)**: Many pixels clustered → high node density, compact graph
- **ER (sparse tracks)**: Few pixels along a line → low node density, extended graph
- **CNN**: Must process full 128×128 image, wastes capacity on 95% background
- **GNN**: Directly operates on 50-200 signal nodes, topology is the signal

**Low-energy regime**:
- NR and ER tracks are faint and look similar to CNN (high error)
- But track shapes (dense vs sparse) remain distinct
- GNN + zero-suppression amplifies shape differences by removing noise

---

## References

- **DynamicEdgeConv**: DGCNN (Wang et al., 2019) — https://arxiv.org/abs/1801.07829
- **CWoLa**: Classification Without Labels (Komiske et al., 2017) — https://arxiv.org/abs/1705.07218
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

## Contact & Debugging

For issues:
1. Check if data folders are correct
2. Inspect `energy_train` stats (should be non-zero)
3. Run `python -c "from preprocessing import *; print('OK')"` to test imports
4. On Leonardo, check `sinfo` and partition availability
