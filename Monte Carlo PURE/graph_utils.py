"""
Build k-NN graphs from preprocessed images.
Each signal pixel becomes a node; edges connect k-nearest neighbors.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist


def image_to_point_cloud(image, signal_mask=None):
    """
    Convert a 2D image to a point cloud (sparse representation).

    Args:
        image: 2D array (H, W), intensity values (already zero-suppressed ideally)
        signal_mask: 2D boolean mask. If None, all non-zero pixels are signal.

    Returns:
        points: (N, 2) array of (y, x) coordinates (in pixel space)
        features: (N, 1) array of intensities
        N: number of points
    """
    if signal_mask is None:
        signal_mask = image > 0

    # Find coordinates of signal pixels
    y_coords, x_coords = np.where(signal_mask)

    if len(y_coords) == 0:
        # Empty image - return dummy node
        points = np.array([[0.0, 0.0]])
        features = np.array([[0.0]])
    else:
        points = np.stack([y_coords, x_coords], axis=1).astype(np.float32)
        features = image[signal_mask].reshape(-1, 1).astype(np.float32)

    return points, features, len(y_coords)


def normalize_coordinates(points, image_shape=(128, 128)):
    """
    Normalize point coordinates to [-1, 1] range.

    Args:
        points: (N, 2) array of (y, x) coordinates
        image_shape: (H, W) shape of original image

    Returns:
        normalized: (N, 2) normalized coordinates in [-1, 1]
    """
    H, W = image_shape
    normalized = points.copy()
    normalized[:, 0] = 2 * (points[:, 0] / (H - 1)) - 1  # y -> [-1, 1]
    normalized[:, 1] = 2 * (points[:, 1] / (W - 1)) - 1  # x -> [-1, 1]
    return normalized


def build_knn_graph(points, features, k=5, normalize_coords=True):
    """
    Build a k-NN graph from point cloud.

    Args:
        points: (N, 2) array of (y, x) coordinates
        features: (N, F) array of node features (e.g., intensity)
        k: number of nearest neighbors
        normalize_coords: if True, normalize coordinates to [-1, 1]

    Returns:
        edge_index: (2, E) torch tensor of source->dest edges (undirected)
        node_features: (N, F+2) tensor of [intensity, normalized_y, normalized_x]
    """
    N = len(points)

    # Handle degenerate case (single node)
    if N <= 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_feat = torch.cat([
            torch.from_numpy(features),
            torch.from_numpy(normalize_coordinates(points, (128, 128)))
        ], dim=1).float()
        return edge_index, node_feat

    # Normalize coordinates for distance computation
    if normalize_coords:
        points_normalized = normalize_coordinates(points)
    else:
        points_normalized = points

    # Compute pairwise distances
    distances = cdist(points_normalized, points_normalized, metric='euclidean')

    # For each node, find k nearest neighbors (excluding itself)
    edges_src = []
    edges_dst = []

    for i in range(N):
        # Sort by distance, exclude self (distance 0)
        sorted_neighbors = np.argsort(distances[i])
        # Take the k nearest (indices 1:k+1 to skip self at 0)
        neighbors = sorted_neighbors[1:min(k+1, N)]

        # Add edges (both directions for undirected graph)
        for j in neighbors:
            edges_src.append(i)
            edges_dst.append(j)
            edges_src.append(j)  # Reverse direction
            edges_dst.append(i)

    # Remove duplicate edges (can happen in undirected conversion)
    edges_set = set()
    unique_src, unique_dst = [], []
    for s, d in zip(edges_src, edges_dst):
        edge = tuple(sorted([s, d]))  # Canonical form
        if edge not in edges_set:
            edges_set.add(edge)
            unique_src.append(s)
            unique_dst.append(d)

    edge_index = torch.tensor([unique_src, unique_dst], dtype=torch.long)

    # Build node features: [intensity, normalized_y, normalized_x]
    points_normalized_tensor = torch.from_numpy(normalize_coordinates(points)).float()
    features_tensor = torch.from_numpy(features).float()
    node_features = torch.cat([features_tensor, points_normalized_tensor], dim=1)

    return edge_index, node_features


def image_to_pyg_data(image, label, signal_mask=None, k=5, normalize_coords=True):
    """
    Convert a preprocessed image to a PyG Data object.

    Args:
        image: 2D array (H, W)
        label: scalar label (0 or 1)
        signal_mask: 2D boolean mask
        k: number of nearest neighbors
        normalize_coords: normalize coordinates in graph building

    Returns:
        data: torch_geometric.data.Data object
    """
    # Convert image to point cloud
    points, features, n_points = image_to_point_cloud(image, signal_mask)

    # Build k-NN graph
    edge_index, node_features = build_knn_graph(
        points, features, k=k, normalize_coords=normalize_coords
    )

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        n_nodes=torch.tensor(n_points, dtype=torch.long),
        energy=torch.tensor(np.sum(image), dtype=torch.float32),  # Total signal
    )

    return data


def batch_to_pyg_dataset(images, labels, signal_masks=None, k=5):
    """
    Convert a batch of preprocessed images to a list of PyG Data objects.

    Args:
        images: (N, H, W) array
        labels: (N,) array of labels
        signal_masks: (N, H, W) boolean array or None
        k: number of nearest neighbors

    Returns:
        dataset: list of torch_geometric.data.Data objects
    """
    N = len(images)
    dataset = []

    if signal_masks is None:
        signal_masks = [None] * N

    for i in range(N):
        data = image_to_pyg_data(
            images[i],
            labels[i],
            signal_mask=signal_masks[i],
            k=k,
            normalize_coords=True,
        )
        dataset.append(data)

        if (i + 1) % max(1, N // 10) == 0:
            print(f"[graph_build] {i+1}/{N} graphs constructed")

    return dataset
