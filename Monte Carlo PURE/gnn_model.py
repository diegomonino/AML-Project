"""
Graph Neural Network models for CYGNO classification.
Uses DynamicEdgeConv (DGCNN-style) for adaptive neighbor aggregation.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, global_max_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


class DynamicGraphNN(torch.nn.Module):
    """
    Dynamic Graph Neural Network using DynamicEdgeConv.

    This model recomputes k-NN neighbors in feature space at each layer,
    which is especially useful for sparse/irregular point clouds like
    particle tracks of varying density (sparse ER vs dense NR).
    """

    def __init__(self, input_features, hidden_dim=64, k=5, dropout=0.3):
        """
        Args:
            input_features: number of input node features (e.g., 3: [intensity, y, x])
            hidden_dim: dimension of hidden layers
            k: number of neighbors for dynamic k-NN graph
            dropout: dropout rate
        """
        super().__init__()

        self.k = k
        self.dropout = dropout

        # Define MLP for edge features in DynamicEdgeConv
        # Input: concatenated source and destination features (2 * in_channels)
        # Output: hidden_dim
        mlp1 = Sequential(
            Linear(2 * input_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = DynamicEdgeConv(mlp1, k=k, aggr="mean")

        # Second layer: now input_features = hidden_dim
        mlp2 = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = DynamicEdgeConv(mlp2, k=k, aggr="mean")

        # Third layer: deeper representation
        mlp3 = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv3 = DynamicEdgeConv(mlp3, k=k, aggr="mean")

        # Global pooling + MLP classifier
        # After 3 layers, each node sees ~k^3 neighbors, capturing long-range structure
        self.fc1 = Linear(2 * hidden_dim, hidden_dim)  # 2x for mean+max pooling
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.out = Linear(hidden_dim // 2, 2)  # Binary classification

        self.bn_fc1 = BatchNorm1d(hidden_dim)
        self.bn_fc2 = BatchNorm1d(hidden_dim // 2)

    def forward(self, x, batch):
        """
        Forward pass.

        Args:
            x: node features (N, input_features)
            batch: batch assignment vector (N,) indicating which graph each node belongs to

        Returns:
            logits: (B, 2) classification logits
        """
        # Message passing layers with dynamic edge updates
        x = self.conv1(x, batch)  # (N, hidden_dim)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, batch)  # (N, hidden_dim)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, batch)  # (N, hidden_dim)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: combine mean and max to capture both average signal and peaks
        x_mean = global_mean_pool(x, batch)  # (B, hidden_dim)
        x_max = global_max_pool(x, batch)   # (B, hidden_dim)
        x_pool = torch.cat([x_mean, x_max], dim=1)  # (B, 2*hidden_dim)

        # Classification head
        x_out = self.fc1(x_pool)
        x_out = self.bn_fc1(x_out)
        x_out = F.relu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        x_out = self.fc2(x_out)
        x_out = self.bn_fc2(x_out)
        x_out = F.relu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        logits = self.out(x_out)  # (B, 2)
        return logits


class SimpleGraphNN(torch.nn.Module):
    """
    Simpler baseline: static k-NN graph (neighbor graph defined once at input).
    Use this if DynamicEdgeConv is too slow on Leonardo.
    """

    def __init__(self, input_features, hidden_dim=64, dropout=0.3):
        super().__init__()

        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(input_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.out = Linear(hidden_dim // 2, 2)

        self.bn_fc1 = BatchNorm1d(hidden_dim)
        self.bn_fc2 = BatchNorm1d(hidden_dim // 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x_out = self.fc1(x_pool)
        x_out = self.bn_fc1(x_out)
        x_out = F.relu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        x_out = self.fc2(x_out)
        x_out = self.bn_fc2(x_out)
        x_out = F.relu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        logits = self.out(x_out)
        return logits
