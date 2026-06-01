"""Model definitions: ECFP-MLP and GraphConv classifiers for Tox21."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ECFP MLP classifier
# -----------------------------------------------------------------------------
class ECFPClassifier(nn.Module):
    """MLP over ECFP fingerprints.

    Linear(2048->512)->BN->ReLU->Dropout ->
    Linear(512->128) ->BN->ReLU->Dropout ->
    Linear(128->12)
    """

    def __init__(self, in_dim: int = 2048, n_tasks: int = 12, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits (B, n_tasks)


# -----------------------------------------------------------------------------
# GraphConv (GCN) classifier
# -----------------------------------------------------------------------------
class GCNClassifier(nn.Module):
    """3x GCNConv -> global mean pool -> Linear head.

    GCNConv(in->h) -> BN -> ReLU
    GCNConv(h->h)  -> BN -> ReLU
    GCNConv(h->h)  -> BN -> ReLU
    global_mean_pool -> Dropout -> Linear(h->n_tasks)
    """

    def __init__(self, node_dim: int = 30, hidden: int = 128,
                 n_tasks: int = 12, dropout: float = 0.3):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_tasks)

    def forward(self, data) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)          # (B, hidden)
        x = self.dropout(x)
        return self.head(x)                       # raw logits (B, n_tasks)


def build_model(kind: str, **kwargs) -> nn.Module:
    """Factory: 'ecfp' -> ECFPClassifier, 'gcn' -> GCNClassifier."""
    kind = kind.lower()
    if kind in ("ecfp", "mlp"):
        return ECFPClassifier(**kwargs)
    if kind in ("gcn", "graph", "graphconv"):
        return GCNClassifier(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")
