"""Tox21 dataset loading & featurization.

Provides two parallel views of the Tox21 multi-label toxicity benchmark:

1. ECFP  : Morgan/circular fingerprints (radius=2, 2048 bits) -> dense tensors
           for the MLP classifier.
2. Graph : MolGraphConvFeaturizer -> PyTorch Geometric `Data` objects for the
           GraphConv classifier.

Both use DeepChem's default *scaffold* split (train/valid/test) and preserve
the NaN missing-label mask via DeepChem's per-task weight matrix `w`
(w == 0  <=>  label was originally NaN/missing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from .__init__ import TOX21_TASKS, N_TASKS
except Exception:  # pragma: no cover - allow running as a flat script
    TOX21_TASKS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]
    N_TASKS = len(TOX21_TASKS)


# -----------------------------------------------------------------------------
# Raw DeepChem loaders
# -----------------------------------------------------------------------------
def load_tox21_ecfp(radius: int = 2, size: int = 2048, reload: bool = True):
    """Load Tox21 with circular (ECFP) fingerprints.

    Returns
    -------
    tasks : list[str]
    datasets : (train, valid, test) DeepChem NumpyDataset objects
    transformers : list of DeepChem transformers
    """
    import deepchem as dc

    featurizer = dc.feat.CircularFingerprint(radius=radius, size=size)
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer, splitter="scaffold", reload=reload
    )
    return tasks, datasets, transformers


def load_tox21_graph(reload: bool = True):
    """Load Tox21 with MolGraphConvFeaturizer (for PyTorch Geometric)."""
    import deepchem as dc

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer, splitter="scaffold", reload=reload
    )
    return tasks, datasets, transformers


# -----------------------------------------------------------------------------
# ECFP -> PyTorch tensors / DataLoader
# -----------------------------------------------------------------------------
@dataclass
class ECFPSplit:
    X: "np.ndarray"   # (N, n_bits) float32
    y: "np.ndarray"   # (N, n_tasks) float32, NaN where missing
    w: "np.ndarray"   # (N, n_tasks) float32 mask (1=valid, 0=missing)
    ids: "np.ndarray"  # SMILES strings


def _dc_to_ecfp_split(ds) -> ECFPSplit:
    X = ds.X.astype("float32")
    y = ds.y.astype("float32")
    w = ds.w.astype("float32")
    # Encode missing labels explicitly as NaN so downstream masking is robust.
    y = np.where(w > 0, y, np.nan).astype("float32")
    return ECFPSplit(X=X, y=y, w=(w > 0).astype("float32"), ids=ds.ids)


def get_ecfp_splits(radius: int = 2, size: int = 2048, reload: bool = True):
    """Return (train, valid, test) ECFPSplit objects + task names."""
    tasks, (train, valid, test), _ = load_tox21_ecfp(radius, size, reload)
    return (
        _dc_to_ecfp_split(train),
        _dc_to_ecfp_split(valid),
        _dc_to_ecfp_split(test),
        list(tasks),
    )


def make_ecfp_dataloaders(
    batch_size: int = 128,
    radius: int = 2,
    size: int = 2048,
    reload: bool = True,
):
    """Build PyTorch DataLoaders for the ECFP/MLP model.

    Each batch yields (x, y, mask) where mask==0 marks missing labels.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tr, va, te, tasks = get_ecfp_splits(radius, size, reload)

    def _to_ds(split: ECFPSplit) -> "TensorDataset":
        x = torch.from_numpy(split.X)
        # NaN -> 0 in labels; the mask carries validity, loss ignores masked.
        y = torch.from_numpy(np.nan_to_num(split.y, nan=0.0))
        m = torch.from_numpy(split.w)
        return TensorDataset(x, y, m)

    train_loader = DataLoader(_to_ds(tr), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(_to_ds(va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_to_ds(te), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, tasks


# -----------------------------------------------------------------------------
# Graph -> PyTorch Geometric Data / DataLoader
# -----------------------------------------------------------------------------
def _dc_graph_to_pyg(graph, y_row: np.ndarray, w_row: np.ndarray):
    """Convert a single DeepChem GraphData to a PyG Data object."""
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(graph.node_features, dtype=torch.float32)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    edge_attr = None
    if getattr(graph, "edge_features", None) is not None:
        edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32)

    y = torch.tensor(np.nan_to_num(y_row, nan=0.0), dtype=torch.float32).view(1, -1)
    mask = torch.tensor(w_row, dtype=torch.float32).view(1, -1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.mask = mask
    return data


def _build_pyg_list(ds) -> List:
    y = ds.y.astype("float32")
    w = (ds.w.astype("float32") > 0).astype("float32")
    y = np.where(w > 0, y, np.nan)
    out = []
    for i, g in enumerate(ds.X):
        if g is None or getattr(g, "num_nodes", 1) == 0:
            continue
        out.append(_dc_graph_to_pyg(g, y[i], w[i]))
    return out


def make_graph_dataloaders(batch_size: int = 64, reload: bool = True):
    """Build PyTorch Geometric DataLoaders for the GraphConv model.

    Each batch is a PyG `Batch`; `batch.y` is (B, n_tasks) and `batch.mask`
    is the (B, n_tasks) validity mask.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    tasks, (train, valid, test), _ = load_tox21_graph(reload)

    train_list = _build_pyg_list(train)
    valid_list = _build_pyg_list(valid)
    test_list = _build_pyg_list(test)

    train_loader = PyGDataLoader(train_list, batch_size=batch_size, shuffle=True)
    valid_loader = PyGDataLoader(valid_list, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_list, batch_size=batch_size, shuffle=False)

    # Infer node feature dimension for model construction.
    node_dim = train_list[0].x.shape[1] if train_list else 30
    return train_loader, valid_loader, test_loader, list(tasks), node_dim


# -----------------------------------------------------------------------------
# Helpers for EDA
# -----------------------------------------------------------------------------
def label_matrix(reload: bool = True) -> Tuple["np.ndarray", List[str], "np.ndarray"]:
    """Return (y, tasks, smiles) over the full dataset for EDA, NaN preserved."""
    tasks, (train, valid, test), _ = load_tox21_ecfp(reload=reload)
    y = np.concatenate([train.y, valid.y, test.y], axis=0).astype("float32")
    w = np.concatenate([train.w, valid.w, test.w], axis=0).astype("float32")
    ids = np.concatenate([train.ids, valid.ids, test.ids], axis=0)
    y = np.where(w > 0, y, np.nan)
    return y, list(tasks), ids


if __name__ == "__main__":
    # Smoke test (requires deepchem installed).
    tr, va, te, tasks = get_ecfp_splits()
    print("Tasks:", tasks)
    print("Train X:", tr.X.shape, "y:", tr.y.shape, "missing:", np.isnan(tr.y).sum())
