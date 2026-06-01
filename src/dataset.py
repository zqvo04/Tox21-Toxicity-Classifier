"""Tox21 dataset loading & featurization (robust, RDKit-based).

This module deliberately avoids DeepChem's MoleculeNet featurization pipeline,
which crashes on recent NumPy ("inhomogeneous shape") when a molecule fails to
featurize. Instead we:

1. Download the canonical MoleculeNet Tox21 CSV (same data DeepChem uses).
2. Featurize with RDKit directly, skipping/handling invalid molecules:
     - ECFP  : Morgan fingerprint (radius=2, 2048 bits) -> dense float array.
     - Graph : atom/bond features -> PyTorch Geometric `Data` objects.
3. Reproduce DeepChem's *scaffold* split (Bemis-Murcko, largest groups -> train)
   so the train/valid/test partition matches the standard benchmark protocol.

Missing labels (NaN in the CSV) are preserved and exposed via a binary mask
(1 = label present, 0 = missing) so downstream losses/metrics can ignore them.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# RDKit prints many parse warnings for the few malformed Tox21 SMILES; silence
# them so notebook output stays readable. (Invalid molecules are skipped anyway.)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

try:
    from .__init__ import TOX21_TASKS, N_TASKS
except Exception:  # pragma: no cover - allow running as a flat script
    TOX21_TASKS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]
    N_TASKS = len(TOX21_TASKS)

TOX21_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
)
_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".tox21_cache")


# -----------------------------------------------------------------------------
# Raw CSV loading
# -----------------------------------------------------------------------------
def load_tox21_dataframe(cache_dir: str = _DEFAULT_CACHE):
    """Download (and cache) the MoleculeNet Tox21 CSV as a pandas DataFrame."""
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tox21.csv.gz")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(TOX21_URL, path)
    df = pd.read_csv(path)
    return df


def _clean_smiles(df) -> Tuple[List[str], "np.ndarray"]:
    """Keep only rows whose SMILES parse into a valid RDKit molecule.

    Returns (smiles_list, label_matrix) with labels as float (NaN = missing).
    """
    from rdkit import Chem

    smiles, labels = [], []
    y_cols = TOX21_TASKS
    for _, row in df.iterrows():
        smi = row["smiles"]
        if not isinstance(smi, str) or Chem.MolFromSmiles(smi) is None:
            continue
        smiles.append(smi)
        labels.append([row[c] for c in y_cols])
    y = np.array(labels, dtype="float32")  # NaN preserved for missing labels
    return smiles, y


# -----------------------------------------------------------------------------
# Scaffold split (Bemis-Murcko), reproducing DeepChem's ScaffoldSplitter
# -----------------------------------------------------------------------------
def _scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(
    smiles: List[str], frac_train: float = 0.8, frac_valid: float = 0.1
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic scaffold split: largest scaffold groups go to train first."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[_scaffold(smi)].append(i)

    # Sort scaffold sets by size (largest first), ties broken by first index.
    ordered = sorted(groups.values(), key=lambda idx: (len(idx), -idx[0]),
                     reverse=True)

    n = len(smiles)
    n_train, n_valid = frac_train * n, frac_valid * n
    train, valid, test = [], [], []
    for grp in ordered:
        if len(train) + len(grp) <= n_train:
            train += grp
        elif len(valid) + len(grp) <= n_valid:
            valid += grp
        else:
            test += grp
    return train, valid, test


# -----------------------------------------------------------------------------
# Featurizers (RDKit)
# -----------------------------------------------------------------------------
def ecfp_featurize(smiles: List[str], radius: int = 2, n_bits: int = 2048):
    """Morgan/ECFP fingerprints as a dense (N, n_bits) float32 array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    X = np.zeros((len(smiles), n_bits), dtype="float32")
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype="int8")
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


# Atom feature vocabulary for the graph featurizer.
_ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Other"]
_HYBRIDIZATION = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "other"]
NODE_FEAT_DIM = (
    len(_ATOM_LIST) + len(_HYBRIDIZATION) + 5
)  # symbol + hybridization + [degree, formal_charge, num_Hs, aromatic, in_ring]


def _onehot(value, choices):
    vec = [0.0] * len(choices)
    vec[choices.index(value) if value in choices else len(choices) - 1] = 1.0
    return vec


def _atom_features(atom) -> List[float]:
    feats = _onehot(atom.GetSymbol(), _ATOM_LIST)
    feats += _onehot(str(atom.GetHybridization()), _HYBRIDIZATION)
    feats += [
        atom.GetDegree() / 4.0,
        atom.GetFormalCharge(),
        atom.GetTotalNumHs() / 4.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
    ]
    return feats


def mol_to_graph(smiles: str, y_row: np.ndarray, w_row: np.ndarray):
    """Convert a SMILES string to a PyTorch Geometric `Data` object, or None."""
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float32)

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]  # undirected -> both directions
    if len(src) == 0:  # single-atom molecule: add a self-loop
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor(np.nan_to_num(y_row, nan=0.0), dtype=torch.float32).view(1, -1)
    mask = torch.tensor(w_row, dtype=torch.float32).view(1, -1)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.mask = mask
    return data


# -----------------------------------------------------------------------------
# ECFP split container + builders
# -----------------------------------------------------------------------------
@dataclass
class ECFPSplit:
    X: "np.ndarray"   # (N, n_bits) float32
    y: "np.ndarray"   # (N, n_tasks) float32, NaN where missing
    w: "np.ndarray"   # (N, n_tasks) float32 mask (1 valid, 0 missing)
    ids: "np.ndarray"  # SMILES strings


def _make_split(X, y, smiles, idx) -> ECFPSplit:
    yy = y[idx]
    w = (~np.isnan(yy)).astype("float32")
    return ECFPSplit(X=X[idx], y=yy, w=w, ids=np.array(smiles, dtype=object)[idx])


def get_ecfp_splits(radius: int = 2, size: int = 2048, reload: bool = True):
    """Return (train, valid, test) ECFPSplit objects + task names."""
    df = load_tox21_dataframe()
    smiles, y = _clean_smiles(df)
    X = ecfp_featurize(smiles, radius=radius, n_bits=size)
    tr, va, te = scaffold_split(smiles)
    return (
        _make_split(X, y, smiles, tr),
        _make_split(X, y, smiles, va),
        _make_split(X, y, smiles, te),
        list(TOX21_TASKS),
    )


def make_ecfp_dataloaders(
    batch_size: int = 128, radius: int = 2, size: int = 2048, reload: bool = True
):
    """PyTorch DataLoaders for the ECFP/MLP model: batches of (x, y, mask)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tr, va, te, tasks = get_ecfp_splits(radius, size, reload)

    def _ds(s: ECFPSplit):
        x = torch.from_numpy(s.X)
        y = torch.from_numpy(np.nan_to_num(s.y, nan=0.0))
        m = torch.from_numpy(s.w)
        return TensorDataset(x, y, m)

    train_loader = DataLoader(_ds(tr), batch_size=batch_size, shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(_ds(va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_ds(te), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, tasks


# -----------------------------------------------------------------------------
# Graph builders
# -----------------------------------------------------------------------------
def _graph_list(smiles, y, idx):
    out = []
    w = (~np.isnan(y)).astype("float32")
    for i in idx:
        g = mol_to_graph(smiles[i], y[i], w[i])
        if g is not None:
            out.append(g)
    return out


def make_graph_dataloaders(batch_size: int = 64, reload: bool = True):
    """PyTorch Geometric DataLoaders for the GraphConv model.

    Returns (train, valid, test, tasks, node_dim). Each batch exposes
    `batch.y` (B, n_tasks) and `batch.mask` (B, n_tasks).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    df = load_tox21_dataframe()
    smiles, y = _clean_smiles(df)
    tr, va, te = scaffold_split(smiles)

    train_list = _graph_list(smiles, y, tr)
    valid_list = _graph_list(smiles, y, va)
    test_list = _graph_list(smiles, y, te)

    train_loader = PyGDataLoader(train_list, batch_size=batch_size, shuffle=True,
                                 drop_last=True)
    valid_loader = PyGDataLoader(valid_list, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_list, batch_size=batch_size, shuffle=False)

    node_dim = train_list[0].x.shape[1] if train_list else NODE_FEAT_DIM
    return train_loader, valid_loader, test_loader, list(TOX21_TASKS), node_dim


# -----------------------------------------------------------------------------
# Convenience wrappers used by the notebooks
# -----------------------------------------------------------------------------
class _GraphDataset:
    """Lightweight stand-in exposing .y and .w for pos_weight computation."""

    def __init__(self, y: np.ndarray):
        self.y = y
        self.w = (~np.isnan(y)).astype("float32")


def load_tox21_graph(reload: bool = True):
    """Return (tasks, (train_ds, valid_ds, test_ds), transformers=[]).

    The datasets expose `.y` (NaN-preserved) and `.w` (mask) — enough for
    pos_weight computation in the notebooks.
    """
    df = load_tox21_dataframe()
    smiles, y = _clean_smiles(df)
    tr, va, te = scaffold_split(smiles)
    datasets = (_GraphDataset(y[tr]), _GraphDataset(y[va]), _GraphDataset(y[te]))
    return list(TOX21_TASKS), datasets, []


def get_test_smiles() -> "np.ndarray":
    """SMILES of the test split, aligned to saved predictions order."""
    df = load_tox21_dataframe()
    smiles, _ = _clean_smiles(df)
    _, _, te = scaffold_split(smiles)
    return np.array(smiles, dtype=object)[te]


def label_matrix(reload: bool = True) -> Tuple["np.ndarray", List[str], "np.ndarray"]:
    """Return (y, tasks, smiles) over the full dataset for EDA, NaN preserved."""
    df = load_tox21_dataframe()
    smiles, y = _clean_smiles(df)
    return y, list(TOX21_TASKS), np.array(smiles, dtype=object)


if __name__ == "__main__":
    tr, va, te, tasks = get_ecfp_splits()
    print("Tasks:", tasks)
    print("Split sizes:", tr.X.shape[0], va.X.shape[0], te.X.shape[0])
    print("Train X:", tr.X.shape, "missing labels:", int(np.isnan(tr.y).sum()))
