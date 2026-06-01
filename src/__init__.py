"""Tox21 Toxicity Classifier — source package.

Modules
-------
dataset  : DeepChem Tox21 loading (ECFP & GraphConv featurizers, NaN masking).
models   : ECFPClassifier (MLP) and GCNClassifier (PyG GraphConv).
losses   : MaskedBCEWithLogitsLoss, FocalLoss, pos_weight utilities.
train    : Common Trainer with early stopping & checkpointing.
evaluate : Per-task ROC-AUC / PR-AUC / F1 metrics.
"""

__version__ = "0.1.0"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

N_TASKS = len(TOX21_TASKS)
