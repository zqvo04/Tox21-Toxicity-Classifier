"""Evaluation metrics for masked multi-label Tox21 classification.

Computes per-task ROC-AUC, PR-AUC (average precision), and F1 (at a tunable
threshold), ignoring missing labels via the mask. Also returns the mean
ROC-AUC across the 12 tasks — the headline metric for the model comparison.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def _valid(y_col: np.ndarray, m_col: np.ndarray) -> np.ndarray:
    """Boolean index of valid (non-missing) entries for one task."""
    return m_col > 0


def per_task_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    tasks: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a per-task metrics DataFrame.

    Columns: task, n, n_pos, roc_auc, pr_auc, f1.
    Tasks with only one class present (after masking) get NaN AUC.
    """
    rows = []
    for j, task in enumerate(tasks):
        idx = _valid(y_true[:, j], mask[:, j])
        yt = y_true[idx, j].astype(int)
        yp = y_prob[idx, j]
        n = int(idx.sum())
        n_pos = int(yt.sum())

        roc = pr = f1 = np.nan
        if n > 0 and 0 < n_pos < n:  # need both classes
            roc = roc_auc_score(yt, yp)
            pr = average_precision_score(yt, yp)
            f1 = f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
        rows.append(
            dict(task=task, n=n, n_pos=n_pos, roc_auc=roc, pr_auc=pr, f1=f1)
        )
    df = pd.DataFrame(rows)
    return df


def summary_metrics(df: pd.DataFrame) -> dict:
    """Aggregate mean ROC-AUC / PR-AUC / F1 across tasks (ignoring NaN)."""
    return {
        "mean_roc_auc": float(np.nanmean(df["roc_auc"].values)),
        "mean_pr_auc": float(np.nanmean(df["pr_auc"].values)),
        "mean_f1": float(np.nanmean(df["f1"].values)),
    }


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    tasks: List[str],
    threshold: float = 0.5,
    name: str = "model",
) -> Tuple[pd.DataFrame, dict]:
    """Convenience: return (per_task_df, summary_dict)."""
    df = per_task_metrics(y_true, y_prob, mask, tasks, threshold)
    summ = summary_metrics(df)
    summ["model"] = name
    return df, summ


# -----------------------------------------------------------------------------
# Plotting helpers (used by notebooks)
# -----------------------------------------------------------------------------
def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    tasks: List[str],
    save_path: Optional[str] = None,
    title: str = "ROC curves (per task)",
):
    """Plot a 3x4 grid of per-task ROC curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    n = len(tasks)
    ncols, nrows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for j, task in enumerate(tasks):
        ax = axes[j]
        idx = mask[:, j] > 0
        yt = y_true[idx, j].astype(int)
        yp = y_prob[idx, j]
        if 0 < yt.sum() < len(yt):
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def find_false_negatives(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    ids: np.ndarray,
    task_idx: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return SMILES of false negatives for a task (true=1, pred=0)."""
    idx = mask[:, task_idx] > 0
    yt = y_true[idx, task_idx].astype(int)
    yp = y_prob[idx, task_idx]
    smiles = np.asarray(ids)[idx]
    fn = (yt == 1) & (yp < threshold)
    return pd.DataFrame(
        dict(smiles=smiles[fn], true=yt[fn], prob=yp[fn])
    ).sort_values("prob")
