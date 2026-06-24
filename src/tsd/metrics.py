from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .interestingness import LABEL_COL


def interestingness_metrics(labeled: pd.DataFrame, spans: pd.DataFrame, target_col: str) -> dict:
    y = pd.to_numeric(labeled[target_col], errors="coerce")
    label = labeled[LABEL_COL].astype(bool)
    return {
        "n_rows": int(len(labeled)),
        "n_interesting_rows": int(label.sum()),
        "support": float(label.mean()) if len(label) else 0.0,
        "target_mean_all": float(y.mean()),
        "target_mean_interesting": float(y[label].mean()) if label.any() else None,
        "target_mean_shift": float(y[label].mean() - y.mean()) if label.any() else None,
        "n_spans": int(len(spans)),
        "n_interesting_spans": int(spans[LABEL_COL].sum()) if len(spans) else 0,
        "span_support": float(spans[LABEL_COL].mean()) if len(spans) else 0.0,
    }


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out

