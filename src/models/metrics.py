"""
Evaluation metrics for medical retinal imaging:
- Accuracy, Precision, Recall/Sensitivity, Specificity
- Macro & Weighted F1-score
- Multi-class One-vs-Rest AUROC
- Quadratic Weighted Kappa (QWK)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


def quadratic_weighted_kappa(y_true, y_pred):
    """Calculates Quadratic Weighted Kappa (standard for DR severity assessment)."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(y_true, y_pred, y_probs=None, num_classes=5):
    """
    Computes a comprehensive dictionary of clinical classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "quadratic_weighted_kappa": float(qwk),
    }

    # Per-class sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivities = []
    specificities = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    metrics["macro_sensitivity"] = float(np.mean(sensitivities))
    metrics["macro_specificity"] = float(np.mean(specificities))

    # AUROC calculation
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        try:
            if num_classes == 2:
                # Binary AUROC
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                # Multi-class OvR AUROC
                auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            metrics["auroc"] = float(auc)
        except Exception:
            metrics["auroc"] = None

    return metrics
