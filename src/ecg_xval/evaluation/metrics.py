"""
Auditable evaluation metrics with explicit handling of masked and non-estimable classes.
Corresponds to Section 7 (Evaluation and Analysis) of the research proposal.
"""
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


NOT_ESTIMABLE = "not estimable"


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eval_mask: np.ndarray,
    thresholds: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes per-class and macro metrics over eligible records only.
    Classes without both positive and negative outcomes are labelled 'not estimable'
    and are never silently averaged over incompatible class sets.
    """
    per_class: Dict[str, Dict[str, Any]] = {}
    evaluable_f1s: List[float] = []
    all_classes_evaluable = True

    for idx, cls_name in enumerate(class_names):
        valid = eval_mask[:, idx] == 1
        yt = y_true[valid, idx]
        yp = y_prob[valid, idx]
        yhat = (yp >= thresholds[idx]).astype(int)

        n_evaluable = int(valid.sum())
        n_pos = int(yt.sum())
        n_neg = n_evaluable - n_pos

        if n_evaluable == 0 or n_pos == 0 or n_neg == 0:
            all_classes_evaluable = False
            per_class[cls_name] = {
                "n_evaluable": n_evaluable,
                "n_positive": n_pos,
                "precision": NOT_ESTIMABLE,
                "recall": NOT_ESTIMABLE,
                "f1": NOT_ESTIMABLE,
                "auprc": NOT_ESTIMABLE,
                "auroc": NOT_ESTIMABLE,
            }
            continue

        f1_val = float(f1_score(yt, yhat, zero_division=0))
        evaluable_f1s.append(f1_val)
        per_class[cls_name] = {
            "n_evaluable": n_evaluable,
            "n_positive": n_pos,
            "precision": float(precision_score(yt, yhat, zero_division=0)),
            "recall": float(recall_score(yt, yhat, zero_division=0)),
            "f1": f1_val,
            "auprc": float(average_precision_score(yt, yp)),
            "auroc": float(roc_auc_score(yt, yp)),
        }

    macro_f1 = float(np.mean(evaluable_f1s)) if (all_classes_evaluable and evaluable_f1s) else NOT_ESTIMABLE
    return {
        "macro_f1_shared_set": macro_f1,
        "all_shared_classes_estimable": all_classes_evaluable,
        "per_class": per_class,
    }
