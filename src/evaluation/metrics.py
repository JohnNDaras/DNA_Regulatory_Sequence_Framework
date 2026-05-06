import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred))}
    try: out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception: out["auroc"] = float("nan")
    try: out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception: out["auprc"] = float("nan")
    return out
