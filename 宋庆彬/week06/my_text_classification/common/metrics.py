"""统一评估函数：BERT/LLM 共用"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, id2name: dict = None):
    """返回 acc, macro_f1, per-class 报告, 混淆矩阵。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    result = {"accuracy": acc, "macro_f1": macro_f1}

    if id2name is not None:
        label_ids = sorted(id2name.keys())
        target_names = [id2name[i] for i in label_ids]
        result["report"] = classification_report(
            y_true, y_pred, labels=label_ids, target_names=target_names, zero_division=0
        )
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=label_ids)

    return result
