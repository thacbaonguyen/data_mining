"""
metrics.py — Các metrics đánh giá mô hình phân loại

Implements:
    - Confusion Matrix
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - F1-Score
    - Specificity

Tất cả implement from scratch, không dùng sklearn.metrics.

Reference: docs/main/data_mining.md L1634-1700
"""

import numpy as np
from collections import defaultdict


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Tính Confusion Matrix.
    
    |              | Predicted Pos | Predicted Neg |
    |--------------|---------------|---------------|
    | Actual Pos   | TP            | FN            |
    | Actual Neg   | FP            | TN            |
    
    Ref: data_mining.md L1634-1639
    
    Parameters
    ----------
    y_true : array-like
        Nhãn thực tế
    y_pred : array-like
        Nhãn dự đoán
    labels : list, optional
        Danh sách nhãn theo thứ tự mong muốn
        
    Returns
    -------
    numpy.ndarray
        Ma trận confusion (n_classes × n_classes)
    list
        Danh sách labels
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    n = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        i = label_to_idx.get(true, -1)
        j = label_to_idx.get(pred, -1)
        if i >= 0 and j >= 0:
            cm[i][j] += 1
    
    return cm, labels


def accuracy(y_true, y_pred):
    """
    Tính Accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Hạn chế: misleading khi class imbalance
    Ví dụ: Class 0 = 9990, Class 1 = 10 → predict all 0 → acc = 99.9%
    
    Ref: data_mining.md L1641-1648
    
    Parameters
    ----------
    y_true, y_pred : array-like
        
    Returns
    -------
    float
        Giá trị accuracy (0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred, positive_label=None):
    """
    Tính Precision.
    
    Precision = TP / (TP + FP)
    
    "Trong các trường hợp dự đoán Positive, bao nhiêu % đúng?"
    
    Ref: data_mining.md L1681-1683
    
    Parameters
    ----------
    y_true, y_pred : array-like
    positive_label : any, optional
        Nhãn positive. Nếu None, lấy nhãn thứ 2 theo thứ tự sort.
        
    Returns
    -------
    float
        Giá trị precision (0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if positive_label is None:
        labels = sorted(set(y_true))
        positive_label = labels[-1] if len(labels) > 1 else labels[0]
    
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def recall(y_true, y_pred, positive_label=None):
    """
    Tính Recall (Sensitivity).
    
    Recall = TP / (TP + FN)
    
    "Trong các trường hợp thực sự Positive, bao nhiêu % được phát hiện?"
    
    Ref: data_mining.md L1685-1687
    
    Parameters
    ----------
    y_true, y_pred : array-like
    positive_label : any, optional
        
    Returns
    -------
    float
        Giá trị recall (0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if positive_label is None:
        labels = sorted(set(y_true))
        positive_label = labels[-1] if len(labels) > 1 else labels[0]
    
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true, y_pred, positive_label=None):
    """
    Tính F1-Score (harmonic mean of Precision and Recall).
    
    F1 = 2 × Precision × Recall / (Precision + Recall)
       = 2TP / (2TP + FP + FN)
    
    Ref: data_mining.md L1689-1691
    
    Parameters
    ----------
    y_true, y_pred : array-like
    positive_label : any, optional
        
    Returns
    -------
    float
        Giá trị F1-score (0-1)
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)


def classification_report(y_true, y_pred):
    """
    Tạo báo cáo phân loại đầy đủ.
    
    Parameters
    ----------
    y_true, y_pred : array-like
        
    Returns
    -------
    dict
        Báo cáo cho từng class + overall
    str
        Bản text format đẹp
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    
    report = {}
    for label in labels:
        p = precision(y_true, y_pred, positive_label=label)
        r = recall(y_true, y_pred, positive_label=label)
        f1 = f1_score(y_true, y_pred, positive_label=label)
        support = np.sum(y_true == label)
        report[label] = {
            'precision': p,
            'recall': r,
            'f1-score': f1,
            'support': int(support)
        }
    
    # Overall accuracy
    acc = accuracy(y_true, y_pred)
    report['accuracy'] = acc
    
    # Format output
    header = f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}\n"
    lines = header + "-" * 55 + "\n"
    for label in labels:
        r = report[label]
        lines += f"{str(label):>15s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10d}\n"
    lines += "-" * 55 + "\n"
    lines += f"{'accuracy':>15s} {'':>10s} {'':>10s} {acc:>10.4f} {len(y_true):>10d}\n"
    
    return report, lines
