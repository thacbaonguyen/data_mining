"""
splitter.py — Tìm phép chia tối ưu cho Decision Tree

Xử lý 2 loại thuộc tính:
    - Continuous: sort → thử mọi midpoint → chọn GINI/Entropy nhỏ nhất
    - Categorical: thử mọi binary partition → chọn tốt nhất

Reference: docs/main/data_mining.md L1230-1270
Reference: docs/main/DMDWv1.2.md L575-591
"""

import numpy as np
from .criteria import gini_index, entropy, information_gain, gain_ratio, weighted_impurity


def _is_categorical(x):
    """Kiểm tra xem một cột có phải categorical không."""
    if hasattr(x, 'dtype'):
        return x.dtype == object or x.dtype.name == 'category'
    return isinstance(x[0], str)


def find_best_split_continuous(X_col, y, criterion='gini'):
    """
    Tìm ngưỡng split tối ưu cho thuộc tính continuous.
    
    Thuật toán:
    1. Sort giá trị thuộc tính
    2. Với mỗi cặp giá trị liên tiếp, tính midpoint
    3. Chia dữ liệu tại midpoint, tính impurity
    4. Chọn midpoint có impurity nhỏ nhất
    
    Ref: data_mining.md L1250-1270 (Continuous Attributes)
    Ref: DMDWv1.2.md L575-591 (efficient computation)
    
    Parameters
    ----------
    X_col : array-like
        Giá trị 1 cột thuộc tính (continuous)
    y : array-like
        Nhãn lớp
    criterion : str
        'gini' hoặc 'entropy'
        
    Returns
    -------
    float
        Giá trị ngưỡng tốt nhất
    float
        Impurity tốt nhất (hoặc gain tốt nhất)
    """
    X_col = np.array(X_col, dtype=float)
    y = np.array(y)
    
    # Sort theo giá trị thuộc tính
    sorted_indices = np.argsort(X_col)
    X_sorted = X_col[sorted_indices]
    y_sorted = y[sorted_indices]
    
    best_threshold = None
    best_impurity = float('inf')
    
    # Thử mọi midpoint giữa 2 giá trị liên tiếp khác nhau
    for i in range(1, len(X_sorted)):
        if X_sorted[i] == X_sorted[i - 1]:
            continue
        
        threshold = (X_sorted[i] + X_sorted[i - 1]) / 2.0
        
        y_left = y_sorted[:i]
        y_right = y_sorted[i:]
        
        imp = weighted_impurity([y_left, y_right], criterion)
        
        if imp < best_impurity:
            best_impurity = imp
            best_threshold = threshold
    
    return best_threshold, best_impurity


def find_best_split_categorical(X_col, y, criterion='gini'):
    """
    Tìm phép chia tối ưu cho thuộc tính categorical.
    
    Thuật toán (binary split):
    1. Lấy tất cả unique categories
    2. Thử mọi cách chia thành 2 nhóm
    3. Chọn cách chia có impurity nhỏ nhất
    
    Tối ưu: Nếu quá nhiều categories, dùng heuristic
    (sort categories theo tỷ lệ positive class)
    
    Ref: data_mining.md L1230-1245 (Nominal Attributes)
    
    Parameters
    ----------
    X_col : array-like
        Giá trị 1 cột thuộc tính (categorical)
    y : array-like
        Nhãn lớp
    criterion : str
        'gini' hoặc 'entropy'
        
    Returns
    -------
    set
        Tập categories thuộc nhánh trái
    float
        Impurity tốt nhất
    """
    X_col = np.array(X_col)
    y = np.array(y)
    
    categories = list(set(X_col))
    
    if len(categories) <= 1:
        return None, float('inf')
    
    best_cats_left = None
    best_impurity = float('inf')
    
    # Heuristic: sort categories theo tỷ lệ positive class
    # Rồi thử n-1 split points (tương tự continuous)
    # Đây là tối ưu cho binary classification
    
    # Tính tỷ lệ positive cho mỗi category
    labels = sorted(set(y))
    positive_label = labels[-1] if len(labels) > 1 else labels[0]
    
    cat_positive_rate = {}
    for cat in categories:
        mask = X_col == cat
        if np.sum(mask) > 0:
            cat_positive_rate[cat] = np.sum(y[mask] == positive_label) / np.sum(mask)
        else:
            cat_positive_rate[cat] = 0.0
    
    # Sort categories theo positive rate
    sorted_cats = sorted(categories, key=lambda c: cat_positive_rate[c])
    
    # Thử n-1 split points
    for i in range(1, len(sorted_cats)):
        cats_left = set(sorted_cats[:i])
        
        mask_left = np.isin(X_col, list(cats_left))
        y_left = y[mask_left]
        y_right = y[~mask_left]
        
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        
        imp = weighted_impurity([y_left, y_right], criterion)
        
        if imp < best_impurity:
            best_impurity = imp
            best_cats_left = cats_left
    
    return best_cats_left, best_impurity


def find_best_split(X, y, feature_names=None, categorical_features=None, criterion='gini'):
    """
    Tìm phép chia tối ưu trên tất cả thuộc tính.
    
    Parameters
    ----------
    X : numpy.ndarray (n_samples × n_features)
        Ma trận dữ liệu
    y : array-like
        Nhãn lớp
    feature_names : list of str, optional
        Tên các thuộc tính
    categorical_features : list of int, optional
        Indices của các thuộc tính categorical
    criterion : str
        'gini' hoặc 'entropy'
        
    Returns
    -------
    dict
        {
            'feature': int (index),
            'feature_name': str,
            'threshold': float (for continuous),
            'categories_left': set (for categorical),
            'is_categorical': bool,
            'impurity': float,
            'gain': float
        }
        hoặc None nếu không tìm được split hợp lệ
    """
    y = np.array(y)
    n_features = X.shape[1]
    
    if categorical_features is None:
        categorical_features = []
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    parent_impurity = gini_index(y) if criterion == 'gini' else entropy(y)
    
    best_split = None
    best_impurity = float('inf')
    
    for feat_idx in range(n_features):
        X_col = X[:, feat_idx]
        
        if feat_idx in categorical_features:
            cats_left, imp = find_best_split_categorical(X_col, y, criterion)
            if cats_left is not None and imp < best_impurity:
                best_impurity = imp
                best_split = {
                    'feature': feat_idx,
                    'feature_name': feature_names[feat_idx],
                    'threshold': None,
                    'categories_left': cats_left,
                    'is_categorical': True,
                    'impurity': imp,
                    'gain': parent_impurity - imp
                }
        else:
            # Continuous
            try:
                X_col_float = X_col.astype(float)
            except (ValueError, TypeError):
                # Nếu không convert được → coi là categorical
                cats_left, imp = find_best_split_categorical(X_col, y, criterion)
                if cats_left is not None and imp < best_impurity:
                    best_impurity = imp
                    best_split = {
                        'feature': feat_idx,
                        'feature_name': feature_names[feat_idx],
                        'threshold': None,
                        'categories_left': cats_left,
                        'is_categorical': True,
                        'impurity': imp,
                        'gain': parent_impurity - imp
                    }
                continue
            
            threshold, imp = find_best_split_continuous(X_col_float, y, criterion)
            if threshold is not None and imp < best_impurity:
                best_impurity = imp
                best_split = {
                    'feature': feat_idx,
                    'feature_name': feature_names[feat_idx],
                    'threshold': threshold,
                    'categories_left': None,
                    'is_categorical': False,
                    'impurity': imp,
                    'gain': parent_impurity - imp
                }
    
    return best_split
