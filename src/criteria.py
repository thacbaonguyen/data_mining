"""
criteria.py — Tiêu chí phân nhánh cho Decision Tree

Implements:
    - GINI Index (used in CART)
    - Entropy (used in ID3)
    - Information Gain (used in ID3)
    - Gain Ratio (used in C4.5)
    - Classification Error

Reference: docs/main/data_mining.md Part 3, Sections 3.11.1–3.11.7
"""

import numpy as np
from collections import Counter


def gini_index(y):
    """
    Tính chỉ số GINI của một node.
    
    GINI(t) = 1 - Σ [p(i|t)]²
    
    - GINI = 0: tất cả thuộc 1 lớp (pure)
    - GINI = 1 - 1/n_classes: phân bố đều (impure nhất)
    
    Ref: data_mining.md L1305-1320
    
    Parameters
    ----------
    y : array-like
        Mảng nhãn lớp tại node
        
    Returns
    -------
    float
        Giá trị GINI index
    """
    if len(y) == 0:
        return 0.0
    
    counter = Counter(y)
    n = len(y)
    
    gini = 1.0
    for count in counter.values():
        p = count / n
        gini -= p ** 2
    
    return gini


def entropy(y):
    """
    Tính Entropy của một node.
    
    Entropy(t) = -Σ p(i|t) × log₂(p(i|t))
    
    - Entropy = 0: tất cả thuộc 1 lớp (pure)
    - Entropy = log₂(n_classes): phân bố đều (impure nhất)
    
    Ref: data_mining.md L1371-1385
    
    Parameters
    ----------
    y : array-like
        Mảng nhãn lớp tại node
        
    Returns
    -------
    float
        Giá trị Entropy
    """
    if len(y) == 0:
        return 0.0
    
    counter = Counter(y)
    n = len(y)
    
    ent = 0.0
    for count in counter.values():
        p = count / n
        if p > 0:
            ent -= p * np.log2(p)
    
    return ent


def classification_error(y):
    """
    Tính Classification Error của một node.
    
    Error(t) = 1 - max_i P(i|t)
    
    Ref: data_mining.md L1401-1406
    
    Parameters
    ----------
    y : array-like
        Mảng nhãn lớp tại node
        
    Returns
    -------
    float
        Giá trị Classification Error
    """
    if len(y) == 0:
        return 0.0
    
    counter = Counter(y)
    n = len(y)
    max_p = max(count / n for count in counter.values())
    
    return 1.0 - max_p


def information_gain(y_parent, y_children):
    """
    Tính Information Gain của một phép chia.
    
    IG = Entropy(parent) - Σ (nᵢ/n) × Entropy(childᵢ)
    
    - IG lớn → phép chia tốt
    - Dùng trong ID3
    - Nhược điểm: bias toward attributes with many values
    
    Ref: data_mining.md L1371-1395
    
    Parameters
    ----------
    y_parent : array-like
        Mảng nhãn lớp của node cha
    y_children : list of array-like
        Danh sách mảng nhãn lớp của các node con
        
    Returns
    -------
    float
        Giá trị Information Gain
    """
    n = len(y_parent)
    parent_entropy = entropy(y_parent)
    
    # Weighted average entropy of children
    weighted_child_entropy = 0.0
    for child in y_children:
        if len(child) > 0:
            weighted_child_entropy += (len(child) / n) * entropy(child)
    
    return parent_entropy - weighted_child_entropy


def split_info(y_children):
    """
    Tính SplitINFO cho Gain Ratio.
    
    SplitINFO = -Σ (nᵢ/n) × log₂(nᵢ/n)
    
    Ref: data_mining.md L1392
    
    Parameters
    ----------
    y_children : list of array-like
        Danh sách mảng nhãn lớp của các node con
        
    Returns
    -------
    float
        Giá trị SplitINFO
    """
    n = sum(len(child) for child in y_children)
    if n == 0:
        return 0.0
    
    si = 0.0
    for child in y_children:
        if len(child) > 0:
            p = len(child) / n
            if p > 0:
                si -= p * np.log2(p)
    
    return si


def gain_ratio(y_parent, y_children):
    """
    Tính Gain Ratio của một phép chia.
    
    GainRatio = InformationGain / SplitINFO
    
    - Khắc phục nhược điểm của IG (bias toward many values)
    - Dùng trong C4.5
    - SplitINFO penalizes splits with many small partitions
    
    Ref: data_mining.md L1385-1399
    
    Parameters
    ----------
    y_parent : array-like
        Mảng nhãn lớp của node cha
    y_children : list of array-like
        Danh sách mảng nhãn lớp của các node con
        
    Returns
    -------
    float
        Giá trị Gain Ratio
    """
    ig = information_gain(y_parent, y_children)
    si = split_info(y_children)
    
    # Tránh chia cho 0
    if si == 0:
        return 0.0
    
    return ig / si


def weighted_impurity(y_children, criterion='gini'):
    """
    Tính impurity có trọng số cho một phép chia.
    
    Weighted = Σ (nᵢ/n) × Impurity(childᵢ)
    
    Parameters
    ----------
    y_children : list of array-like
        Danh sách mảng nhãn lớp của các node con
    criterion : str
        'gini', 'entropy', hoặc 'error'
        
    Returns
    -------
    float
        Giá trị impurity có trọng số
    """
    criterion_fn = {
        'gini': gini_index,
        'entropy': entropy,
        'error': classification_error
    }
    
    fn = criterion_fn.get(criterion, gini_index)
    n = sum(len(child) for child in y_children)
    
    if n == 0:
        return 0.0
    
    weighted = 0.0
    for child in y_children:
        if len(child) > 0:
            weighted += (len(child) / n) * fn(child)
    
    return weighted
