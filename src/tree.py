"""
tree.py — Decision Tree Classifier (From Scratch)

Implements Hunt's Algorithm with:
    - GINI Index hoặc Entropy (criterion)
    - Pre-pruning (max_depth, min_samples_split, min_samples_leaf)
    - Post-pruning (Reduced Error Pruning)
    - Feature importance

Reference: docs/main/data_mining.md Part 3
"""

import numpy as np
from collections import Counter

from .node import Node
from .splitter import find_best_split
from .criteria import gini_index, entropy


class DecisionTreeClassifier:
    """
    Decision Tree Classifier — From Scratch.
    
    Thuật toán: Hunt's Algorithm (đệ quy, top-down, greedy)
    
    Parameters
    ----------
    criterion : str, default='gini'
        Tiêu chí split: 'gini' hoặc 'entropy'
    max_depth : int or None, default=None
        Độ sâu tối đa. None = không giới hạn.
    min_samples_split : int, default=2
        Số mẫu tối thiểu để tiếp tục split.
    min_samples_leaf : int, default=1
        Số mẫu tối thiểu tại mỗi leaf node.
    min_impurity_decrease : float, default=0.0
        Ngưỡng gain tối thiểu để chấp nhận split.
    
    Ref: data_mining.md L1270-1300 (Hunt's Algorithm)
    Ref: data_mining.md L1533-1610 (Pruning)
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        
        self.root = None
        self.feature_names = None
        self.categorical_features = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self._n_nodes = 0
        self._n_leaves = 0
        self._max_depth_reached = 0
    
    def fit(self, X, y, feature_names=None, categorical_features=None):
        """
        Huấn luyện mô hình Decision Tree.
        
        Sử dụng Hunt's Algorithm:
        1. Nếu tất cả cùng lớp → tạo leaf
        2. Nếu không thể split → tạo leaf (majority class)
        3. Tìm best split → tạo 2 children → đệ quy
        
        Parameters
        ----------
        X : array-like (n_samples × n_features)
        y : array-like (n_samples,)
        feature_names : list of str, optional
        categorical_features : list of int, optional
        
        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = sorted(set(y))
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features_)]
        self.categorical_features = categorical_features or []
        
        # Initialize feature importances
        self._feature_importances = np.zeros(self.n_features_)
        self._n_nodes = 0
        self._n_leaves = 0
        self._max_depth_reached = 0
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total = np.sum(self._feature_importances)
        if total > 0:
            self.feature_importances_ = self._feature_importances / total
        else:
            self.feature_importances_ = self._feature_importances
        
        return self
    
    def _build_tree(self, X, y, depth):
        """
        Xây dựng cây đệ quy (Hunt's Algorithm).
        
        Điều kiện dừng (Ref: data_mining.md L1420-1430):
        1. Tất cả mẫu cùng lớp
        2. Không đủ mẫu để split
        3. Đạt max_depth
        4. Không tìm được split tốt
        """
        node = Node()
        node.num_samples = len(y)
        node.depth = depth
        node.class_distribution = dict(Counter(y))
        
        # Majority class
        majority_class = Counter(y).most_common(1)[0][0]
        
        # Tính impurity
        if self.criterion == 'gini':
            node.impurity = gini_index(y)
        else:
            node.impurity = entropy(y)
        
        # Update max depth
        self._max_depth_reached = max(self._max_depth_reached, depth)
        self._n_nodes += 1
        
        # === ĐIỀU KIỆN DỪNG ===
        
        # 1. Tất cả cùng lớp
        if len(set(y)) == 1:
            node.is_leaf = True
            node.prediction = majority_class
            self._n_leaves += 1
            return node
        
        # 2. Đạt max_depth
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = majority_class
            self._n_leaves += 1
            return node
        
        # 3. Không đủ mẫu để split
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = majority_class
            self._n_leaves += 1
            return node
        
        # === TÌM BEST SPLIT ===
        best_split = find_best_split(
            X, y,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            criterion=self.criterion
        )
        
        # 4. Không tìm được split hợp lệ
        if best_split is None or best_split['gain'] < self.min_impurity_decrease:
            node.is_leaf = True
            node.prediction = majority_class
            self._n_leaves += 1
            return node
        
        # === THỰC HIỆN SPLIT ===
        node.feature = best_split['feature']
        node.feature_name = best_split['feature_name']
        node.is_categorical = best_split['is_categorical']
        node.gain = best_split['gain']
        
        if best_split['is_categorical']:
            node.categories_left = best_split['categories_left']
            mask_left = np.isin(X[:, node.feature], list(node.categories_left))
        else:
            node.threshold = best_split['threshold']
            mask_left = X[:, node.feature].astype(float) <= node.threshold
        
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[~mask_left], y[~mask_left]
        
        # Kiểm tra min_samples_leaf
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = majority_class
            self._n_leaves += 1
            return node
        
        # Update feature importance
        # Importance = n_samples × gain
        self._feature_importances[node.feature] += len(y) * node.gain
        
        # Đệ quy xây children
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        
        Duyệt cây từ root → leaf cho mỗi sample.
        
        Parameters
        ----------
        X : array-like (n_samples × n_features)
            
        Returns
        -------
        numpy.ndarray
            Mảng nhãn dự đoán
        """
        X = np.array(X)
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def _predict_one(self, x, node):
        """Dự đoán 1 sample bằng cách duyệt cây."""
        if node.is_leaf:
            return node.prediction
        
        if node.is_categorical:
            if x[node.feature] in node.categories_left:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
        else:
            if float(x[node.feature]) <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
    
    def prune(self, X_val, y_val):
        """
        Post-pruning: Reduced Error Pruning.
        
        Thuật toán:
        1. Với mỗi internal node (bottom-up):
           - Tạm thời chuyển thành leaf (majority class)
           - Nếu accuracy trên validation set KHÔNG giảm → giữ pruning
           - Nếu accuracy giảm → hoàn tác
        
        Ref: data_mining.md L1586-1610
        
        Parameters
        ----------
        X_val : array-like
            Dữ liệu validation
        y_val : array-like
            Nhãn validation
        """
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        self._prune_node(self.root, X_val, y_val)
        
        # Recalculate stats
        self._n_nodes = 0
        self._n_leaves = 0
        self._count_nodes(self.root)
    
    def _prune_node(self, node, X_val, y_val):
        """Đệ quy prune từ dưới lên."""
        if node.is_leaf:
            return
        
        # Đệ quy prune children trước (bottom-up)
        if node.left and not node.left.is_leaf:
            self._prune_node(node.left, X_val, y_val)
        if node.right and not node.right.is_leaf:
            self._prune_node(node.right, X_val, y_val)
        
        # Thử prune node này
        # Accuracy trước khi prune
        acc_before = np.sum(self.predict(X_val) == y_val) / len(y_val)
        
        # Lưu lại children
        left_backup = node.left
        right_backup = node.right
        is_leaf_backup = node.is_leaf
        prediction_backup = node.prediction
        
        # Tạm chuyển thành leaf
        node.is_leaf = True
        majority = Counter(y_val).most_common(1)[0][0] if len(y_val) > 0 else node.prediction
        # Dùng majority class từ training data (class_distribution)
        node.prediction = max(node.class_distribution, key=node.class_distribution.get)
        
        # Accuracy sau khi prune
        acc_after = np.sum(self.predict(X_val) == y_val) / len(y_val)
        
        # Nếu accuracy không giảm → giữ pruning
        if acc_after >= acc_before:
            pass  # Giữ leaf
        else:
            # Hoàn tác
            node.left = left_backup
            node.right = right_backup
            node.is_leaf = is_leaf_backup
            node.prediction = prediction_backup
    
    def _count_nodes(self, node):
        """Đếm số nodes và leaves."""
        if node is None:
            return
        self._n_nodes += 1
        if node.is_leaf:
            self._n_leaves += 1
        else:
            self._count_nodes(node.left)
            self._count_nodes(node.right)
    
    def get_depth(self):
        """Trả về độ sâu cây."""
        return self._get_depth(self.root)
    
    def _get_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
    
    def get_n_leaves(self):
        """Trả về số leaf nodes."""
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def print_tree(self, node=None, indent=""):
        """In cây dạng text."""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            print(f"{indent}→ [{node.prediction}] (samples={node.num_samples}, "
                  f"dist={node.class_distribution})")
            return
        
        if node.is_categorical:
            condition = f"{node.feature_name} in {node.categories_left}"
        else:
            condition = f"{node.feature_name} <= {node.threshold:.4f}"
        
        print(f"{indent}{condition} (samples={node.num_samples}, "
              f"gain={node.gain:.4f})")
        
        print(f"{indent}├── True:")
        self.print_tree(node.left, indent + "│   ")
        
        print(f"{indent}└── False:")
        self.print_tree(node.right, indent + "    ")
    
    def __repr__(self):
        depth = self.get_depth() if self.root else 0
        n_leaves = self.get_n_leaves() if self.root else 0
        return (f"DecisionTreeClassifier(criterion='{self.criterion}', "
                f"max_depth={self.max_depth}, depth={depth}, "
                f"n_leaves={n_leaves})")
