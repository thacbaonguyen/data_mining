"""
node.py — Cấu trúc Node cho Decision Tree

Mỗi node lưu trữ:
    - Thông tin phân nhánh (feature, threshold/categories)
    - Node con (children)
    - Nhãn dự đoán (nếu là leaf)
    - Metadata (impurity, num_samples, class_distribution)
"""


class Node:
    """
    Node trong cây quyết định.
    
    Có 2 loại:
    - Internal node: có feature, threshold, children
    - Leaf node: có prediction (is_leaf = True)
    """
    
    def __init__(self):
        # Thông tin split
        self.feature = None           # Index hoặc tên thuộc tính dùng để split
        self.feature_name = None      # Tên thuộc tính (để hiển thị)
        self.threshold = None         # Ngưỡng split (cho continuous)
        self.categories_left = None   # Categories thuộc nhánh trái (cho categorical)
        self.is_categorical = False   # Thuộc tính có phải categorical không?
        
        # Children
        self.left = None              # Node con trái (≤ threshold hoặc in categories_left)
        self.right = None             # Node con phải (> threshold hoặc not in categories_left)
        
        # Leaf info
        self.is_leaf = False          # Có phải nút lá không?
        self.prediction = None        # Nhãn dự đoán (majority class)
        
        # Metadata
        self.impurity = None          # GINI hoặc Entropy tại node
        self.num_samples = 0          # Số mẫu tại node
        self.class_distribution = {}  # {class: count}
        self.depth = 0                # Depth trong cây
        self.gain = None              # Information Gain hoặc GINI Gain
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction}, samples={self.num_samples})"
        else:
            if self.is_categorical:
                return (f"Node(feature={self.feature_name}, "
                        f"categories_left={self.categories_left}, "
                        f"samples={self.num_samples})")
            else:
                return (f"Node(feature={self.feature_name}, "
                        f"threshold={self.threshold:.4f}, "
                        f"samples={self.num_samples})")
