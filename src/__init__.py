"""
Decision Tree From Scratch
==========================

Package chứa implementation Decision Tree Classifier from scratch.

Modules:
    - criteria: GINI, Entropy, Information Gain, Gain Ratio
    - splitter: Tìm best split (continuous + categorical)
    - node: Cấu trúc Node
    - tree: DecisionTreeClassifier
    - metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

from .tree import DecisionTreeClassifier
from .criteria import gini_index, entropy, information_gain, gain_ratio
from .metrics import (
    confusion_matrix, accuracy, precision, recall, f1_score,
    classification_report
)
