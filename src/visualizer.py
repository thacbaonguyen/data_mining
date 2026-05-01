"""
visualizer.py — Trực quan hóa Decision Tree và kết quả đánh giá

Includes:
    - plot_tree: Vẽ cây quyết định bằng matplotlib
    - plot_confusion_matrix: Vẽ confusion matrix heatmap
    - plot_feature_importance: Vẽ bar chart feature importance
    - plot_accuracy_vs_depth: Vẽ accuracy theo max_depth (overfitting analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .metrics import confusion_matrix as calc_cm


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix',
                          cmap='Blues', figsize=(6, 5), ax=None):
    """
    Vẽ Confusion Matrix dạng heatmap.
    
    Parameters
    ----------
    y_true, y_pred : array-like
    labels : list, optional
    title : str
    cmap : str
    figsize : tuple
    ax : matplotlib.axes.Axes, optional
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    cm, lbl = calc_cm(y_true, y_pred, labels)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=lbl, yticklabels=lbl,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')
    
    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances, top_n=15,
                            title='Feature Importance', figsize=(10, 6), ax=None):
    """
    Vẽ bar chart Feature Importance.
    
    Parameters
    ----------
    feature_names : list of str
    importances : array-like
    top_n : int
        Số features hiển thị
    title : str
    figsize : tuple
    ax : matplotlib.axes.Axes, optional
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    indices = np.argsort(importances)[::-1][:top_n]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    names = [feature_names[i] for i in indices]
    values = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    fig.tight_layout()
    return fig


def plot_accuracy_vs_depth(depths, train_accs, test_accs,
                           title='Accuracy vs Max Depth', figsize=(8, 5)):
    """
    Vẽ đồ thị Accuracy vs Max Depth (phân tích overfitting).
    
    Parameters
    ----------
    depths : list of int
    train_accs : list of float
    test_accs : list of float
    title : str
    figsize : tuple
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(depths, train_accs, 'o-', color='#2196F3', label='Train', linewidth=2, markersize=8)
    ax.plot(depths, test_accs, 's-', color='#F44336', label='Test', linewidth=2, markersize=8)
    
    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)
    
    fig.tight_layout()
    return fig


def plot_comparison_bar(metrics_dict, title='From Scratch vs Sklearn',
                        figsize=(10, 5)):
    """
    Vẽ grouped bar chart so sánh metrics.
    
    Parameters
    ----------
    metrics_dict : dict
        {'Accuracy': {'From Scratch': 0.91, 'Sklearn': 0.91}, ...}
    title : str
    figsize : tuple
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics_dict.keys())
    models = list(list(metrics_dict.values())[0].keys())
    
    x = np.arange(len(metric_names))
    width = 0.35
    colors = ['#2196F3', '#FF9800']
    
    for i, model in enumerate(models):
        values = [metrics_dict[m][model] for m in metric_names]
        bars = ax.bar(x + i * width - width / 2, values, width,
                      label=model, color=colors[i], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')
    
    fig.tight_layout()
    return fig


def print_tree_text(node, indent="", is_last=True):
    """
    In cây quyết định dạng text (đẹp hơn).
    
    Parameters
    ----------
    node : Node
    indent : str
    is_last : bool
    """
    connector = "└── " if is_last else "├── "
    
    if node.is_leaf:
        dist_str = ", ".join(f"{k}:{v}" for k, v in node.class_distribution.items())
        print(f"{indent}{connector}[{node.prediction}] (n={node.num_samples}, {dist_str})")
        return
    
    if node.is_categorical:
        condition = f"{node.feature_name} ∈ {node.categories_left}"
    else:
        condition = f"{node.feature_name} ≤ {node.threshold:.2f}"
    
    print(f"{indent}{connector}{condition} (n={node.num_samples}, gain={node.gain:.4f})")
    
    new_indent = indent + ("    " if is_last else "│   ")
    
    if node.left:
        print_tree_text(node.left, new_indent, is_last=False)
    if node.right:
        print_tree_text(node.right, new_indent, is_last=True)
