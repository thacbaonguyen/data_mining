# Pipeline Documentation — Data-Centric KDD

## Overview

Du an nay ap dung quy trinh KDD (Knowledge Discovery in Databases) de khai pha du lieu
Bank Marketing — du doan khach hang co dang ky tien gui ky han hay khong.

Thuat toan chinh: **Decision Tree** (from scratch + sklearn).

## KDD Pipeline

```
Selection + Preprocessing (NB01)
         |
    Transformation (NB02)
    |-- Split train/test (stratified, truoc moi transformation)
    |-- One-Hot Encoding (fit on train only)
    |-- Scenario A: co duration (benchmark)
    |-- Scenario B: khong duration (realistic)
    |-- Ablation Study
         |
    Data Mining (NB03 + NB04)
    |-- NB03: Decision Tree from scratch (Hunt's Algorithm)
    |-- NB04: Sklearn DecisionTreeClassifier
    |-- Cung criterion=gini, max_depth=5, random_state=42
         |
    Evaluation (NB05)
    |-- Holdout: Accuracy, Precision, Recall, F1, Confusion Matrix
    |-- 5-fold Stratified CV (encoder fit per fold)
    |-- Scratch vs Sklearn comparison
    |-- Majority baseline comparison
         |
    Knowledge Discovery (NB06)
    |-- Decision Rules
    |-- Feature Importance (A vs B)
    |-- Customer Segmentation (support >= 1%)
    |-- Duration Impact Analysis
    |-- Business Insights + Limitations
```

## Dataset

- **Source**: UCI Bank Marketing (Moro et al., 2014)
- **File**: `data/bank-additional/bank-additional-full.csv`
- **Raw**: 41,188 rows x 21 columns
- **Cleaned** (after dedup): ~41,176 rows
- **Target**: `y` (yes/no), imbalance ~89%/11%

## Notebooks

| # | File | KDD Step | Noi dung |
|---|---|---|---|
| 01 | `01_eda.ipynb` | Selection + Preprocessing | Muc tieu, chon data, EDA, clean |
| 02 | `02_transformation.ipynb` | Transformation | Split, encode, scenario A/B, ablation |
| 03 | `03_from_scratch.ipynb` | Data Mining | DT from scratch, 2 scenarios |
| 04 | `04_sklearn.ipynb` | Data Mining | Sklearn DT, cung dieu kien |
| 05 | `05_evaluation.ipynb` | Evaluation | CV, holdout, comparison |
| 06 | `06_knowledge.ipynb` | Knowledge | Rules, segments, insights |

## Data Flow

```
bank-additional-full.csv
    |
    NB01 (EDA only, no output files)
    |
    NB02 --> data/processed_data.pkl  (scenario_A, scenario_B, encoder)
         --> data/raw_split.pkl       (raw X_train/X_test for CV)
    |
    NB03 --> data/results_scratch.pkl (y_pred, metrics, feature_importance)
    |
    NB04 --> data/results_sklearn.pkl (y_pred, metrics, feature_importance)
    |
    NB05 (reads all pkl, no new output)
    |
    NB06 (reads all pkl, no new output)
```

## Source Code (src/)

| File | Muc dich |
|---|---|
| `tree.py` | Decision Tree Classifier (Hunt's Algorithm) |
| `criteria.py` | Gini Index, Entropy, Information Gain, Gain Ratio |
| `splitter.py` | Best split search |
| `node.py` | Node class |
| `metrics.py` | Accuracy, Precision, Recall, F1, Confusion Matrix |
| `visualizer.py` | Tree visualization, plots |

## Key Design Decisions

1. **Duration**: Hai scenario (A = benchmark, B = realistic). Duration la prediction-time
   unavailable, khong phai train/test leakage.

2. **Imbalance**: Khong dung SMOTE (vo nghia tren one-hot). Dung F1/Recall cho evaluation.
   Undersampling chi la thi nghiem bo sung trong ablation.

3. **CV**: Encoder fit per fold de tranh leakage nhe trong cross-validation.

4. **Scaling**: Khong can (Decision Tree chi dung threshold splits).

5. **Unknown**: Giu nguyen nhu category rieng (mang thong tin).
