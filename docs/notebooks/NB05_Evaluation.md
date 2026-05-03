# NB05: Evaluation + Comparison

> **Bước KDD**: Evaluation
> **File**: `notebooks/05_evaluation.ipynb` — 33 cells
> **Mục đích**: Đánh giá toàn diện mô hình, so sánh Scratch vs Sklearn, so sánh đa thuật toán

---

## Cell 0 [Markdown] — Tiêu đề notebook

## Cell 1 [Code] — Setup môi trường

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix as sk_cm)
from src.tree import DecisionTreeClassifier as ScratchDT
from src.metrics import f1_score as scratch_f1
```

**Giải thích:**
- Import cả metric từ sklearn VÀ metric tự code (để dùng trong CV Boxplot)
- `confusion_matrix as sk_cm`: Đặt alias để tránh xung đột tên

## Cell 2 [Markdown] — Tiêu đề Load Data

## Cell 3 [Code] — Load tất cả dữ liệu và kết quả

```python
with open("data/processed_data.pkl", "rb") as f:  data = pickle.load(f)
with open("data/results_scratch.pkl", "rb") as f: res_scratch = pickle.load(f)
with open("data/results_sklearn.pkl", "rb") as f:  res_sklearn = pickle.load(f)
with open("data/raw_split.pkl", "rb") as f:        raw = pickle.load(f)
```

**Giải thích:**
- Load 4 file: dữ liệu gốc, kết quả scratch, kết quả sklearn (bao gồm NB+RF), dữ liệu raw cho CV

## Cell 4 [Markdown] — Tiêu đề Majority Baseline

## Cell 5 [Code] — Majority Baseline

```python
y_majority = np.array(["no"] * len(y_test))
```

**Giải thích:**
- Tạo "mô hình ngu nhất": Dự đoán tất cả là "no"
- Accuracy = 88.7% (vì 88.7% data thực sự là "no")
- Nhưng F1 cho class "yes" = 0 (không tìm được bất kỳ khách hàng tiềm năng nào)
- **Mục đích**: Đặt baseline để chứng minh mô hình DT có giá trị hơn đoán bừa

## Cell 6 [Markdown] — Tiêu đề Holdout Comparison

## Cell 7 [Code] — Bảng so sánh Holdout (4 metrics × 2 scenarios × 2 models)

```python
def format_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="yes"),
        "Recall": recall_score(y_true, y_pred, pos_label="yes"),
        "F1": f1_score(y_true, y_pred, pos_label="yes"),
    }
```

**Giải thích:**
- Tạo bảng DataFrame so sánh Scratch vs Sklearn trên cả Scenario A và B
- `pos_label="yes"`: Chỉ định class quan tâm là "yes" (class thiểu số)
- **Kết quả**: Scratch ≈ Sklearn (sai khác < 0.001)

## Cell 8 [Markdown] — Tiêu đề Confusion Matrices

## Cell 9 [Code] — Vẽ 4 Confusion Matrix

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
cases = [
    ("Scenario A — Scratch", ...),
    ("Scenario A — Sklearn", ...),
    ("Scenario B — Scratch", ...),
    ("Scenario B — Sklearn", ...),
]
```

**Giải thích:**
- 4 heatmap: 2 scenarios × 2 models
- Mỗi ô: Hàng = Actual (thực tế), Cột = Predicted (dự đoán)
  - Góc trên-trái = True Negative (đoán "no" đúng)
  - Góc dưới-phải = True Positive (đoán "yes" đúng)
  - Góc trên-phải = False Positive (đoán "yes" sai)
  - Góc dưới-trái = False Negative (đoán "no" sai — bỏ sót khách tiềm năng)

**Output**: `report/05_confusion_matrices.png`

## Cell 10 [Markdown] — Tiêu đề Cross-Validation

## Cell 11 [Code] — Setup CV variables

```python
X_train_raw = raw["X_train_raw"]
y_train_raw = raw["y_train"]
cat_cols = raw["cat_cols"]
```

**Giải thích:**
- Load dữ liệu RAW (chưa encode) từ `raw_split.pkl`
- Dùng dữ liệu raw vì mỗi fold phải fit encoder riêng

## Cell 12 [Code] — 5-Fold CV Scenario B (Encoder fit per fold)

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw_B, y_train_raw)):
    # 1. Chia fold
    # 2. Fit encoder trên fold-train
    # 3. Transform fold-val
    # 4. Train Scratch DT + Sklearn DT
    # 5. Tính F1, Accuracy cho mỗi model
```

**Giải thích chi tiết:**
- `StratifiedKFold(n_splits=5)`: Chia 5 phần, mỗi phần giữ tỷ lệ yes/no
- **Trong mỗi vòng lặp (fold):**
  1. Tách dữ liệu RAW thành fold-train và fold-val
  2. Tạo encoder MỚI, fit CHỈ trên fold-train
  3. Transform cả fold-train và fold-val
  4. Train cả Scratch và Sklearn trên fold-train đã encode
  5. Predict trên fold-val, tính metrics
- **Tại sao encoder fit per fold?** Chống data leakage ở mức CV

## Cell 13 [Code] — 5-Fold CV Scenario A

Tương tự Cell 12 nhưng giữ nguyên cột duration.

## Cell 14 [Markdown] — Kết luận Evaluation sơ bộ

## Cell 15 [Markdown] — Tiêu đề Hyperparameter Analysis

## Cell 16 [Code] — Phân tích max_depth

```python
depths = [3, 5, 7, 10, None]
for d in depths:
    sk = DecisionTreeClassifier(max_depth=d, ...)
    sk.fit(...)
```

**Giải thích:**
- Thử 5 giá trị max_depth trên Scenario B
- Ghi lại: Accuracy, F1, Recall, Precision, số lá (n_leaves)
- **Kết quả:**
  - depth=3: F1 thấp (cây quá đơn giản)
  - depth=5: F1 cân bằng, 32 lá (dễ diễn giải)
  - depth=None: 4973 lá (overfit, quá phức tạp)

## Cell 17 [Code] — Biểu đồ Depth Analysis

- Trái: 4 metrics vs max_depth
- Phải: Số lá vs max_depth (complexity)

**Output**: `report/05_depth_analysis.png`

## Cell 18 [Markdown] — Tiêu đề Precision/Recall Trade-off

## Cell 19 [Code] — Undersampling Trade-off

```python
from imblearn.under_sampling import RandomUnderSampler
```

**Giải thích:**
- Train DT trên dữ liệu đã undersample
- So sánh với DT trên dữ liệu gốc (imbalanced)
- **Kết quả**: Undersampling tăng Recall (+0.33) nhưng giảm Precision (-0.20)
- **Ý nghĩa**: Tùy mục tiêu kinh doanh mà chọn model phù hợp

## Cell 20 [Markdown] — Tiêu đề Precision-Recall Curve

## Cell 21 [Code] — Vẽ PR Curve (Scenario A và B)

```python
y_proba = sk.predict_proba(d["X_test"])[:, list(sk.classes_).index("yes")]
prec_vals, rec_vals, thresholds = precision_recall_curve(y_test_bin, y_proba)
ap = average_precision_score(y_test_bin, y_proba)
```

**Giải thích:**
- `predict_proba()`: Trả về xác suất (ví dụ: 70% yes, 30% no) thay vì nhãn cứng
- Thay đổi ngưỡng (threshold) từ 0 đến 1 → Mỗi ngưỡng cho 1 cặp (Precision, Recall) khác nhau
- AP (Average Precision) = Diện tích dưới đường cong PR
- **Kết quả**: AP Scenario A > AP Scenario B (nhờ duration)

**Output**: `report/05_pr_curve.png`

## Cell 22 [Markdown] — Tiêu đề CV Boxplot

## Cell 23 [Code] — Cross-Validation Boxplot (Scratch vs Sklearn)

**Giải thích:**
- Chạy 5-fold CV với encoder fit per fold cho cả 4 trường hợp: Scratch A, Sklearn A, Scratch B, Sklearn B
- Vẽ boxplot phân phối F1 Score qua 5 folds
- **Mục đích**: Chứng minh (1) Scratch ≈ Sklearn, (2) Mô hình ổn định (std thấp)

**Output**: `report/05_cv_boxplot.png`

## Cell 24 [Markdown] — Tiêu đề Learning Curve

## Cell 25 [Code] — Learning Curve (Dữ liệu đã đủ chưa?)

```python
train_sizes, train_scores, val_scores = learning_curve(
    sk, d["X_train"], d["y_train"],
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring=make_scorer(f1_score, pos_label="yes"),
    ...
)
```

**Giải thích:**
- Tăng dần lượng data training (10% → 100%) và ghi lại F1 Score
- Nếu 2 đường (Train, Validation) hội tụ → Dữ liệu đã đủ
- Nếu vẫn cách xa → Cần thêm data hoặc model đang overfit
- `pos_label="yes"`: Chỉ định class "yes" làm positive label (vì data dùng string, không phải số 0/1)

**Output**: `report/05_learning_curve.png`

## Cell 26 [Markdown] — Tiêu đề So sánh đa thuật toán

## Cell 27 [Code] — Bảng so sánh 4 thuật toán × 2 Scenarios

```python
models = {
    "DT Scratch":   res_scratch["scenario_B"]["y_pred"],
    "DT Sklearn":   res_sklearn["scenario_B"]["y_pred"],
    "Naive Bayes":  nb_res["scenario_B"]["y_pred"],
    "Random Forest": rf_res["scenario_B"]["y_pred"],
}
```

**Giải thích:**
- Tạo bảng DataFrame chứa Accuracy, Precision, Recall, F1 của 4 thuật toán
- In riêng cho Scenario A và Scenario B

## Cell 28 [Markdown] — Tiêu đề Biểu đồ F1

## Cell 29 [Code] — Biểu đồ bar F1 Score (4 thuật toán × 2 Scenarios)

- Grouped bar chart: Xanh = Scenario A, Cam = Scenario B
- Mỗi nhóm = 1 thuật toán, ghi giá trị F1 trên đầu mỗi bar
- **Nhận xét**: RF thường ổn định hơn DT đơn lẻ, NB nhanh nhưng có hạn chế

**Output**: `report/05_multi_model_f1.png`

## Cell 30 [Markdown] — Tiêu đề Confusion Matrix NB+RF

## Cell 31 [Code] — CM cho Naive Bayes và Random Forest (Scenario B)

- 2 heatmap cạnh nhau: NB (trái) và RF (phải)
- So sánh phân bố FP/FN với DT ở Cell 9

**Output**: `report/05_cm_extra_models.png`

## Cell 32 [Code] — Kết luận Evaluation tổng hợp

- In bảng tóm tắt cuối cùng
- So sánh Scratch vs Sklearn: sai khác < 0.001
- Ghi nhận kết quả của Naive Bayes và Random Forest
- Nhận xét về giới hạn và hướng cải tiến
