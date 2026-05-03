# TÀI LIỆU HỌC TẬP — ĐỒ ÁN KHAI PHÁ DỮ LIỆU

## 1. Tổng quan dự án

### 1.1 Mục tiêu
- **Bài toán**: Dự đoán khách hàng có đăng ký tiền gửi kỳ hạn (term deposit) hay không
- **Target**: Biến `y` (yes/no) — Binary Classification với Class Imbalance (~89% no, ~11% yes)
- **Dataset**: UCI Bank Marketing (Moro et al., 2014) — 41,188 dòng × 21 cột

### 1.2 Mục tiêu học thuật
1. Triển khai **Decision Tree from scratch** theo Hunt's Algorithm
2. So sánh kết quả với **sklearn DecisionTreeClassifier** trong cùng điều kiện
3. So sánh mở rộng với **Naive Bayes** và **Random Forest** (sklearn) để đánh giá đa thuật toán
4. Áp dụng quy trình **KDD** (Knowledge Discovery in Databases) — Fayyad et al., 1996

### 1.3 Mục tiêu khai phá dữ liệu
1. Rút ra đặc điểm/nhóm khách hàng có khả năng đăng ký cao
2. Phân tích tác động của chất lượng dữ liệu (duration, imbalance) lên mô hình
3. Đưa ra khuyến nghị dựa trên pattern quan sát (KHÔNG phải nhân quả)

### 1.4 Kết quả đạt được

| Metric | Scenario A (benchmark) | Scenario B (realistic) |
|--------|----------------------|----------------------|
| Accuracy | 0.9144 | 0.9009 |
| Precision(yes) | 0.6534 | 0.6522 |
| Recall(yes) | 0.5119 | 0.2586 |
| F1(yes) | 0.5740 | 0.3704 |
| CV F1 (mean±std) | 0.5747 ± 0.0104 | 0.3630 ± 0.0258 |

- From scratch vs sklearn: sai khác < 0.001 trên cả CV Accuracy và F1
- Top segment: retired + basic.4y có yes-rate 31% (lift 2.75× so với baseline 11.3%)

---

## 2. Luồng dự án & Pipeline

### 2.1 Tổng quan pipeline KDD

```
┌─────────────────────────────────────────────────────────────┐
│  NB01: Selection + Preprocessing                            │
│  Load raw data (41,188) → Check null/duplicate/unknown      │
│  → EDA: target distribution, categorical, numerical,        │
│    correlation, duration analysis                            │
│  Output: Cleaned dataset (~41,176), hiểu rõ dữ liệu        │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NB02: Transformation                                       │
│  Split 80/20 stratified (TRƯỚC mọi encoding)                │
│  → One-Hot Encoding (fit on train, transform test)          │
│  → Scenario A (có duration) / B (không duration)            │
│  → Ablation Study: đo tác động từng bước transformation    │
│  Output: processed_data.pkl, raw_split.pkl                  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NB03: Data Mining — From Scratch                           │
│  Train DT (gini, max_depth=5) trên cả Scenario A & B       │
│  → Classification report, feature importance                │
│  Output: results_scratch.pkl                                │
│                                                             │
│  NB04: Data Mining — Sklearn + Multi-Algorithm              │
│  Decision Tree: cùng criterion, max_depth, random_state     │
│  Naive Bayes (GaussianNB): baseline xác suất                │
│  Random Forest (100 cây, depth=5): ensemble nâng cao        │
│  → So sánh đa thuật toán trên cùng data                    │
│  Output: results_sklearn.pkl (DT + NB + RF)                 │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NB05: Evaluation                                           │
│  Majority baseline → Holdout comparison (4 metrics)         │
│  → 5-fold Stratified CV (encoder fit per fold)              │
│  → Hyperparameter analysis (max_depth 3/5/7/10/None)        │
│  → Precision/Recall trade-off (undersampling)               │
│  → PR Curve, CV Boxplot, Learning Curve                     │
│  → So sánh đa thuật toán (DT/NB/RF) × 2 Scenarios          │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NB06: Knowledge Discovery                                  │
│  Decision Rules (strong/exploratory) → Customer Segments    │
│  → Feature Importance A vs B → Duration Impact              │
│  → Kết luận + Khuyến nghị + Giới hạn                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Lý do thiết kế từng bước

**Tại sao split TRƯỚC encoding?**
- Nếu fit encoder trên toàn bộ data → encoder "nhìn thấy" category của test → information leakage
- Nguyên tắc: mọi transformation chỉ fit trên train

**Tại sao 2 scenario (A/B)?**
- Duration chỉ biết SAU cuộc gọi → không dùng được để dự đoán TRƯỚC chiến dịch
- Scenario A: benchmark thuật toán (nhiều paper dùng duration)
- Scenario B: mô hình thực tế (không có duration)
- Đây là **prediction-time unavailability**, KHÔNG phải train/test leakage

**Tại sao giữ "unknown" thay vì drop/fill?**
- Fill bằng mode → thiên vị class đa số (dataset imbalanced)
- Drop rows → mất ~25% dữ liệu
- "unknown" mang thông tin: ngân hàng KHÔNG CÓ data về khách → bản thân sự thiếu thông tin có thể là signal

**Tại sao không dùng SMOTE?**
- SMOTE tạo điểm trung bình giữa 2 mẫu → vô nghĩa trên one-hot encoded features (0/1)
- Ví dụ: trung bình giữa "married" và "single" = 0.5 → không tồn tại trong thực tế

**Imbalance Strategy của project là gì?**
- Dataset mất cân bằng mạnh (~89% no / ~11% yes), nên accuracy đơn thuần dễ gây hiểu nhầm
- Dùng stratified split và 5-fold Stratified CV để giữ tỷ lệ class ổn định
- Đánh giá bằng Precision/Recall/F1 cho class `yes`, không chỉ nhìn accuracy
- Undersampling chỉ dùng như ablation để phân tích trade-off, không phải scenario chính

**Tại sao không scaling?**
- Decision Tree chỉ dùng threshold splits (x <= t), không tính khoảng cách
- Scaling không ảnh hưởng kết quả → bỏ qua để đơn giản

**Tại sao CV encoder fit per fold?**
- Nếu dùng encoder global → validation fold đã "nhìn thấy" categories từ training fold
- Chuẩn nhất: trong mỗi fold, fit encoder riêng trên fold-train, transform fold-val

### 2.3 Data flow & artifacts cần nhớ

| Artifact | Tạo từ notebook | Nội dung | Dùng ở đâu |
|----------|----------------|----------|------------|
| `bank-additional-full.csv` | Raw data | 41,188 dòng × 21 cột | NB01, NB02 |
| `raw_split.pkl` | NB02 | Train/test raw sau split, chưa encode | NB05 CV per fold |
| `processed_data.pkl` | NB02 | Scenario A/B đã one-hot, sẵn sàng train | NB03, NB04, NB05 |
| `results_scratch.pkl` | NB03 | Prediction, report, importance của DT scratch | NB05, NB06 |
| `results_sklearn.pkl` | NB04 | Prediction, report, importance của DT + NB + RF | NB05 |
| `report/*.png` | NB01-NB06 | 23 hình: EDA, ablation, trees, multi-model, insight | Demo/báo cáo |

**Cách nhớ nhanh**:
Raw CSV → split/encode thành `processed_data.pkl` → train ra `results_*.pkl` → evaluation/knowledge tạo insight và hình trong `report/`.

### 2.4 Feature dictionary tối thiểu phải thuộc

| Nhóm | Feature quan trọng | Ý nghĩa cần nhớ |
|------|-------------------|-----------------|
| Client info | `age`, `job`, `marital`, `education` | Thông tin nhân khẩu học khách hàng |
| Campaign info | `contact`, `month`, `day_of_week`, `campaign` | Cách và thời điểm liên hệ trong chiến dịch hiện tại |
| Risk feature | `duration` | Thời lượng cuộc gọi; chỉ biết sau cuộc gọi, không dùng cho dự đoán trước chiến dịch |
| Previous campaign | `pdays`, `previous`, `poutcome` | Lịch sử liên hệ trước đó; `pdays=999` thường nghĩa là chưa từng được liên hệ |
| Economic indicators | `emp.var.rate`, `euribor3m`, `nr.employed`, `cons.*` | Bối cảnh kinh tế vĩ mô tại thời điểm chiến dịch |
| Target | `y` | Khách có đăng ký term deposit hay không |

### 2.5 Tiêu chí tri thức theo KDD

Một insight trong NB06 chỉ nên được xem là "tri thức" nếu đạt 4 tiêu chí:

| Tiêu chí | Câu tự kiểm tra |
|----------|-----------------|
| Valid | Có số liệu/metric/support chứng minh không? |
| Novel | Có thêm hiểu biết mới, không quá hiển nhiên không? |
| Useful | Có giúp ra quyết định marketing không? |
| Understandable | Người không chuyên có hiểu được không? |

Ví dụ: `retired + basic.4y` có yes-rate 31%, lift 2.75 và support 1.4% → có thể là segment đáng chú ý. Nhưng vẫn phải nói đây là **pattern quan sát**, không phải quan hệ nhân quả.

---

## 3. Giải thích code theo luồng

### 3.1 Source code (`src/`)

#### `src/node.py` — Cấu trúc nút cây
- **Class `Node`**: Lưu thông tin mỗi nút trong cây
- Thuộc tính chính: `feature_name`, `threshold`, `left/right`, `prediction`, `is_leaf`, `class_distribution`
- Nút lá: `is_leaf=True`, `prediction` = majority class tại nút đó

#### `src/criteria.py` — Hàm đánh giá split

| Hàm | Input | Output | Công thức |
|-----|-------|--------|-----------|
| `gini_index(y)` | Mảng nhãn | Float [0, 0.5] với binary | GINI(t) = 1 - Σ[p(i|t)]² |
| `entropy(y)` | Mảng nhãn | Float [0, 1] | H(t) = -Σ p(i|t) × log₂(p(i|t)) |
| `information_gain(y_parent, y_children)` | Nhãn cha, danh sách nhãn con | Float | IG = I(parent) - Σ(Nⱼ/N) × I(childⱼ) |
| `gain_ratio(y_parent, y_children)` | Nhãn cha, danh sách nhãn con | Float | GR = IG / SplitInfo |
| `weighted_impurity(y_children, criterion)` | Danh sách nhãn con | Float | Trung bình có trọng số impurity |

**Quyết định**: Dùng Gini vì (1) tính toán nhanh hơn Entropy (không cần log), (2) kết quả gần tương đương.

#### `src/splitter.py` — Tìm split tốt nhất

| Hàm | Mục đích |
|-----|----------|
| `find_best_split_continuous(X_col, y)` | Sort giá trị, thử mọi midpoint, chọn split có impurity thấp nhất |
| `find_best_split_categorical(X_col, y)` | Sort category theo positive-rate, thử n-1 split points |
| `find_best_split(X, y, ...)` | Duyệt tất cả features, gọi hàm phù hợp, trả về best split |

**Lưu ý kỹ thuật**: Với continuous features, sort O(n log n) rồi duyệt O(n) midpoints → tổng O(n log n) per feature.

**Lưu ý khi bị hỏi xoáy**: Categorical split trong code là heuristic phù hợp binary classification, không phải duyệt toàn bộ `2^(k-1)-1` subset. Trong pipeline chính, dữ liệu categorical đã được one-hot để scratch và sklearn so sánh công bằng.

#### `src/tree.py` — Decision Tree Classifier

**Class `DecisionTreeClassifier`**:
- `__init__`: Nhận `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- `fit(X, y)`: Gọi `_build_tree` đệ quy
- `_build_tree(X, y, depth)`: **Hunt's Algorithm**:
  1. Nếu tất cả y cùng lớp → leaf
  2. Nếu depth >= max_depth → leaf (majority class)
  3. Nếu n_samples < min_samples_split → leaf
  4. Tìm best split → tạo 2 children → đệ quy
- `predict(X)`: Duyệt cây cho từng mẫu
- `feature_importances_`: Tính bằng weighted impurity decrease tại mỗi split
- `prune(X_val, y_val)`: Reduced Error Pruning (post-pruning)

#### `src/metrics.py` — Metrics đánh giá

| Hàm | Công thức | Ý nghĩa |
|-----|-----------|---------|
| `accuracy(y_true, y_pred)` | (TP+TN)/(TP+TN+FP+FN) | Tỷ lệ dự đoán đúng tổng thể |
| `precision(y_true, y_pred, pos)` | TP/(TP+FP) | Trong số dự đoán "yes", bao nhiêu đúng |
| `recall(y_true, y_pred, pos)` | TP/(TP+FN) | Trong số thực sự "yes", bao nhiêu được tìm thấy |
| `f1_score(y_true, y_pred, pos)` | 2×P×R/(P+R) | Trung bình điều hòa Precision và Recall |
| `confusion_matrix(y_true, y_pred)` | Ma trận 2×2 | TP, TN, FP, FN |

---

### 3.2 Notebooks

#### NB01: `01_eda.ipynb`
**Mục đích**: Hiểu bài toán, chọn dataset, khám phá dữ liệu, tiền xử lý cơ bản

| Cell | Nội dung | Logic |
|------|----------|-------|
| Setup | Import libs, set PROJECT_ROOT | `os.chdir()` để path nhất quán |
| Load | `pd.read_csv(sep=";")` | Dataset dùng semicolon separator |
| Null check | `df.isnull().sum()` | Kết quả: 0 null |
| Duplicate | `df.drop_duplicates()` | 41,188 → 41,176 (removed 12) |
| Unknown | Count "unknown" per column | Quyết định: giữ nguyên |
| Target plot | Bar + Pie chart | Thấy 88.7% no / 11.3% yes |
| Categorical | `pd.crosstab` normalized | Yes-rate khác nhau theo job, education... |
| Numerical | Histogram overlay by target | Duration phân biệt rõ nhất |
| Correlation | `sns.heatmap` | euribor3m ↔ nr.employed tương quan cao |
| Duration | 3 subplot phân tích | Mean 553s (yes) vs 220s (no) |

#### NB02: `02_transformation.ipynb`
**Mục đích**: Biến đổi dữ liệu, tách scenario, ablation study

| Cell | Nội dung | Quyết định kỹ thuật |
|------|----------|-------------------|
| Split | `train_test_split(stratify=y)` | 80/20, stratify giữ tỷ lệ yes/no |
| Encode | `OneHotEncoder(handle_unknown="ignore")` | `ignore` để test có category mới → vector 0 |
| Scenario A | Giữ duration | Benchmark thuật toán |
| Scenario B | `drop(columns=dur_cols)` | Realistic, loại duration |
| Ablation | Train cùng model trên 3 phiên bản data | Cô lập tác động từng bước |
| Save | `pickle.dump` | 2 file pkl cho downstream notebooks |

**Ablation results:**

| Scenario | Acc | Prec | Rec | F1 |
|----------|-----|------|-----|-----|
| A: có duration | 0.9144 | 0.6534 | 0.5119 | 0.5740 |
| B: không duration | 0.9009 | 0.6522 | 0.2586 | 0.3704 |
| B+: undersample | 0.8719 | 0.4480 | 0.5894 | 0.5091 |

#### NB03: `03_from_scratch.ipynb`
**Mục đích**: Train DT from scratch trên cả 2 scenario

- Giải thích Hunt's Algorithm, Gini Index, Stopping Criteria
- Train riêng Scenario A và B → classification report, feature importance
- Visualize tree structure bằng `print_tree()`
- Feature importance: duration chiếm 50.9% ở Scenario A

#### NB04: `04_sklearn.ipynb`
**Mục đích**: Train sklearn DT + Naive Bayes + Random Forest, so sánh đa thuật toán

- **Decision Tree**: `criterion="gini"`, `max_depth=5`, `random_state=42` — cùng điều kiện với scratch
- **Naive Bayes** (`GaussianNB`): Baseline xác suất, không cần hyperparameter tuning
- **Random Forest** (`n_estimators=100, max_depth=5`): Ensemble 100 cây, majority vote
- Visualize cây DT bằng `plot_tree()`, so sánh Feature Importance DT vs RF
- Lưu kết quả cả 3 thuật toán vào `results_sklearn.pkl`

#### NB05: `05_evaluation.ipynb`
**Mục đích**: Đánh giá toàn diện, so sánh công bằng, đa thuật toán

| Section | Nội dung |
|---------|----------|
| Majority baseline | Predict tất cả "no" → Acc 88.7% nhưng F1=0 |
| Holdout | 4 metrics × 2 scenario × 2 model = bảng so sánh đầy đủ |
| Confusion matrices | 4 heatmap (A/B × scratch/sklearn) |
| CV | 5-fold, encoder fit per fold, cùng folds cho cả 2 model |
| Depth analysis | max_depth 3/5/7/10/None — depth=5 cân bằng tốt |
| Trade-off | Undersampling tăng Recall +0.33 nhưng giảm Precision -0.20 |
| PR Curve | Precision-Recall curve cho Scenario A và B |
| CV Boxplot | Phân phối F1 qua 5 folds — Scratch vs Sklearn |
| Learning Curve | Train vs Validation F1 theo lượng data |
| So sánh đa thuật toán | Bảng + biểu đồ F1: DT Scratch / DT Sklearn / NB / RF |
| CM NB+RF | Confusion matrix cho Naive Bayes và Random Forest |

#### NB06: `06_knowledge.ipynb`
**Mục đích**: Rút tri thức, KHÔNG chỉ báo metric

| Section | Nội dung |
|---------|----------|
| Feature importance A vs B | Cho thấy duration "che lấp" features thực tế |
| Decision rules | Lọc support ≥ 50, ghi rõ exploratory rules |
| Segments | job×education, month×contact, minimum support ≥ 1% |
| WHY explanation | Tháng 3/9/10 gọi ít hơn → chọn lọc kỹ → yes-rate cao |
| Imbalance Strategy | Tổng kết cách xử lý mất cân bằng lớp và trade-off Precision/Recall |
| Kết luận | Trung thực, có điều kiện, ghi rõ giới hạn |

**Công thức NB06 cần nhớ**:
- `support` = số mẫu thuộc rule/segment
- `confidence` hoặc `yes-rate` = số mẫu `yes` / tổng số mẫu trong rule/segment
- `lift` = yes-rate của segment / yes-rate toàn dataset
- Rule có confidence cao nhưng support nhỏ chỉ là exploratory, không dùng làm insight chính

---

## 4. Giải thích biểu đồ & kết quả

### 4.1 `01_target_distribution.png`
- **Thể hiện**: Phân phối target y (yes/no)
- **Cách đọc**: Bar chart (trái) = số lượng, Pie chart (phải) = tỷ lệ %
- **Kết luận**: 88.7% no / 11.3% yes → class imbalance nghiêm trọng (ratio 8:1)
- **Ý nghĩa**: Accuracy đơn thuần không đủ (predict all "no" = 88.7% accuracy nhưng vô dụng). Phải dùng F1/Recall cho class "yes"

### 4.2 `01_categorical_analysis.png`
- **Thể hiện**: Yes-rate theo từng biến categorical (10 biến)
- **Cách đọc**: Stacked bar ngang, đỏ = no, xanh = yes. Thanh dài hơn phía xanh = yes-rate cao
- **Kết luận**: `poutcome=success` có yes-rate cao nhất (~65%). `contact=cellular` tốt hơn `telephone`. Tháng 3, 9, 10 nổi bật
- **Ý nghĩa**: Lịch sử tiếp thị (poutcome) là predictor mạnh — khách từng đăng ký có xu hướng đăng ký lại

### 4.3 `01_numerical_distribution.png`
- **Thể hiện**: Histogram overlay (no=đỏ, yes=xanh) cho 10 biến numerical
- **Cách đọc**: Vùng 2 màu tách biệt = feature phân biệt tốt; chồng chéo = ít phân biệt
- **Kết luận**: `duration` phân biệt rõ nhất. `nr.employed`, `euribor3m` cũng có sự khác biệt
- **Ý nghĩa**: Duration là feature mạnh nhất NHƯNG prediction-time unavailable

### 4.4 `01_correlation_heatmap.png`
- **Thể hiện**: Ma trận tương quan giữa 10 biến numerical
- **Cách đọc**: Đỏ = tương quan dương mạnh, Xanh = tương quan âm mạnh. Số = hệ số Pearson
- **Kết luận**: `euribor3m` ↔ `nr.employed` (r ≈ 0.95) — multicollinearity cao
- **Ý nghĩa**: DT không bị ảnh hưởng bởi multicollinearity (chỉ chọn 1 feature per split), nhưng cần lưu ý khi diễn giải feature importance

### 4.5 `01_boxplots.png`
- **Thể hiện**: Boxplot theo target cho 6 biến chọn lọc
- **Cách đọc**: Box = Q1-Q3, line = median, dots = outliers
- **Kết luận**: Duration median cao hơn rõ rệt ở nhóm yes. Campaign có nhiều outlier

### 4.6 `01_duration_analysis.png`
- **Thể hiện**: 3 góc nhìn về duration — distribution, mean, yes-rate theo duration range
- **Cách đọc**: (Trái) histogram overlay, (Giữa) bar chart mean, (Phải) yes-rate tăng theo duration
- **Kết luận**: Mean 553s (yes) vs 220s (no). Duration > 600s → yes-rate > 50%
- **Ý nghĩa**: Duration rất predictive nhưng chỉ biết sau cuộc gọi → cần Scenario A/B

### 4.7 `02_ablation_study.png`
- **Thể hiện**: So sánh 3 phiên bản data (A, B, B+) trên 4 metrics
- **Cách đọc**: 3 nhóm bar (xanh=A, cam=B, tím=B+), cao hơn = tốt hơn
- **Kết luận**: Duration tăng F1 từ 0.37→0.57. Undersampling tăng Recall nhưng giảm Precision
- **Ý nghĩa**: Chất lượng dữ liệu (có/không duration, balanced/imbalanced) ảnh hưởng lớn hơn việc chọn model → data-centric

### 4.8 `03_feature_importance_A.png` & `03_feature_importance_B.png`
- **Thể hiện**: Top 15 features theo importance (from scratch)
- **Cách đọc**: Bar ngang, dài hơn = quan trọng hơn
- **Kết luận**: Scenario A: duration=50.9%, nr.employed=35.3%. Scenario B: nr.employed=64.2%, pdays=13.6%
- **Ý nghĩa**: Khi loại duration, chỉ số kinh tế vĩ mô (nr.employed, euribor3m) và lịch sử tiếp thị (pdays, poutcome) nổi lên — đây là features mà ngân hàng BIẾT TRƯỚC cuộc gọi

### 4.9 `04_sklearn_tree_A.png`
- **Thể hiện**: Cấu trúc cây sklearn (max_depth=5)
- **Cách đọc**: Mỗi node: điều kiện split, gini, samples, class. Màu đậm = purer
- **Kết luận**: Root split tại duration=303.5 (Scenario A)

### 4.10 `05_confusion_matrices.png`
- **Thể hiện**: 4 confusion matrix (A/B × scratch/sklearn)
- **Cách đọc**: Hàng = Actual, Cột = Predicted. Góc trên-trái = TN, dưới-phải = TP
- **Kết luận**: Scratch ≈ Sklearn. Scenario B có nhiều FN hơn (Recall thấp)
- **Ý nghĩa**: Không có duration, model "thận trọng" hơn — ít predict "yes"

### 4.11 `05_depth_analysis.png`
- **Thể hiện**: (Trái) Metrics vs max_depth, (Phải) Số leaves vs max_depth
- **Cách đọc**: Trục X = depth, Trục Y trái = score, Trục Y phải = complexity
- **Kết luận**: depth=5 (32 leaves) được chọn vì dễ diễn giải và metric ổn định. depth=10 có thể F1 cao hơn nhẹ nhưng cây phức tạp hơn; depth=None (4973 leaves) overfit
- **Ý nghĩa**: Pre-pruning (max_depth) hiệu quả hơn để tránh overfitting

### 4.12 `06_importance_comparison.png`
- **Thể hiện**: Feature importance Scenario A vs B cạnh nhau
- **Cách đọc**: 2 bar chart, xanh=A, cam=B
- **Kết luận**: Duration "che lấp" mọi features khác ở Scenario A
- **Ý nghĩa**: Minh họa vì sao cần Scenario B để rút insight thực tế trước cuộc gọi

### 4.13 `06_segment_heatmap.png`
- **Thể hiện**: Heatmap yes-rate theo job × education
- **Cách đọc**: Màu đậm = yes-rate cao. Chỉ hiện segments có support ≥ 1%
- **Kết luận**: retired+basic.4y = 31% (gấp 2.75× baseline). admin+university = 14.3%
- **Ý nghĩa**: Nhóm hưu trí + học vấn thấp có tỷ lệ đăng ký cao nhất — có thể do thu nhập ổn định + thời gian rảnh

### 4.14 `04_importance_dt_vs_rf.png`
- **Thể hiện**: So sánh Top 15 feature importance giữa DT (1 cây) và RF (100 cây) trên Scenario B
- **Cách đọc**: Bar ngang, dài hơn = quan trọng hơn
- **Kết luận**: DT tập trung vào ít features (nr.employed chiếm ưu thế tuyệt đối), RF phân tán đều hơn
- **Ý nghĩa**: RF ổn định hơn và tổng quát hơn vì mỗi cây nhìn subset features khác nhau

### 4.15 `05_pr_curve.png`
- **Thể hiện**: Precision-Recall curve cho Scenario A và B
- **Cách đọc**: Đường cong càng gần góc trên-phải = mô hình càng tốt. AP (Average Precision) = diện tích dưới đường cong
- **Kết luận**: AP Scenario A cao hơn nhờ duration. Đường baseline (ngang) = tỷ lệ yes trong test set

### 4.16 `05_cv_boxplot.png`
- **Thể hiện**: Phân phối F1 qua 5 folds cho Scratch và Sklearn (cả A và B)
- **Cách đọc**: Box hẹp và std thấp = mô hình ổn định
- **Kết luận**: Scratch ≈ Sklearn qua mọi fold. Std thấp chứng minh kết quả không phụ thuộc cách chia dữ liệu

### 4.17 `05_learning_curve.png`
- **Thể hiện**: F1 Score của Train và Validation khi tăng dần lượng data training
- **Cách đọc**: 2 đường hội tụ = dữ liệu đủ, còn cách xa = cần thêm data hoặc đang overfit
- **Kết luận**: Khoảng cách Train-Validation hẹp → 41k dòng đã đủ cho depth=5

### 4.18 `05_multi_model_f1.png`
- **Thể hiện**: Biểu đồ bar F1 Score của 4 thuật toán (DT Scratch, DT Sklearn, NB, RF) × 2 Scenarios
- **Cách đọc**: Xanh = Scenario A, Cam = Scenario B. Cao hơn = tốt hơn
- **Kết luận**: RF thường ổn định hơn DT đơn lẻ. NB nhanh nhưng giả định independence có thể hạn chế
- **Ý nghĩa**: Cho thấy DT là lựa chọn cân bằng tốt giữa hiệu suất và khả năng diễn giải

### 4.19 `05_cm_extra_models.png`
- **Thể hiện**: Confusion matrix của Naive Bayes và Random Forest trên Scenario B
- **Cách đọc**: Giống như confusion matrix DT, hàng = Actual, cột = Predicted
- **Kết luận**: So sánh phân bố FP/FN giữa NB và RF với DT

## 5. Kịch bản demo nhanh

### 5.1 Flow demo 7 phút

| Thời lượng | Mở notebook | Nói gì |
|------------|-------------|--------|
| 1 phút | NB01 | Giới thiệu bài toán, dataset, class imbalance, biến `duration` |
| 1 phút | NB02 | Nhấn mạnh split trước encoding, Scenario A/B, không scaling, không SMOTE |
| 1 phút | NB03 | Giải thích DT from scratch: Gini, best split, max_depth=5 |
| 1 phút | NB04 | Sklearn DT đối chứng + Naive Bayes + Random Forest: so sánh đa thuật toán |
| 1.5 phút | NB05 | Scratch vs Sklearn, majority baseline, 5-fold CV, PR Curve, biểu đồ đa thuật toán |
| 1.5 phút | NB06 | Rút tri thức: feature importance, rules, segments, giới hạn |

### 5.2 Bốn câu phải nói thật chắc

1. **Duration không dùng cho mô hình thực tế** vì chỉ biết sau cuộc gọi. Vì vậy project có Scenario A để benchmark và Scenario B để realistic.
2. **Accuracy không đủ** vì majority baseline đã 88.7%; phải nhìn Precision/Recall/F1 cho class `yes`.
3. **Insight không phải nhân quả**; các segment/rules là pattern quan sát, cần kiểm chứng thêm nếu triển khai thật.
4. **Imbalance Strategy**: project dùng stratified split/CV, metric cho class `yes`, và chỉ dùng undersampling như ablation để phân tích trade-off.

### 5.3 Ba câu nên tránh

| Không nên nói | Nên nói thay thế |
|---------------|------------------|
| "From scratch giống hoàn toàn sklearn" | "From scratch gần như tương đương sklearn, sai khác rất nhỏ" |
| "Scenario B rất tốt để triển khai thật" | "Scenario B thực tế hơn, nhưng recall còn thấp và cần cải thiện nếu triển khai" |
| "Tháng 3/9/10 làm khách đăng ký nhiều hơn" | "Tháng 3/9/10 có yes-rate cao hơn trong dữ liệu, chưa kết luận nhân quả" |

---

## 6. Câu hỏi vấn đáp

### 6.1 Lý thuyết nền tảng

**Q1: Decision Tree hoạt động như thế nào?**
> DT xây dựng cây từ trên xuống (top-down) theo Hunt's Algorithm:
> 1. Chọn feature + threshold cho split tốt nhất (giảm impurity nhiều nhất)
> 2. Chia data thành 2 nhánh
> 3. Đệ quy cho mỗi nhánh cho đến khi gặp stopping criteria
> Prediction: duyệt cây từ root đến leaf, trả về majority class tại leaf

**Q2: Gini Index là gì? Tính như thế nào?**
> GINI(t) = 1 - Σ[p(i|t)]² với p(i|t) là tỷ lệ class i tại node t.
> Binary: GINI = 1 - p² - (1-p)² = 2p(1-p).
> GINI = 0 → pure (tất cả cùng class). GINI = 0.5 → impure nhất (50/50).
> Ví dụ: node 70% no / 30% yes → GINI = 1 - 0.7² - 0.3² = 0.42

**Q3: Entropy khác Gini thế nào?**
> Entropy H(t) = -Σ p(i|t) × log₂(p(i|t)). Range [0, 1] cho binary.
> Entropy = 0 → pure. Entropy = 1 → impure nhất.
> Khác biệt: Entropy dùng log (chậm hơn), Gini dùng bình phương (nhanh hơn).
> Trong thực tế kết quả gần như tương đương. Paper gốc C4.5 dùng Entropy, CART dùng Gini.

**Q4: Information Gain là gì?**
> IG = Impurity(parent) - Σ(Nⱼ/N) × Impurity(childⱼ)
> Đo mức giảm impurity sau split. IG cao → split tốt. DT chọn split có IG lớn nhất.

**Q5: Overfitting trong DT xảy ra khi nào?**
> Khi cây quá sâu, fit noise trong training data. Dấu hiệu: train accuracy rất cao nhưng test accuracy thấp.
> Giải pháp: Pre-pruning (max_depth, min_samples_split) hoặc Post-pruning (Reduced Error Pruning).
> Trong project: max_depth=5 → 32 leaves, dễ diễn giải. max_depth=None → 4973 leaves, quá phức tạp và có dấu hiệu overfit.

**Q6: Tại sao chọn Gini mà không phải Entropy?**
> (1) Gini tính nhanh hơn (không cần log). (2) Kết quả gần tương đương Entropy trên dataset này. (3) CART (sklearn) mặc định dùng Gini. Nếu muốn chứng minh, có thể chạy cả 2 và so sánh.

### 6.2 Dữ liệu & tiền xử lý

**Q7: Dataset này có gì đặc biệt?**
> (1) Dữ liệu thực từ chiến dịch telemarketing ngân hàng Bồ Đào Nha, giai đoạn May 2008 đến November 2010.
> (2) Kết hợp 3 loại biến: client info (age, job), campaign info (contact, month), economic indicators (euribor3m, nr.employed).
> (3) Class imbalance 89:11 — phản ánh thực tế (ít người đăng ký).
> (4) Biến duration gây prediction-time unavailability.

**Q8: Tại sao không drop "unknown"?**
> "unknown" chiếm ~25% ở một số cột (default, education). Drop → mất quá nhiều data.
> Fill bằng mode → thiên vị class "no" (class đa số). Giữ nguyên → DT tự học pattern "khách thiếu thông tin có xu hướng khác".

**Q9: 12 duplicate rows có ảnh hưởng không?**
> 12/41,188 = 0.03% → gần như không ảnh hưởng kết quả. Nhưng xóa để đảm bảo tính sạch sẽ, tránh bị hỏi "có check duplicate không?"

**Q10: Tại sao stratified split?**
> Với class imbalance 89:11, random split có thể tạo train/test có tỷ lệ khác nhau. Stratified đảm bảo cả train và test đều giữ đúng tỷ lệ 89:11.

### 6.3 Thiết kế mô hình & pipeline

**Q11: Tại sao dùng KDD mà không phải CRISP-DM?**
> KDD (Fayyad 1996) tập trung vào **quy trình khai phá tri thức từ dữ liệu** — đúng tinh thần môn học. CRISP-DM thiên về quy trình dự án (business understanding, deployment). Cả hai tương thích nhau, KDD phù hợp hơn cho bài tập học thuật.

**Q12: Tại sao max_depth=5?**
> (1) depth=3: Recall quá thấp (0.19), cây quá đơn giản.
> (2) depth=5: F1=0.37, Precision=0.65, 32 leaves — cân bằng.
> (3) depth=7: F1 gần bằng (0.36), 99 leaves — phức tạp hơn nhưng không cải thiện rõ.
> (4) depth=10: F1 có thể cao hơn nhẹ, nhưng cây phức tạp hơn nhiều.
> (5) depth=None: 4973 leaves, quá phức tạp và có dấu hiệu overfit.
> → depth=5 không phải tối ưu tuyệt đối về F1, nhưng là lựa chọn tốt nhất xét tỷ lệ performance/complexity và khả năng diễn giải.

**Q13: Tại sao dùng One-Hot Encoding mà không Label Encoding?**
> Label Encoding tạo thứ tự giả (education: 0,1,2,3) → DT split tại threshold có thể gộp nhầm categories.
> One-Hot: mỗi category thành biến 0/1 riêng → DT split chính xác trên từng category.
> Nhược điểm One-Hot: tăng số features (21→63) → nhưng DT xử lý tốt high-dimensional data.

**Q14: Pipeline NB02 có bị leakage không?**
> Không. (1) Split trước encoding. (2) Encoder fit CHỈ trên train. (3) Test chỉ transform. (4) CV: encoder fit per fold. Đây là cách chặt chẽ nhất.

### 6.4 Kết quả & đánh giá

**Q15: Accuracy 90% có tốt không?**
> KHÔNG đủ để kết luận tốt. Majority baseline (predict all "no") đã đạt 88.7%.
> Model chỉ hơn baseline 1.3% accuracy. Phải nhìn F1/Recall: F1=0.37 cho class "yes" — model phát hiện được 26% khách đăng ký. Tốt hơn 0% nhưng chưa tốt cho ứng dụng thực tế.

**Q16: Tại sao Recall Scenario B chỉ 0.26?**
> (1) Class imbalance 89:11 → model thiên về predict "no". (2) Không có duration (feature mạnh nhất). (3) max_depth=5 giới hạn complexity.
> Cải thiện: undersampling tăng Recall lên 0.59 (nhưng giảm Precision 0.65→0.45).

**Imbalance Strategy cần nhớ**
> Project không cố "làm cân bằng dữ liệu bằng mọi giá". Chiến lược đúng là: giữ đánh giá công bằng bằng stratified split/CV, báo Precision/Recall/F1 cho class `yes`, và dùng undersampling như một thí nghiệm phụ để thấy trade-off: Recall tăng nhưng Precision giảm.

**Q17: From scratch vs sklearn có giống nhau không?**
> Gần như tương đương. CV diff < 0.001 trên cả Accuracy và F1.
> Sai khác nhỏ (nếu có): (1) Tie-breaking khi nhiều features có cùng gain. (2) Cách chọn threshold (Python vs C/Cython). (3) Floating point precision.

**Q18: Tại sao không dùng AUC-ROC?**
> AUC-ROC đo khả năng phân biệt tổng thể (ranking), nhưng không phản ánh performance tại threshold cụ thể. Với class imbalance, F1 và Precision/Recall trực quan hơn và liên quan trực tiếp đến quyết định kinh doanh (gọi hay không gọi).

### 6.5 Câu hỏi phản biện

**Q19: Tại sao chọn Decision Tree làm trọng tâm? Các thuật toán khác thế nào?**
> DT là trọng tâm vì: (1) Mục tiêu học thuật: implement from scratch để hiểu sâu bản chất thuật toán. (2) DT dễ diễn giải → rút tri thức dễ hơn.
> Project đã so sánh thêm với **Naive Bayes** (trường phái xác suất) và **Random Forest** (trường phái ensemble) trong NB04+NB05.
> Kết quả: RF có xu hướng ổn định hơn DT đơn lẻ, nhưng DT đơn lẻ vẫn vượt trội về khả năng diễn giải và rút tri thức — đúng mục tiêu KDD.

**Q20: Tại sao không dùng class_weight="balanced"?**
> `class_weight="balanced"` trong sklearn tăng trọng số lỗi của class thiểu số theo tần suất nghịch đảo. Nó không giống hoàn toàn undersampling, nhưng cùng mục tiêu: giảm thiên vị về class "no". Đây là hướng cải thiện tiềm năng, chưa thử trong project này — có thể thêm vào ablation.

**Q21: Unknown values giữ nguyên có đúng không?**
> Có nhiều cách: (1) Drop (mất data). (2) Fill mode (bias). (3) Giữ nguyên (DT tự học). Mỗi cách có trade-off. Trong pipeline này, "unknown" được giữ như một category riêng và sau one-hot sẽ thành feature riêng. Nếu muốn chặt hơn, có thể thử cả 3 cách và so sánh.

**Q22: max_depth=5 chọn thủ công, tại sao không grid search?**
> Đúng, grid search + CV sẽ tối ưu hơn. Trong project: đã phân tích depth=[3,5,7,10,None] thủ công (NB05 §7). Grid search toàn diện hơn (kết hợp max_depth × min_samples_split × criterion) là hướng cải thiện.

**Q23: Correlation ≠ Causation — vậy insight có giá trị không?**
> Có, nhưng ở mức khác nhau: (1) Feature importance → gợi ý yếu tố quan trọng, cần thí nghiệm để xác nhận. (2) Segments → nhóm khách tiềm năng cho target marketing, cần A/B testing để validate. (3) Decision rules → heuristic cho nhân viên tư vấn, không phải quy luật tuyệt đối.

### 6.6 Câu hỏi mở rộng

**Q24: Cải tiến tiềm năng?**
> (1) Đã thực hiện: so sánh với Naive Bayes và Random Forest.
> (2) Grid search hyperparameters (max_depth × min_samples × criterion).
> (3) Feature engineering: tạo biến mới (campaign_per_day, contact_rate).
> (4) Thử class_weight="balanced" thay vì undersampling.
> (5) Cost-sensitive learning: gán chi phí khác nhau cho FP và FN.
> (6) Thử Gradient Boosting (XGBoost, LightGBM) để đẩy hiệu suất lên cao hơn.

**Q25: Ứng dụng thực tế?**
> (1) **Target marketing**: Ưu tiên gọi nhóm retired + basic.4y (lift 2.75×).
> (2) **Campaign timing**: Tháng 3/9/10 có conversion cao trong dữ liệu; đây là pattern quan sát, cần kiểm chứng thêm trước khi ra quyết định.
> (3) **Decision support**: Rules từ DT → checklist cho nhân viên tư vấn.
> (4) **Resource allocation**: Undersampling model (Recall cao) cho chiến dịch "cast wide net"; Baseline model (Precision cao) cho chiến dịch tiết kiệm.

**Q26: Nếu có thêm data, làm gì?**
> (1) Thu thập data gần đây hơn (2020+) để cập nhật model.
> (2) Thêm features: lịch sử giao dịch, digital behavior.
> (3) Tăng minority class bằng cách thu thập thêm (không phải SMOTE).
> (4) Validate model trên thị trường khác (Việt Nam vs Bồ Đào Nha).

**Q27: Quy trình KDD có 6 bước, project thiếu bước nào?**
> Đầy đủ: (1) Selection → NB01, (2) Preprocessing → NB01, (3) Transformation → NB02, (4) Data Mining → NB03+04, (5) Evaluation → NB05, (6) Knowledge → NB06.
> Có thể bổ sung: lặp lại pipeline với cấu hình khác (iteration), nhưng 1 vòng lặp đã đủ cho đồ án.

**Q28: DT from scratch của em khác sklearn ở điểm nào?**
> (1) Cùng thuật toán Hunt, cùng Gini, cùng logic split. (2) Scratch dùng Python thuần → chậm hơn sklearn (C/Cython). (3) Code scratch có nhánh hỗ trợ categorical split trực tiếp nếu truyền `categorical_features`, nhưng trong pipeline chính cả scratch và sklearn đều dùng one-hot để so sánh công bằng. (4) Tie-breaking có thể khác (thứ tự duyệt features). (5) Kết quả gần tương đương (CV diff < 0.001).
