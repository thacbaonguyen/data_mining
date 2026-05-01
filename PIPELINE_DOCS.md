# 📚 TÀI LIỆU PIPELINE DỰ ÁN — DECISION TREE FROM SCRATCH

> Tài liệu này giải thích **toàn bộ luồng hoạt động** của dự án.
> Đọc theo thứ tự từ trên xuống dưới để hiểu cách mọi thứ kết nối với nhau.

---

## 🗂️ Cấu trúc dự án

```
kpdl/
├── src/                    ← 📦 CODE FROM SCRATCH (6 modules)
│   ├── __init__.py         ← Export các class/function chính
│   ├── criteria.py         ← Tính GINI, Entropy, Information Gain
│   ├── splitter.py         ← Tìm điểm chia tối ưu
│   ├── node.py             ← Cấu trúc 1 nút trong cây
│   ├── tree.py             ← ⭐ CLASS CHÍNH: DecisionTreeClassifier
│   ├── metrics.py          ← Accuracy, Precision, Recall, F1
│   └── visualizer.py       ← Vẽ biểu đồ
├── notebooks/              ← 📓 4 NOTEBOOKS (chạy trên Colab)
│   ├── 01_eda.ipynb        ← Khảo sát + tiền xử lý dữ liệu
│   ├── 02_from_scratch.ipynb ← Chạy DT from scratch
│   ├── 03_sklearn.ipynb    ← Chạy DT bằng sklearn
│   └── 04_comparison.ipynb ← So sánh 2 cách
├── data/                   ← 📊 DỮ LIỆU
│   └── bank-additional/bank-additional-full.csv
└── report/                 ← 📸 HÌNH ẢNH OUTPUT
```

---

## 🔄 LUỒNG HOẠT ĐỘNG TỔNG QUAN

```
[Dataset CSV] 
     ↓ Notebook 1
[Tiền xử lý: xóa null, encoding, scaling]
     ↓ Lưu processed_data.pkl
     ├──→ Notebook 2: From Scratch (dùng src/)
     └──→ Notebook 3: Sklearn
              ↓
         Notebook 4: So sánh → Kết quả cuối cùng
```

---

# PHẦN 1: CÁC MODULE TRONG `src/`

> Đây là phần code **tự viết** (from scratch). 
> Luồng gọi: `tree.py` → gọi `splitter.py` → gọi `criteria.py`
> Kết quả lưu trong `node.py`. Đánh giá bằng `metrics.py`.

---

## 📄 File 1: `criteria.py` — Tiêu chí đánh giá "độ lộn xộn"

### Vai trò
Tính xem 1 nhóm dữ liệu có **thuần khiết** (pure) hay **lộn xộn** (impure).
Thuần khiết = tất cả cùng lớp. Lộn xộn = lẫn lộn nhiều lớp.

### Các function chính

#### `gini_index(y)` — Chỉ số GINI
```python
def gini_index(y):
    # y = danh sách nhãn, ví dụ: ["yes", "yes", "no", "no"]
    
    counter = Counter(y)      # Đếm: {"yes": 2, "no": 2}
    n = len(y)                # Tổng = 4
    
    gini = 1.0
    for count in counter.values():
        p = count / n          # p(yes) = 2/4 = 0.5
        gini -= p ** 2         # gini = 1 - 0.5² - 0.5² = 0.5
    
    return gini
```

**Ý nghĩa:**
- `GINI = 0` → tất cả cùng lớp (tốt nhất)
- `GINI = 0.5` → chia đều 50/50 (tệ nhất cho 2 lớp)

#### `entropy(y)` — Entropy
```python
def entropy(y):
    # Giống GINI nhưng dùng logarithm
    for count in counter.values():
        p = count / n
        ent -= p * np.log2(p)   # Dùng log base 2
    return ent
```

**Ý nghĩa:** Giống GINI nhưng thang đo khác. Entropy = 0 → pure, Entropy = 1 → mixed (2 lớp).

#### `information_gain(y_parent, y_children)` — Lượng thông tin thu được
```python
def information_gain(y_parent, y_children):
    # y_parent = nhãn của node cha
    # y_children = [nhãn_con_trái, nhãn_con_phải]
    
    parent_entropy = entropy(y_parent)           # Entropy trước khi chia
    weighted_child = Σ (nᵢ/n) × entropy(childᵢ) # Entropy sau khi chia
    
    return parent_entropy - weighted_child        # Giảm được bao nhiêu?
```

**Ý nghĩa:** IG lớn = phép chia tốt (giảm nhiều "lộn xộn").

#### `gain_ratio(y_parent, y_children)` — Tỷ lệ lượng thông tin
```python
def gain_ratio(y_parent, y_children):
    ig = information_gain(y_parent, y_children)
    si = split_info(y_children)   # Penalize chia thành quá nhiều nhánh nhỏ
    return ig / si
```

**Ý nghĩa:** Khắc phục nhược điểm của IG khi thuộc tính có quá nhiều giá trị.

---

## 📄 File 2: `node.py` — Cấu trúc 1 nút trong cây

### Vai trò
Mỗi nút trong cây quyết định được lưu bằng class `Node`.

```python
class Node:
    # --- Thông tin phân nhánh ---
    feature = None           # Thuộc tính dùng để chia (ví dụ: "age")
    threshold = None         # Ngưỡng chia (ví dụ: 30.5)
    categories_left = None   # Dùng cho categorical (ví dụ: {"student", "retired"})
    
    # --- Con trái / con phải ---
    left = None              # Node con trái (≤ threshold)
    right = None             # Node con phải (> threshold)
    
    # --- Nếu là lá ---
    is_leaf = False          # Có phải nút lá không?
    prediction = None        # Nhãn dự đoán (ví dụ: "yes")
    
    # --- Metadata ---
    num_samples = 0          # Bao nhiêu mẫu tại node này
    class_distribution = {}  # {"yes": 100, "no": 200}
```

**Ví dụ trực quan:**
```
         [duration ≤ 319.5]     ← Node (internal)
          /              \
    [nr.employed ≤ 5087]   [poutcome_success ≤ 0.5]  ← Nodes
      /         \              /         \
   [no]        [yes]        [no]        [yes]    ← Leaf nodes
```

---

## 📄 File 3: `splitter.py` — Tìm điểm chia tối ưu

### Vai trò
Trả lời câu hỏi: **"Nên chia theo thuộc tính nào? Tại giá trị bao nhiêu?"**

### Luồng hoạt động

```
find_best_split(X, y)
  ├── Với MỖI thuộc tính (cột):
  │     ├── Nếu là số (continuous):
  │     │     → find_best_split_continuous()
  │     └── Nếu là chữ (categorical):
  │           → find_best_split_categorical()
  └── Chọn thuộc tính có GINI/Entropy THẤP NHẤT
```

#### `find_best_split_continuous(X_col, y)` — Chia thuộc tính số
```python
def find_best_split_continuous(X_col, y, criterion='gini'):
    # Ví dụ: X_col = [25, 30, 35, 40, 45]
    #         y     = [no, no, yes, yes, yes]
    
    # Bước 1: Sort theo giá trị
    sorted_indices = np.argsort(X_col)  # [0, 1, 2, 3, 4]
    
    # Bước 2: Thử mọi midpoint
    # Midpoints: 27.5, 32.5, 37.5, 42.5
    for i in range(1, len(X_sorted)):
        threshold = (X_sorted[i] + X_sorted[i-1]) / 2.0  # = 32.5
        
        y_left = y[:i]     # [no, no]
        y_right = y[i:]    # [yes, yes, yes]
        
        # Bước 3: Tính GINI cho mỗi cách chia
        imp = weighted_impurity([y_left, y_right])
        
        # Bước 4: Giữ lại cách chia có GINI thấp nhất
        if imp < best_impurity:
            best_threshold = threshold  # = 32.5
```

**Kết quả:** "Chia tại age ≤ 32.5 là tối ưu"

#### `find_best_split_categorical(X_col, y)` — Chia thuộc tính chữ
```python
def find_best_split_categorical(X_col, y, criterion='gini'):
    # Ví dụ: X_col = ["student", "retired", "admin", "blue-collar"]
    
    # Heuristic: Sort categories theo tỷ lệ "yes"
    # student: 80% yes, retired: 60% yes, admin: 30% yes, blue-collar: 10% yes
    sorted_cats = ["blue-collar", "admin", "retired", "student"]
    
    # Thử chia: {blue-collar} vs {admin, retired, student}
    #           {blue-collar, admin} vs {retired, student}
    #           ... chọn cách chia tốt nhất
```

**Kết quả:** "Chia job ∈ {student, retired} vs còn lại là tối ưu"

---

## 📄 File 4: `tree.py` — ⭐ CLASS CHÍNH

### Vai trò
Đây là file quan trọng nhất. Chứa `DecisionTreeClassifier` với 3 method chính:
- `fit()` → xây dựng cây
- `predict()` → dự đoán
- `prune()` → cắt tỉa

### `fit(X, y)` — Xây dựng cây (Hunt's Algorithm)

```python
def fit(self, X, y, feature_names=None):
    # X = ma trận dữ liệu (32950 × 63)
    # y = nhãn ["yes", "no", "no", ...]
    
    self.root = self._build_tree(X, y, depth=0)  # ← Bắt đầu đệ quy
```

#### `_build_tree(X, y, depth)` — Hàm đệ quy xây cây

```
_build_tree(X, y, depth=0)
│
├── KIỂM TRA ĐIỀU KIỆN DỪNG:
│   ├── Tất cả cùng lớp?          → Tạo leaf, return
│   ├── Đạt max_depth?            → Tạo leaf (majority class), return
│   └── Ít hơn min_samples_split? → Tạo leaf, return
│
├── TÌM BEST SPLIT:
│   └── find_best_split(X, y)     → Trả về: feature, threshold, gain
│
├── Không tìm được split tốt?     → Tạo leaf, return
│
└── CHIA DỮ LIỆU + ĐỆ QUY:
    ├── X_left, y_left   = dữ liệu bên trái  (≤ threshold)
    ├── X_right, y_right  = dữ liệu bên phải (> threshold)
    ├── node.left  = _build_tree(X_left, y_left, depth+1)   ← ĐỆ QUY
    └── node.right = _build_tree(X_right, y_right, depth+1) ← ĐỆ QUY
```

**Code quan trọng:**
```python
def _build_tree(self, X, y, depth):
    node = Node()
    majority_class = Counter(y).most_common(1)[0][0]  # Lớp chiếm đa số
    
    # === ĐIỀU KIỆN DỪNG ===
    if len(set(y)) == 1:                    # Tất cả cùng lớp
        node.is_leaf = True
        node.prediction = majority_class
        return node
    
    if self.max_depth and depth >= self.max_depth:  # Đạt giới hạn
        node.is_leaf = True
        node.prediction = majority_class
        return node
    
    # === TÌM BEST SPLIT ===
    best_split = find_best_split(X, y, ...)
    
    if best_split is None:                  # Không chia được
        node.is_leaf = True
        node.prediction = majority_class
        return node
    
    # === CHIA + ĐỆ QUY ===
    node.feature = best_split['feature']
    node.threshold = best_split['threshold']
    
    mask_left = X[:, node.feature] <= node.threshold
    node.left  = self._build_tree(X[mask_left],  y[mask_left],  depth+1)
    node.right = self._build_tree(X[~mask_left], y[~mask_left], depth+1)
    
    return node
```

### `predict(X)` — Dự đoán

```python
def predict(self, X):
    return [self._predict_one(x, self.root) for x in X]

def _predict_one(self, x, node):
    if node.is_leaf:
        return node.prediction              # Trả về "yes" hoặc "no"
    
    if x[node.feature] <= node.threshold:    # Đi trái
        return self._predict_one(x, node.left)
    else:                                    # Đi phải
        return self._predict_one(x, node.right)
```

**Ví dụ:** Khách hàng mới có `duration=500, nr_employed=5100`:
```
Root: duration ≤ 319.5?  → 500 > 319.5 → ĐI PHẢI
  → nr_employed ≤ 5087?  → 5100 > 5087 → ĐI PHẢI
    → Leaf: prediction = "no"
```

### `prune(X_val, y_val)` — Cắt tỉa (Post-pruning)

```python
def prune(self, X_val, y_val):
    # Duyệt từ DƯỚI LÊN (bottom-up):
    # 1. Thử biến node thành leaf (dùng majority class)
    # 2. Nếu accuracy trên validation KHÔNG GIẢM → giữ pruning
    # 3. Nếu accuracy GIẢM → hoàn tác
```

---

## 📄 File 5: `metrics.py` — Đánh giá mô hình

### Các function

```python
# Confusion Matrix: bảng đếm đúng/sai
confusion_matrix(y_true, y_pred)
# Trả về:
#          Predicted
#          Yes    No
# Actual Yes  TP     FN
#        No   FP     TN

# Accuracy: tỷ lệ dự đoán đúng tổng thể
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision: "Dự đoán yes → bao nhiêu % đúng thật?"
precision = TP / (TP + FP)

# Recall: "Thực tế yes → bao nhiêu % được tìm ra?"
recall = TP / (TP + FN)

# F1-Score: trung bình điều hòa của Precision và Recall
f1 = 2 × Precision × Recall / (Precision + Recall)
```

---

## 📄 File 6: `visualizer.py` — Vẽ biểu đồ

| Function | Vẽ gì |
|---|---|
| `plot_confusion_matrix()` | Heatmap xanh TP/FN/FP/TN |
| `plot_feature_importance()` | Bar chart ngang top 15 features |
| `plot_accuracy_vs_depth()` | Đường Train vs Test (phát hiện overfitting) |
| `plot_comparison_bar()` | 2 cột cạnh nhau: Scratch vs Sklearn |

---

# PHẦN 2: CÁC NOTEBOOKS

## 📓 Notebook 1: `01_eda.ipynb` — Khảo sát dữ liệu

### Luồng
```
Load CSV → Kiểm tra null → Kiểm tra "unknown" → Kiểm tra duplicate
    ↓
8 nhóm biểu đồ (phân phối target, categorical, numeric, boxplot,
                  correlation, tỷ lệ yes/no, duration, age)
    ↓
Phát biểu giả thuyết (H1-H5)
    ↓
Tiền xử lý: One-Hot Encoding (KHÔNG scaling — DT không cần)
    ↓
Train/Test Split (80/20 stratified)
    ↓
Lưu: data/processed_data.pkl   ← FILE NÀY CẦN CHO NOTEBOOK 2, 3, 4
```

> **Lưu ý:** Decision Tree invariant to monotonic transforms → KHÔNG cần StandardScaler.
> Nếu muốn scale, phải split trước, fit scaler trên train, transform test (tránh data leakage).

### Code quan trọng

```python
# One-Hot Encoding: biến "job" = "student" → tạo cột "job_student" = 1
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=False)

# KHÔNG scaling vì Decision Tree chỉ so sánh ngưỡng, không tính khoảng cách.
# Nếu cần scale cho thuật toán khác:
#   scaler = StandardScaler()
#   X_train_scaled = scaler.fit_transform(X_train)  # fit trên TRAIN
#   X_test_scaled  = scaler.transform(X_test)       # transform TEST (KHÔNG fit lại)

# Stratified split: giữ tỷ lệ yes/no giống nhau trong train và test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 📓 Notebook 2: `02_from_scratch.ipynb` — Chạy code tự viết

### Luồng
```
Load processed_data.pkl
    ↓
Minh họa: tính GINI, Entropy bằng tay → verify code đúng
    ↓
Train DT (GINI, depth=5) → Predict → Evaluate → Confusion Matrix
    ↓
Train DT (Entropy, depth=5) → So sánh GINI vs Entropy
    ↓
Thử nhiều max_depth [2,3,4,5,7,10,15,20,None] → Vẽ Overfitting chart
    ↓
Feature Importance → Top 15 features
    ↓
Post-Pruning → So sánh trước/sau pruning
    ↓
Lưu: data/results_scratch.pkl
```

### Code quan trọng

```python
# Import module tự viết
from src.tree import DecisionTreeClassifier

# Tạo model: criterion='gini', max_depth=5
dt = DecisionTreeClassifier(criterion='gini', max_depth=5)

# Huấn luyện: truyền X_train (ma trận số), y_train (nhãn)
dt.fit(X_train, y_train, feature_names=feature_names)

# Dự đoán
y_pred = dt.predict(X_test)

# Đánh giá
from src.metrics import accuracy, f1_score
print(f"Accuracy: {accuracy(y_test, y_pred)}")  # → 0.9185
```

---

## 📓 Notebook 3: `03_sklearn.ipynb` — Chạy bằng thư viện

### Luồng
```
Load processed_data.pkl
    ↓
Sklearn DT (GINI, depth=5) → Predict → Evaluate
    ↓
Sklearn DT (Entropy, depth=5)
    ↓
Vẽ cây bằng sklearn.tree.plot_tree()
    ↓
Depth Analysis (giống Notebook 2)
    ↓
Cross-Validation 5-fold
    ↓
Lưu: data/results_sklearn.pkl
```

### Code quan trọng

```python
from sklearn.tree import DecisionTreeClassifier

# Sklearn chỉ cần 3 dòng:
dt_sk = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_sk.fit(X_train, y_train)
y_pred_sk = dt_sk.predict(X_test)
```

---

## 📓 Notebook 4: `04_comparison.ipynb` — So sánh

### Luồng
```
Load data → Chạy cả 2 model (from scratch + sklearn)
    ↓
BẢNG SO SÁNH: Accuracy, Precision, Recall, F1, Depth, Leaves, Time
    ↓
Bar chart so sánh metrics
    ↓
So sánh Feature Importance (2 chart cạnh nhau)
    ↓
So sánh Accuracy vs Depth (2 đường chồng nhau)
    ↓
So sánh Confusion Matrix (2 heatmap cạnh nhau)
    ↓
KẾT LUẬN: From Scratch ≈ Sklearn → Code đúng! ✅
```

### Kết quả mong đợi
```
═══════════════════════════════════════════════════════════════════
         SO SÁNH FROM SCRATCH vs SKLEARN (GINI, depth=5)
═══════════════════════════════════════════════════════════════════
Metric               From Scratch         Sklearn       Match?
─────────────────────────────────────────────────────────────────
Accuracy                   0.9144          0.9144          ✅
Precision (yes)            0.6534          0.6538          ≈
Recall (yes)               0.5119          0.5108          ≈
F1-Score (yes)             0.5740          0.5735          ≈
Tree Depth                      5               5          ✅
Num Leaves                     31              31          ✅
─────────────────────────────────────────────────────────────────
Train Time                28.33s          0.2567s       ~110x
═══════════════════════════════════════════════════════════════════

→ Accuracy, depth, leaves KHỚP HOÀN TOÀN.
→ Precision/Recall/F1 sai khác rất nhỏ (<0.001) do khác biệt
  tie-breaking/split handling giữa Python thuần và Cython/C.
```

---

## 📌 TÓM TẮT LUỒNG GỌI CODE

```
Notebook 2 gọi:
  dt.fit(X, y)
    └── tree.py: _build_tree(X, y, depth=0)
          └── splitter.py: find_best_split(X, y)
                ├── find_best_split_continuous(X_col, y)
                │     └── criteria.py: weighted_impurity() → gini_index()
                └── find_best_split_categorical(X_col, y)
                      └── criteria.py: weighted_impurity() → gini_index()
          └── Tạo Node (node.py)
          └── Đệ quy _build_tree() cho left/right

  dt.predict(X)
    └── tree.py: _predict_one(x, root)
          └── Duyệt cây: root → left/right → ... → leaf → prediction

  accuracy(y_test, y_pred)
    └── metrics.py: so sánh từng phần tử
```
