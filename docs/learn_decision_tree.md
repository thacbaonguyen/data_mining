# TỰ HỌC DECISION TREE TỪ ĐẦU

> **Đối tượng**: Người có nền tảng Python, không tham gia code thuật toán, cần hiểu để bảo vệ đồ án.
>
> **Thời gian**: 2-3 giờ đọc kỹ.
>
> **Cách đọc**: Đọc lần lượt từ Phần 1 → 6. Mỗi phần có ví dụ tính tay — hãy tự tính theo.

---

## Phần 1: Decision Tree là gì? (15 phút)

### 1.1 Ý tưởng cốt lõi

Decision Tree (Cây Quyết Định) là một **cây if/else tự động**. Thay vì bạn viết:

```python
if refund == "Yes":
    return "Không gian lận"
elif marital_status == "Married":
    return "Không gian lận"
elif taxable_income > 80000:
    return "Gian lận"
else:
    return "Không gian lận"
```

Thuật toán DT sẽ **tự tìm ra** các điều kiện if/else tốt nhất từ dữ liệu.

### 1.2 Cấu trúc cây

```
         [Node gốc]          ← Root: điều kiện đầu tiên
         /         \
    [Node]       [Node]       ← Internal nodes: điều kiện tiếp
    /    \          |
 [Lá]  [Lá]      [Lá]        ← Leaf nodes: kết quả dự đoán
```

**3 loại node:**

| Loại | Vai trò | Ví dụ |
|------|---------|-------|
| **Root** | Điều kiện quan trọng nhất | `duration <= 303.5?` |
| **Internal** | Điều kiện phụ | `nr.employed <= 5087?` |
| **Leaf** | Kết quả cuối cùng | → predict "yes" hoặc "no" |

### 1.3 Trong code: `src/node.py`

```python
class Node:
    # Thông tin split
    self.feature_name = None      # Tên thuộc tính (vd: "duration")
    self.threshold = None         # Ngưỡng (vd: 303.5)
    self.left = None              # Nhánh trái (≤ threshold)
    self.right = None             # Nhánh phải (> threshold)

    # Leaf info
    self.is_leaf = False          # Có phải nút lá không?
    self.prediction = None        # Kết quả dự đoán ("yes"/"no")

    # Metadata
    self.impurity = None          # Độ "lộn xộn" tại node
    self.num_samples = 0          # Bao nhiêu mẫu ở node này
    self.class_distribution = {}  # {"yes": 30, "no": 70}
```

**Hiểu đơn giản**: Mỗi node giống 1 "phòng". Phòng trung gian có câu hỏi (điều kiện). Phòng cuối cùng (lá) có câu trả lời.

---

## Phần 2: Gini Index — Đo "độ lộn xộn" (30 phút)

### 2.1 Vấn đề cần giải quyết

Khi xây cây, tại mỗi node ta phải chọn **thuộc tính nào để split**. Cần một thước đo để so sánh các cách split → đó là **Gini Index**.

### 2.2 Công thức

```
GINI(t) = 1 - Σ [p(i|t)]²
```

Trong đó `p(i|t)` là **tỷ lệ** class i tại node t.

**Với binary (2 class):**

```
GINI = 1 - p(yes)² - p(no)²
```

### 2.3 Ví dụ tính tay ⭐

**Node A**: 70 "no" + 30 "yes" (tổng 100)

```
p(no)  = 70/100 = 0.7
p(yes) = 30/100 = 0.3

GINI = 1 - (0.7)² - (0.3)²
     = 1 - 0.49 - 0.09
     = 0.42
```

**Node B**: 100 "no" + 0 "yes" (pure — tất cả cùng class)

```
GINI = 1 - (1.0)² - (0.0)²
     = 1 - 1 - 0
     = 0.0  ← Pure! Không lộn xộn gì.
```

**Node C**: 50 "no" + 50 "yes" (tệ nhất — 50/50)

```
GINI = 1 - (0.5)² - (0.5)²
     = 1 - 0.25 - 0.25
     = 0.5  ← Lộn xộn nhất!
```

### 2.4 Quy tắc nhớ

| GINI | Ý nghĩa |
|------|---------|
| **0** | Hoàn hảo — tất cả cùng class |
| **0.5** | Tệ nhất — 50/50, không phân biệt được |
| **Càng nhỏ** | Càng tốt — node càng "sạch" |

### 2.5 Trong code: `src/criteria.py` dòng 18-50

```python
def gini_index(y):
    counter = Counter(y)        # Đếm: {"no": 70, "yes": 30}
    n = len(y)                  # Tổng: 100

    gini = 1.0
    for count in counter.values():
        p = count / n           # p(no) = 0.7, p(yes) = 0.3
        gini -= p ** 2          # 1.0 - 0.49 - 0.09 = 0.42

    return gini
```

**Ánh xạ**: Công thức `1 - Σ p²` → code `gini = 1.0` rồi `gini -= p ** 2` trong vòng for.

### 2.6 Entropy — Đo theo cách khác (biết qua là đủ)

```
Entropy(t) = -Σ p(i|t) × log₂(p(i|t))
```

- Entropy = 0 → pure
- Entropy = 1 → tệ nhất (binary)
- Kết quả **gần giống Gini**, nhưng tính chậm hơn (phải dùng log)

**Dự án dùng Gini** vì: (1) nhanh hơn, (2) CART dùng Gini, (3) kết quả tương đương.

---

## Phần 3: Chọn Best Split — Bước quan trọng nhất (30 phút)

### 3.1 Ý tưởng

Tại mỗi node, thuật toán **thử mọi cách chia** và chọn cách **giảm Gini nhiều nhất**.

### 3.2 Ví dụ tính tay ⭐

**Dữ liệu ban đầu** (100 khách hàng):

```
Trước split: 70 "no" + 30 "yes"
GINI(trước) = 0.42
```

**Thử split 1**: `duration <= 300`

```
Nhánh trái (≤ 300):  60 "no" + 5 "yes"  (65 mẫu)
  → GINI(trái) = 1 - (60/65)² - (5/65)² = 0.142

Nhánh phải (> 300):  10 "no" + 25 "yes"  (35 mẫu)
  → GINI(phải) = 1 - (10/35)² - (25/35)² = 0.408

GINI(sau split) = (65/100) × 0.142 + (35/100) × 0.408
                = 0.092 + 0.143
                = 0.235

Gain = GINI(trước) - GINI(sau) = 0.42 - 0.235 = 0.185 ✓
```

**Thử split 2**: `age <= 40`

```
Nhánh trái (≤ 40):  35 "no" + 15 "yes"  (50 mẫu)
  → GINI(trái) = 1 - (35/50)² - (15/50)² = 0.420

Nhánh phải (> 40):  35 "no" + 15 "yes"  (50 mẫu)
  → GINI(phải) = 0.420

GINI(sau split) = (50/100) × 0.420 + (50/100) × 0.420
                = 0.420

Gain = 0.42 - 0.420 = 0.000 ✗ (vô ích!)
```

**→ Chọn split 1** (`duration <= 300`) vì Gain = 0.185 > 0.000

### 3.3 Với thuộc tính continuous (số)

Cách tìm ngưỡng tối ưu:

```
Bước 1: Sort giá trị: [55, 65, 72, 80, 87, 92, 97, 110]
Bước 2: Thử midpoint giữa mỗi cặp: [60, 68.5, 76, 83.5, 89.5, 94.5, 103.5]
Bước 3: Tính GINI cho mỗi midpoint
Bước 4: Chọn midpoint có GINI nhỏ nhất
```

### 3.4 Với thuộc tính categorical (phân loại)

Trong dự án, categorical đã được **One-Hot Encoding** → chuyển thành 0/1 → xử lý như continuous.

Ví dụ: `job = admin` → cột `job_admin` có giá trị 0 hoặc 1. Split `job_admin <= 0.5` nghĩa là "job có phải admin không?"

### 3.5 Trong code: `src/splitter.py`

**Hàm chính** `find_best_split()` (dòng 162-256):

```python
def find_best_split(X, y, ...):
    # Duyệt TẤT CẢ features
    for feat_idx in range(n_features):

        # Với mỗi feature, tìm threshold tốt nhất
        threshold, imp = find_best_split_continuous(X_col, y)

        # So sánh với best hiện tại
        if imp < best_impurity:
            best_impurity = imp
            best_split = {...}  # Lưu lại

    return best_split  # Trả về split tốt nhất
```

**Hàm con** `find_best_split_continuous()` (dòng 23-79):

```python
def find_best_split_continuous(X_col, y):
    # 1. Sort theo giá trị
    sorted_indices = np.argsort(X_col)

    # 2. Thử mọi midpoint
    for i in range(1, len(X_sorted)):
        threshold = (X_sorted[i] + X_sorted[i-1]) / 2.0

        # 3. Chia dữ liệu
        y_left = y_sorted[:i]
        y_right = y_sorted[i:]

        # 4. Tính Gini có trọng số
        imp = weighted_impurity([y_left, y_right])

        # 5. Giữ lại nếu tốt nhất
        if imp < best_impurity:
            best_threshold = threshold
```

**Tóm lại**: Sort → thử mọi midpoint → chọn tốt nhất. Đó là tất cả.

---

## Phần 4: Hunt's Algorithm — Xây cây đệ quy (30 phút)

### 4.1 Pseudocode (quan trọng nhất — phải thuộc)

```
HÀM build_tree(data, depth):

    # BƯỚC 1: Kiểm tra điều kiện dừng
    NẾU tất cả mẫu cùng class:
        → Trả về Lá(class đó)

    NẾU depth >= max_depth:
        → Trả về Lá(class chiếm đa số)

    NẾU số mẫu < min_samples_split:
        → Trả về Lá(class chiếm đa số)

    # BƯỚC 2: Tìm cách split tốt nhất
    best_split = tìm_split(data)      ← Phần 3

    NẾU không tìm được split tốt:
        → Trả về Lá(class chiếm đa số)

    # BƯỚC 3: Chia dữ liệu và đệ quy
    data_trái, data_phải = chia(data, best_split)

    node.left  = build_tree(data_trái, depth + 1)   ← Đệ quy!
    node.right = build_tree(data_phải, depth + 1)    ← Đệ quy!

    Trả về node
```

### 4.2 Ví dụ minh họa từng bước ⭐

**Data đơn giản** (10 khách hàng):

| # | duration | nr.employed | y |
|---|----------|-------------|---|
| 1 | 100 | 5000 | no |
| 2 | 150 | 5100 | no |
| 3 | 200 | 5200 | no |
| 4 | 250 | 4900 | no |
| 5 | 300 | 5300 | no |
| 6 | 350 | 5000 | no |
| 7 | 400 | 4800 | yes |
| 8 | 500 | 5100 | yes |
| 9 | 600 | 4700 | yes |
| 10 | 700 | 5000 | yes |

**Depth 0 (Root)**: 6 "no" + 4 "yes", GINI = 0.48

```
Thử split trên duration:
  - duration <= 325: trái=[6no,0yes], phải=[0no,4yes]
    → GINI = 6/10 × 0.0 + 4/10 × 0.0 = 0.0  ← PERFECT!

→ Chọn: duration <= 325
```

```
          duration <= 325?
          /              \
    [6no, 0yes]      [0no, 4yes]
    GINI = 0.0       GINI = 0.0
    → predict NO     → predict YES
    (Lá — pure)      (Lá — pure)
```

Cây chỉ cần 1 split vì data đơn giản. Với data thật (41K mẫu), cây sẽ sâu hơn (max_depth=5 → tối đa 32 lá).

### 4.3 Điều kiện dừng (Stopping Criteria)

| Điều kiện | Tham số | Tác dụng |
|-----------|---------|----------|
| Tất cả cùng class | — | Không cần split nữa |
| Đạt giới hạn sâu | `max_depth=5` | Tránh cây quá phức tạp |
| Quá ít mẫu | `min_samples_split=2` | Không split node nhỏ |
| Gain quá thấp | `min_impurity_decrease` | Bỏ split vô ích |

### 4.4 Trong code: `src/tree.py` hàm `_build_tree()` (dòng 108-205)

```python
def _build_tree(self, X, y, depth):
    # Majority class (class chiếm đa số)
    majority_class = Counter(y).most_common(1)[0][0]

    # === ĐIỀU KIỆN DỪNG ===

    # 1. Tất cả cùng lớp
    if len(set(y)) == 1:
        node.is_leaf = True
        node.prediction = majority_class
        return node

    # 2. Đạt max_depth
    if self.max_depth is not None and depth >= self.max_depth:
        node.is_leaf = True
        node.prediction = majority_class
        return node

    # 3. Không đủ mẫu
    if len(y) < self.min_samples_split:
        node.is_leaf = True
        node.prediction = majority_class
        return node

    # === TÌM BEST SPLIT ===
    best_split = find_best_split(X, y, ...)

    # 4. Không tìm được split tốt
    if best_split is None:
        node.is_leaf = True
        node.prediction = majority_class
        return node

    # === CHIA DỮ LIỆU & ĐỆ QUY ===
    mask_left = X[:, node.feature] <= node.threshold
    X_left, y_left = X[mask_left], y[mask_left]
    X_right, y_right = X[~mask_left], y[~mask_left]

    node.left  = self._build_tree(X_left, y_left, depth + 1)   # Đệ quy!
    node.right = self._build_tree(X_right, y_right, depth + 1)  # Đệ quy!

    return node
```

**Ánh xạ**: Pseudocode ở 4.1 → code ở đây **gần như 1-1**. Nếu hiểu pseudocode thì đọc code rất dễ.

---

## Phần 5: Predict & Feature Importance (15 phút)

### 5.1 Predict — Dự đoán cho mẫu mới

Cực kỳ đơn giản: **đi từ root xuống leaf**, rẽ trái/phải theo điều kiện.

```python
def _predict_one(self, x, node):
    # Đến lá → trả kết quả
    if node.is_leaf:
        return node.prediction

    # Rẽ trái nếu thỏa điều kiện
    if x[node.feature] <= node.threshold:
        return self._predict_one(x, node.left)
    else:
        return self._predict_one(x, node.right)
```

**Ví dụ**: Khách hàng mới có `duration=450`:

```
Root: duration <= 303.5?  → 450 > 303.5 → rẽ PHẢI
Node: nr.employed <= 5087? → 5000 ≤ 5087 → rẽ TRÁI
Leaf: → predict "yes"
```

### 5.2 Feature Importance — Tính mức quan trọng

**Ý tưởng**: Feature nào giúp **giảm Gini nhiều nhất** → quan trọng nhất.

```
importance(feature) = Σ (n_samples × gain) tại mọi node dùng feature đó
```

Sau đó **normalize** để tổng = 1.0.

**Kết quả trong dự án:**

| Feature | Scenario A | Scenario B |
|---------|-----------|-----------|
| duration | **50.9%** | — (đã loại) |
| nr.employed | 35.3% | **64.2%** |
| pdays | — | 13.6% |

→ Duration "che lấp" mọi feature khác ở Scenario A. Loại duration (Scenario B) → thấy rõ feature thực sự hữu ích.

---

## Phần 6: Overfitting & Pruning (15 phút)

### 6.1 Overfitting là gì?

Cây quá sâu → **học thuộc** data training, kể cả nhiễu → dự đoán tệ trên data mới.

| max_depth | Số lá | Ý nghĩa |
|-----------|-------|---------|
| 3 | 8 | Quá đơn giản, bỏ sót pattern |
| **5** | **32** | **Cân bằng** — dự án chọn cái này |
| 10 | ~200 | Phức tạp hơn nhưng cải thiện ít |
| None | 4973 | Quá phức tạp → overfit |

### 6.2 Pre-pruning (dự án dùng cách này)

**Chặn cây trước khi nó mọc quá sâu** bằng `max_depth=5`.

Tại sao chọn 5?
- depth=3: F1 quá thấp (0.19)
- depth=5: F1=0.37, 32 lá — đủ để diễn giải
- depth=7: F1 gần bằng nhưng 99 lá — phức tạp hơn
- → depth=5 là **điểm cân bằng** giữa hiệu quả và khả năng diễn giải

### 6.3 Post-pruning (code hỗ trợ nhưng không dùng trong pipeline chính)

1. Mọc cây hoàn chỉnh (không giới hạn depth)
2. Từ dưới lên, thử chặt bỏ từng nhánh
3. Nếu accuracy trên validation set không giảm → giữ việc chặt

---

## Phần 7: Tổng kết — Bản đồ tư duy

```
Decision Tree
│
├── 1. ĐO ĐỘ LỘN XỘN
│   └── Gini Index = 1 - Σ p²
│       ├── = 0: pure (tốt)
│       └── = 0.5: tệ nhất (binary)
│
├── 2. CHỌN BEST SPLIT
│   ├── Thử mọi feature × mọi threshold
│   ├── Tính Gini có trọng số sau split
│   └── Chọn split giảm Gini nhiều nhất
│
├── 3. XÂY CÂY ĐỆ QUY (Hunt's Algorithm)
│   ├── Kiểm tra điều kiện dừng
│   ├── Tìm best split
│   ├── Chia data → đệ quy 2 nhánh
│   └── Lặp lại cho đến khi dừng
│
├── 4. DỰ ĐOÁN
│   └── Đi từ root → leaf theo điều kiện
│
├── 5. FEATURE IMPORTANCE
│   └── Feature giảm Gini nhiều nhất = quan trọng nhất
│
└── 6. CHỐNG OVERFIT
    ├── Pre-pruning: max_depth = 5
    └── Post-pruning: chặt nhánh từ dưới lên
```

---

## Phần 8: Ánh xạ File → Lý thuyết

| File | Phần lý thuyết | Đọc gì |
|------|----------------|--------|
| `src/node.py` | Phần 1.3 | Class Node — 56 dòng, rất ngắn |
| `src/criteria.py` | Phần 2 | Hàm `gini_index()` — 12 dòng logic |
| `src/splitter.py` | Phần 3 | Hàm `find_best_split_continuous()` — vòng for thử midpoint |
| `src/tree.py` | Phần 4 | Hàm `_build_tree()` — đệ quy Hunt's Algorithm |
| `src/metrics.py` | Đánh giá | Accuracy, Precision, Recall, F1 — dùng ở NB05 |

---

## Phần 9: 10 câu hỏi bảo vệ hay gặp nhất

### Q1: Decision Tree hoạt động thế nào?
> Xây cây từ trên xuống (Hunt's Algorithm): tại mỗi node, tìm thuộc tính + ngưỡng giảm Gini nhiều nhất, chia data thành 2 nhánh, đệ quy cho đến khi gặp điều kiện dừng.

### Q2: Gini Index là gì? Tính thế nào?
> GINI = 1 - Σ p². Node 70no/30yes → GINI = 1 - 0.49 - 0.09 = 0.42. GINI = 0 là pure, 0.5 là tệ nhất.

### Q3: Tại sao dùng Gini mà không Entropy?
> (1) Gini nhanh hơn (không cần log). (2) Kết quả gần tương đương. (3) CART mặc định dùng Gini.

### Q4: Overfitting xảy ra khi nào?
> Cây quá sâu → fit noise. Dấu hiệu: train acc cao, test acc thấp. Giải pháp: max_depth=5.

### Q5: Tại sao max_depth=5?
> depth=3 quá đơn giản (F1=0.19). depth=5 cân bằng (F1=0.37, 32 lá). depth=None overfit (4973 lá).

### Q6: From scratch vs sklearn khác nhau ở đâu?
> Cùng thuật toán, cùng Gini, cùng logic. Sai khác < 0.001 do tie-breaking và floating point.

### Q7: Feature importance tính thế nào?
> Importance = tổng (n_samples × gain) tại mọi node dùng feature đó, rồi normalize.

### Q8: Tại sao có 2 Scenario (A/B)?
> Duration chỉ biết SAU cuộc gọi. A = benchmark (có duration), B = thực tế (không duration).

### Q9: Accuracy 90% có tốt không?
> KHÔNG đủ kết luận. Majority baseline đã 88.7%. Phải nhìn F1/Recall cho class "yes".

### Q10: Cải tiến được gì?
> (1) Random Forest/XGBoost. (2) Grid search hyperparameters. (3) class_weight="balanced".
