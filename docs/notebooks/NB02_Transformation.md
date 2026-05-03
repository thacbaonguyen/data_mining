# NB02: Transformation

> **Bước KDD**: Transformation
> **File**: `notebooks/02_transformation.ipynb` — 17 cells
> **Mục đích**: Chia dữ liệu, mã hóa, tạo Scenario A/B, chạy Ablation Study

---

## Cell 0 [Markdown] — Tiêu đề notebook

Giới thiệu bước KDD thứ 3: Transformation. Tóm tắt luồng: Split trước → Encode → Scenario A/B → Ablation Study.

## Cell 1 [Code] — Setup môi trường

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as SkDT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

**Giải thích:**
- `train_test_split`: Hàm chia dữ liệu thành tập Train/Test
- `OneHotEncoder`: Bộ mã hóa biến categorical thành dạng 0/1
- `SkDT`: Import sklearn Decision Tree để dùng trong Ablation Study
- 4 hàm metric: Để đo hiệu suất mô hình trong Ablation

## Cell 2 [Markdown] — Tiêu đề Load data

## Cell 3 [Code] — Load & Clean dữ liệu

```python
df = pd.read_csv("data/bank-additional/bank-additional-full.csv", sep=";")
df = df.drop_duplicates().reset_index(drop=True)
```

**Giải thích:**
- Load lại dữ liệu gốc từ file CSV
- Xóa 12 dòng trùng lặp (giống NB01)
- `reset_index(drop=True)`: Đánh lại số thứ tự dòng từ 0

## Cell 4 [Markdown] — Giải thích tại sao Split TRƯỚC Encoding

> Encoder/imputer chỉ fit trên train. Test chỉ transform.
> Decision Tree KHÔNG cần scaling.

Đây là nguyên tắc chống Data Leakage quan trọng nhất trong pipeline.

## Cell 5 [Code] — Train/Test Split 80/20 Stratified

```python
X = df.drop(columns=["y"])      # Tách cột features (20 cột)
y = df["y"]                      # Tách cột target (1 cột)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Giải thích từng tham số:**
- `test_size=0.2`: 20% dữ liệu làm Test, 80% làm Train
- `random_state=42`: Số ngẫu nhiên cố định → Mỗi lần chạy ra cùng kết quả (reproducible)
- `stratify=y`: **Quan trọng nhất** — Đảm bảo tỷ lệ yes/no ở cả Train và Test đều giữ nguyên 89%/11%

**Kết quả:**
- Train: ~32,940 dòng, Test: ~8,236 dòng
- Cả 2 tập đều có ~89% no / ~11% yes

## Cell 6 [Markdown] — Tiêu đề Encoding

## Cell 7 [Code] — One-Hot Encoding (Fit on Train, Transform Test)

```python
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
enc.fit(X_train[cat_cols])                    # CHỈ fit trên Train
train_cat_enc = enc.transform(X_train[cat_cols])  # Transform Train
test_cat_enc = enc.transform(X_test[cat_cols])    # Transform Test
```

**Giải thích từng tham số:**
- `sparse_output=False`: Trả về mảng numpy thông thường (không phải sparse matrix)
- `handle_unknown="ignore"`: Nếu Test có category mới chưa thấy ở Train → Tạo vector toàn số 0 (không báo lỗi)
- `drop=None`: Giữ tất cả cột one-hot (không drop cột đầu tiên)

**Quy trình:**
1. Tìm 10 cột categorical
2. `fit()` trên Train: Encoder ghi nhớ danh sách tất cả categories
3. `transform()` Train: Biến chữ → số 0/1
4. `transform()` Test: Dùng danh sách từ bước 2 để biến chữ → số 0/1
5. Ghép cột categorical đã encode + cột numerical → Bảng dữ liệu hoàn chỉnh

**Kết quả**: 20 cột gốc → ~63 cột sau one-hot encoding

## Cell 8 [Markdown] — Giải thích Scenario A/B

| Scenario | Duration | Mục đích |
|---|---|---|
| **A: Benchmark** | Có | So sánh thuật toán |
| **B: Realistic** | Không | Mô hình thực tế |

## Cell 9 [Code] — Tạo Scenario A và B

```python
dur_cols = [c for c in X_train_final.columns if "duration" in c.lower()]
# Scenario A: giữ tất cả cột (bao gồm duration)
# Scenario B: drop cột duration
X_train_B = X_train_final.drop(columns=dur_cols)
```

**Giải thích:**
- Tìm tất cả cột liên quan đến `duration` (sau one-hot chỉ có 1 cột vì duration là biến số)
- **Scenario A**: Giữ nguyên tất cả features → Benchmark thuật toán
- **Scenario B**: Xóa cột duration → Mô phỏng thực tế (không biết thời lượng cuộc gọi trước khi gọi)
- Mỗi Scenario lưu: `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`

## Cell 10 [Markdown] — Tiêu đề Ablation Study

> B+ (Undersampling) là thí nghiệm BỔ SUNG, không phải scenario chính.

## Cell 11 [Code] — Ablation Study (Đo tác động từng bước)

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_us, y_train_us = rus.fit_resample(data_B["X_train"], data_B["y_train"])
```

**Giải thích:**
- **Mục đích**: Chứng minh chất lượng dữ liệu ảnh hưởng mô hình hơn việc chọn thuật toán
- **3 kịch bản thí nghiệm:**
  1. **A**: Có duration (dữ liệu tốt nhất)
  2. **B**: Không duration (dữ liệu thực tế)
  3. **B+**: Không duration + Undersampling (cân bằng lại class)
- `RandomUnderSampler`: Xóa bớt ngẫu nhiên mẫu class đa số (no) để cân bằng với class thiểu số (yes)
- Train cùng 1 mô hình sklearn DT (`max_depth=5`) trên 3 phiên bản data
- So sánh 4 metric: Accuracy, Precision, Recall, F1

**Kết quả Ablation:**

| Scenario | Acc | Prec | Rec | F1 |
|----------|-----|------|-----|-----|
| A: có duration | 0.9144 | 0.6534 | 0.5119 | 0.5740 |
| B: không duration | 0.9009 | 0.6522 | 0.2586 | 0.3704 |
| B+: undersample | 0.8719 | 0.4480 | 0.5894 | 0.5091 |

**Kết luận Data-Centric:**
- Duration giảm F1 từ 0.57 → 0.37 (ΔF1 = 0.20) → Dữ liệu quyết định hiệu suất
- Undersampling tăng Recall (0.26→0.59) nhưng giảm Precision (0.65→0.45) → Trade-off

## Cell 12 [Markdown] — Tiêu đề Ablation Visualization

## Cell 13 [Code] — Biểu đồ Ablation

- Vẽ grouped bar chart: 3 nhóm bar (A, B, B+) × 4 metric
- Màu: Xanh = A, Cam = B, Tím = B+

**Output**: `report/02_ablation_study.png`

## Cell 14 [Markdown] — Tiêu đề Lưu dữ liệu

## Cell 15 [Code] — Lưu processed data

```python
save_data = {
    "scenario_A": data_A,
    "scenario_B": data_B,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "encoder": enc,
}
with open("data/processed_data.pkl", "wb") as f:
    pickle.dump(save_data, f)
```

**Giải thích:**
- Đóng gói toàn bộ dữ liệu đã xử lý vào 1 file pickle
- File này sẽ được NB03, NB04, NB05 load lên để train/evaluate
- Đồng thời lưu `raw_split.pkl` chứa dữ liệu raw chưa encode (dùng cho CV per fold ở NB05)

**Output:**
- `data/processed_data.pkl`: Dữ liệu đã encode, chia Scenario A/B
- `data/raw_split.pkl`: Dữ liệu raw, dùng cho Cross-Validation

## Cell 16 [Markdown] — Tổng kết NB02

| Chỉ số | Giá trị |
|---|---|
| Train/Test split | 80/20 stratified |
| Encoding | One-Hot (fit on train only) |
| Scaling | Không (DT không cần) |
| Scenario A | Có duration, benchmark |
| Scenario B | Không duration, realistic |
| B+ (ablation only) | Undersampling, thí nghiệm bổ sung |
