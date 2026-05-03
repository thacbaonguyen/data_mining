# NB06: Knowledge Discovery

> **Bước KDD**: Knowledge (Bước cuối cùng)
> **File**: `notebooks/06_knowledge.ipynb` — 23 cells
> **Mục đích**: Rút tri thức từ mô hình, đưa ra khuyến nghị kinh doanh, ghi nhận giới hạn

---

## Cell 0 [Markdown] — Tiêu đề notebook

## Cell 1 [Code] — Setup môi trường

```python
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Cell 2 [Markdown] — Tiêu đề Load Data

## Cell 3 [Code] — Load dữ liệu và mô hình

```python
with open("data/processed_data.pkl", "rb") as f:    data = pickle.load(f)
with open("data/results_scratch.pkl", "rb") as f:   res_scratch = pickle.load(f)
with open("data/results_sklearn.pkl", "rb") as f:   res_sklearn = pickle.load(f)
```

- Load thêm file CSV gốc để phân tích segment (cần dữ liệu dạng chữ gốc, không phải one-hot)

## Cell 4 [Markdown] — Tiêu đề Feature Importance So sánh

## Cell 5 [Code] — Feature Importance A vs B (cạnh nhau)

```python
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
```

**Giải thích:**
- Vẽ 2 bar chart cạnh nhau: Scenario A (trái) vs Scenario B (phải)
- **Mục đích**: Cho thấy duration "che lấp" mọi features khác ở Scenario A
- Khi loại duration (Scenario B), các features thực tế nổi lên: `nr.employed`, `pdays`, `euribor3m`

**Tri thức rút ra:**
- Chỉ số kinh tế vĩ mô (`nr.employed`, `euribor3m`) ảnh hưởng lớn → Thời điểm kinh tế tốt = khách hàng sẵn sàng đầu tư hơn
- Lịch sử tiếp thị (`pdays`, `poutcome`) quan trọng → Khách từng liên hệ thành công có xu hướng đăng ký lại

**Output**: `report/06_importance_comparison.png`

## Cell 6 [Markdown] — Tiêu đề Decision Rules

## Cell 7 [Code] — Trích xuất Decision Rules từ cây

```python
d_B = data["scenario_B"]
sk = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
sk.fit(d_B["X_train"], d_B["y_train"])
# Duyệt đệ quy từ root đến leaf để rút ra rules
```

**Giải thích:**
- Duyệt qua từng đường đi (path) từ gốc đến lá trên cây Scenario B
- Mỗi path = 1 "luật" (rule) dạng: `NẾU (điều kiện 1) VÀ (điều kiện 2) THÌ → yes/no`
- Lọc chỉ giữ các rules cho class "yes" (khách đăng ký)
- Phân loại rules:
  - **Strong rules** (support ≥ 50 mẫu): Đáng tin cậy, có thể dùng cho khuyến nghị
  - **Exploratory rules** (support < 50): Thú vị nhưng cần kiểm chứng thêm

**Ví dụ rule:**
> NẾU `nr.employed <= 5087.65` VÀ `pdays <= 3.5` VÀ `euribor3m <= 1.266` THÌ → **yes** (confidence 68%, support 156)

## Cell 8 [Markdown] — Tiêu đề Customer Segmentation

## Cell 9 [Code] — Phân khúc khách hàng (Job × Education)

```python
baseline_rate = (df["y"] == "yes").mean()  # ~11.3%
seg = df.groupby(["job", "education"]).agg(
    total=("y", "count"),
    yes_count=("y", lambda x: (x == "yes").sum())
)
seg["yes_rate"] = seg["yes_count"] / seg["total"]
seg["lift"] = seg["yes_rate"] / baseline_rate
```

**Giải thích:**
- Tính tỷ lệ đăng ký (`yes_rate`) cho mỗi nhóm (Job × Education)
- `lift` = yes_rate của nhóm / yes_rate toàn dataset → Lift > 1 = nhóm tốt hơn trung bình
- Lọc chỉ giữ segments có `support ≥ 1%` tổng data (tránh nhóm quá nhỏ)

**Phát hiện:**
- `retired + basic.4y`: yes_rate = 31%, lift = 2.75× (gấp gần 3 lần trung bình)
- `admin + university.degree`: yes_rate = 14.3%

## Cell 10 [Markdown] — Tiêu đề Heatmap

## Cell 11 [Code] — Vẽ Heatmap (Job × Education)

```python
pivot = seg_filtered.pivot_table(index="job", columns="education", values="yes_rate")
sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd")
```

**Giải thích:**
- Bảng nhiệt 2 chiều: Hàng = Nghề nghiệp, Cột = Trình độ học vấn
- Màu đậm = yes_rate cao → Nhóm khách tiềm năng

**Output**: `report/06_segment_heatmap.png`

## Cell 12 [Markdown] — Tiêu đề Segment theo Month × Contact

## Cell 13 [Code] — Phân khúc Month × Contact

```python
seg_mc = df.groupby(["month", "contact"]).agg(...)
```

**Giải thích:**
- Tìm xem tháng nào + phương thức liên hệ nào cho yes-rate cao nhất
- **Phát hiện**: Tháng 3, 9, 10 có yes-rate cao bất thường

## Cell 14 [Markdown] — Giải thích tại sao tháng 3, 9, 10 có yes-rate cao

**Phân tích (KHÔNG phải nhân quả):**
- Tháng 3/9/10 có số cuộc gọi ÍT hơn các tháng khác (may/jun/jul/aug)
- Giả thuyết: Ngân hàng chọn lọc kỹ hơn → Chỉ gọi khách có khả năng cao → yes-rate cao
- Đây là **selection bias trong chiến dịch**, không phải "tháng 3 khiến người ta đăng ký"

## Cell 15 [Markdown] — Tiêu đề Duration Impact

## Cell 16 [Code] — Phân tích tác động Duration

```python
print("Duration (yes):", df[df["y"]=="yes"]["duration"].describe())
print("Duration (no):",  df[df["y"]=="no"]["duration"].describe())
```

**Giải thích:**
- So sánh thống kê duration giữa nhóm yes và no
- Mean: 553s (yes) vs 220s (no)
- **Tri thức**: Cuộc gọi dài ≠ nhân quả "gọi lâu → khách mua". Có thể: khách đã quan tâm → nói chuyện lâu hơn

## Cell 17 [Markdown] — Tiêu đề Data-Centric Mining

## Cell 18 [Code] — Tổng kết Data-Centric Mining

**Giải thích:**
- In bảng tóm tắt Ablation Study:
  - Bỏ duration: F1 giảm 0.20
  - Undersampling: Recall tăng 0.33, Precision giảm 0.20
- **Kết luận**: Chất lượng dữ liệu (Data) ảnh hưởng lớn hơn việc chọn thuật toán (Model)
- Đây là tư tưởng **Data-Centric AI** — Thay vì cố gắng tuning model, hãy cải thiện data

## Cell 19 [Markdown] — Tiêu đề Imbalance Strategy

## Cell 20 [Code] — Tổng kết chiến lược xử lý mất cân bằng

**Giải thích:**
- Project KHÔNG cố "làm cân bằng dữ liệu bằng mọi giá"
- Chiến lược: Stratified split/CV + Metric cho class "yes" + Undersampling chỉ là ablation
- Không dùng SMOTE vì: Tạo điểm trung bình trên one-hot features (0/1) → Vô nghĩa

## Cell 21 [Markdown] — Tiêu đề Kết luận + Khuyến nghị

## Cell 22 [Code] — Kết luận cuối cùng

```python
print("=" * 70)
print("  KẾT LUẬN & KHUYẾN NGHỊ")
print("=" * 70)
```

**Nội dung:**
1. **Tri thức chính**:
   - Nhóm `retired + basic.4y` có tỷ lệ đăng ký 31% (lift 2.75×)
   - Chỉ số kinh tế vĩ mô (nr.employed, euribor3m) ảnh hưởng mạnh nhất
   - Tháng 3/9/10 có yes-rate cao (do chọn lọc chiến dịch)

2. **Khuyến nghị**:
   - Ưu tiên gọi nhóm hưu trí + học vấn basic.4y
   - Xem xét thời điểm kinh tế khi lên kế hoạch chiến dịch
   - Dùng model Precision-cao cho chiến dịch tiết kiệm, Recall-cao cho chiến dịch mở rộng

3. **Giới hạn** (Trung thực ghi nhận):
   - Pattern quan sát, KHÔNG phải nhân quả
   - Dữ liệu từ 2008-2010 (Bồ Đào Nha), chưa validate trên thị trường khác
   - Recall Scenario B còn thấp (0.26) → Bỏ sót 74% khách tiềm năng
   - Chỉ dùng 1 thuật toán chính (DT), ensemble (RF) đã tốt hơn nhưng chưa explore hết
