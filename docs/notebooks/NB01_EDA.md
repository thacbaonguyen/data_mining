# NB01: Problem Understanding + Selection + EDA + Preprocessing

> **Bước KDD**: Selection & Preprocessing
> **File**: `notebooks/01_eda.ipynb` — 21 cells
> **Mục đích**: Hiểu bài toán, chọn dataset, khám phá dữ liệu, tiền xử lý cơ bản

---

## Cell 0 [Markdown] — Tiêu đề notebook

Giới thiệu notebook thuộc bước 1-2 của quy trình KDD (Selection & Preprocessing).

## Cell 1 [Code] — Setup môi trường

```python
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

**Giải thích chi tiết:**
- `numpy`: Thư viện tính toán số học (mảng, ma trận)
- `pandas`: Thư viện xử lý dữ liệu dạng bảng (DataFrame)
- `matplotlib.pyplot`: Thư viện vẽ biểu đồ cơ bản
- `seaborn`: Thư viện vẽ biểu đồ thống kê đẹp hơn matplotlib
- `PROJECT_ROOT`: Đặt thư mục gốc để mọi đường dẫn file nhất quán
- `REPORT_DIR = 'report'`: Thư mục lưu tất cả ảnh biểu đồ xuất ra

## Cell 2 [Markdown] — Problem Understanding

Mô tả bài toán:
- **Đầu vào**: Thông tin khách hàng ngân hàng (tuổi, nghề nghiệp, học vấn...)
- **Đầu ra**: Dự đoán `y` = yes (đăng ký tiền gửi) hay no (từ chối)
- **Mục tiêu học thuật**: Code Decision Tree từ đầu + so sánh với thư viện
- **Mục tiêu khai phá**: Tìm nhóm khách hàng tiềm năng

## Cell 3 [Markdown] — Selection: Tại sao chọn UCI Bank Marketing

Bảng mô tả dataset:
- **Nguồn**: UCI Machine Learning Repository (uy tín học thuật)
- **Kích thước**: 41,188 dòng × 21 cột
- **Thách thức**: Class imbalance (89% no / 11% yes), biến `duration` gây prediction-time unavailability

## Cell 4 [Code] — Load dữ liệu & xem phân phối Target

```python
df = pd.read_csv("data/bank-additional/bank-additional-full.csv", sep=";")
```

**Giải thích:**
- File CSV này dùng dấu chấm phẩy (`;`) làm phân cách thay vì dấu phẩy thông thường
- In ra kích thước dataset (41,188 × 21)
- In ra phân phối target `y`: bao nhiêu `yes`, bao nhiêu `no`
- **Kết quả**: 36,548 no (88.7%) / 4,640 yes (11.3%)

## Cell 5 [Code] — Thông tin tổng quan

- In ra số dòng, số cột
- Phân loại cột thành 2 nhóm:
  - **Categorical (10 cột)**: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome
  - **Numerical (11 cột)**: age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed, y
- Gọi `.describe()` để xem thống kê cơ bản (min, max, mean, std) cho các cột số

## Cell 6 [Markdown] — Tiêu đề Preprocessing

## Cell 7 [Code] — Kiểm tra Null, Duplicate, Unknown

**3 bước kiểm tra chất lượng dữ liệu:**

1. **Null check** (`df.isnull().sum().sum()`):
   - Kết quả: 0 giá trị null → Dataset sạch, không cần fill missing
2. **Duplicate check** (`df.duplicated().sum()`):
   - Kết quả: 12 dòng trùng lặp → Xóa đi, còn 41,176 dòng
3. **Unknown check**:
   - Đếm số lượng giá trị `"unknown"` trong từng cột categorical
   - Quyết định: **Giữ nguyên** "unknown" vì:
     - Drop → mất quá nhiều data (~25% ở một số cột)
     - Fill bằng mode → thiên vị class đa số (no)
     - Giữ nguyên → DT tự học pattern "thiếu thông tin"

## Cell 8 [Markdown] — Tiêu đề EDA

## Cell 9 [Code] — Biểu đồ phân phối Target (Class Imbalance)

**Vẽ 2 biểu đồ:**
- **Bar chart (trái)**: Số lượng yes vs no → Thấy rõ sự chênh lệch
- **Pie chart (phải)**: Tỷ lệ % → 88.7% no / 11.3% yes

**Kết luận**: Class imbalance nghiêm trọng (tỷ lệ 8:1). Accuracy đơn thuần sẽ không đủ để đánh giá mô hình.

**Output**: `report/01_target_distribution.png`

## Cell 10 [Markdown] — Tiêu đề Categorical Analysis

## Cell 11 [Code] — Phân tích biến Categorical

```python
ct = pd.crosstab(df[col], df['y'], normalize='index') * 100
```

**Giải thích:**
- Vẽ 10 biểu đồ bar ngang (stacked) cho 10 biến categorical
- Mỗi biểu đồ cho thấy tỷ lệ yes/no trong từng giá trị của biến đó
- `normalize='index'`: Tính tỷ lệ % theo hàng (không phải số tuyệt đối)

**Phát hiện quan trọng:**
- `poutcome=success` (kết quả chiến dịch trước = thành công): yes-rate ~65% → Feature mạnh nhất
- `contact=cellular`: yes-rate cao hơn `telephone`
- `month`: Tháng 3, 9, 10 có yes-rate cao bất thường
- `job=student/retired`: yes-rate cao hơn các nghề khác

**Output**: `report/01_categorical_analysis.png`

## Cell 12 [Markdown] — Tiêu đề Numerical Distribution

## Cell 13 [Code] — Phân tích biến Numerical

```python
df[df['y'] == 'no'][col].hist(...)   # Histogram cho class "no"
df[df['y'] == 'yes'][col].hist(...)  # Histogram cho class "yes"
```

**Giải thích:**
- Vẽ 10 histogram overlay (chồng lên nhau) cho 10 biến số
- Đỏ = nhóm "no", Xanh = nhóm "yes"
- `density=True`: Chuẩn hóa để so sánh hình dạng phân phối (vì yes ít hơn no rất nhiều)

**Phát hiện:**
- `duration`: Phân biệt rõ nhất — nhóm yes có duration dài hơn nhiều
- `nr.employed`, `euribor3m`: Có sự khác biệt nhẹ giữa 2 nhóm
- `age`, `campaign`: Gần như chồng chéo → Phân biệt yếu

**Output**: `report/01_numerical_distribution.png`

## Cell 14 [Markdown] — Tiêu đề Correlation Heatmap

## Cell 15 [Code] — Ma trận tương quan

```python
corr = df[num_features].corr()
sns.heatmap(corr, mask=mask, annot=True, ...)
```

**Giải thích:**
- Tính hệ số tương quan Pearson giữa mọi cặp biến số
- `mask = np.triu(...)`: Chỉ hiển thị nửa dưới (vì ma trận đối xứng)
- Số trên mỗi ô = hệ số tương quan (-1 đến +1)

**Phát hiện:**
- `euribor3m` ↔ `nr.employed`: r ≈ 0.95 → Multicollinearity rất cao
- DT không bị ảnh hưởng bởi multicollinearity (chỉ chọn 1 feature/split)
- Nhưng cần lưu ý khi diễn giải feature importance

**Output**: `report/01_correlation_heatmap.png`

## Cell 16 [Markdown] — Tiêu đề Boxplots

## Cell 17 [Code] — Boxplot theo Target

```python
df.boxplot(column=col, by='y', ax=...)
```

**Giải thích:**
- Vẽ boxplot cho 6 biến quan trọng, tách theo nhóm yes/no
- Box = khoảng Q1-Q3 (50% data), line giữa = median, chấm = outlier

**Phát hiện:**
- `duration`: Median nhóm yes cao hơn rõ rệt → Cuộc gọi dài = khả năng đăng ký cao
- `campaign`: Nhiều outlier → Có khách bị gọi >30 lần

**Output**: `report/01_boxplots.png`

## Cell 18 [Markdown] — Tiêu đề Duration Analysis

Ghi chú quan trọng: Duration chỉ biết SAU KHI cuộc gọi kết thúc → Không dùng được cho dự đoán TRƯỚC chiến dịch.

## Cell 19 [Code] — Phân tích Duration chi tiết

**Vẽ 3 subplot:**
1. **Distribution** (trái): Histogram overlay no vs yes
2. **Mean** (giữa): Bar chart mean duration: 553s (yes) vs 220s (no)
3. **Yes-rate theo khoảng duration** (phải): Duration >600s → yes-rate >50%

**Kết luận**: Duration là feature mạnh nhất nhưng KHÔNG dùng được trong thực tế → Cần tách Scenario A/B ở NB02.

**Output**: `report/01_duration_analysis.png`

## Cell 20 [Markdown] — Tổng kết NB01

Bảng tóm tắt:
- Raw: 41,188 → Cleaned: 41,176 (xóa 12 duplicate)
- Null: 0, Unknown: giữ nguyên
- Class imbalance: 89% no / 11% yes
- Feature mạnh nhất: `duration` (nhưng prediction-time unavailable)
