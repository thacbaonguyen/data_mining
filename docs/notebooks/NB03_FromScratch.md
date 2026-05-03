# NB03: Data Mining — Decision Tree From Scratch

> **Bước KDD**: Data Mining (Phần 1)
> **File**: `notebooks/03_from_scratch.ipynb` — 19 cells
> **Mục đích**: Train Decision Tree tự code từ đầu, đánh giá trên cả 2 Scenario

---

## Cell 0 [Markdown] — Tiêu đề notebook

## Cell 1 [Code] — Setup môi trường

```python
from src.tree import DecisionTreeClassifier as ScratchDT
from src.metrics import accuracy, precision, recall, f1_score
```

**Giải thích:**
- Import class `DecisionTreeClassifier` từ file `src/tree.py` (do chúng ta tự code)
- Import các hàm metric từ `src/metrics.py` (cũng tự code)
- Đây là điểm khác biệt cốt lõi: **KHÔNG dùng thư viện sklearn** cho mô hình

## Cell 2 [Markdown] — Tiêu đề Load Data

## Cell 3 [Code] — Load dữ liệu đã xử lý

```python
with open("data/processed_data.pkl", "rb") as f:
    data = pickle.load(f)
```

**Giải thích:**
- Load file pickle đã tạo ở NB02
- `data["scenario_A"]` và `data["scenario_B"]` chứa sẵn X_train, X_test, y_train, y_test

## Cell 4 [Markdown] — Giải thích thuật toán Decision Tree

Cell markdown giải thích lý thuyết:
- **Hunt's Algorithm**: Xây cây từ trên xuống, đệ quy
- **Gini Index**: Thước đo độ hỗn loạn (impurity) tại mỗi nút
- **Stopping Criteria**: max_depth, min_samples_split, min_samples_leaf
- **Prediction**: Duyệt cây từ gốc đến lá, trả về class đa số tại lá

## Cell 5 [Markdown] — Tiêu đề Train Scenario A

## Cell 6 [Code] — Train DT Scratch trên Scenario A

```python
d_A = data["scenario_A"]
dt_A = ScratchDT(criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=1)
dt_A.fit(d_A["X_train"], d_A["y_train"], feature_names=d_A["feature_names"])
y_pred_A = dt_A.predict(d_A["X_test"])
```

**Giải thích từng tham số:**
- `criterion="gini"`: Dùng chỉ số Gini để đánh giá chất lượng mỗi lần chia cắt
- `max_depth=5`: Cây tối đa 5 tầng (Pre-pruning — Tỉa trước)
- `min_samples_split=2`: Nút phải có ≥ 2 mẫu mới được chia tiếp
- `min_samples_leaf=1`: Nút lá phải có ≥ 1 mẫu
- `feature_names`: Truyền tên cột để cây in ra rule dễ đọc

**Quy trình bên trong khi gọi `fit()`:**
1. Gọi `_build_tree(X, y, depth=0)`
2. Kiểm tra stopping criteria (tất cả cùng class? depth >= 5? ít mẫu?)
3. Gọi `find_best_split()` từ `src/splitter.py` → Tìm feature + threshold tốt nhất
4. Chia data thành 2 nhánh → Đệ quy cho mỗi nhánh
5. Kết quả: Một cấu trúc cây gồm nhiều object `Node` lồng nhau

**In classification report**: Accuracy, Precision, Recall, F1 cho class "yes"

## Cell 7 [Markdown] — Tiêu đề Visualize Tree A

## Cell 8 [Code] — In cấu trúc cây (Text)

```python
print("Decision Tree Structure (Scenario A — max_depth=5):")
dt_A.print_tree()
```

**Giải thích:**
- Hàm `print_tree()` in ra cấu trúc cây dạng text (giống cây thư mục)
- Mỗi dòng hiển thị: điều kiện chia cắt, số mẫu, phân phối class, dự đoán
- Ví dụ output: `|--- duration <= 303.50 (samples=32940, yes=17.2%)`
- Cây Scratch không vẽ được đồ họa như sklearn (do dùng class Node tự định nghĩa)

## Cell 9 [Markdown] — Tiêu đề Feature Importance A

## Cell 10 [Code] — Feature Importance Scenario A

```python
importance_A = dt_A.feature_importances_
idx = np.argsort(importance_A)[::-1][:15]
```

**Giải thích:**
- `feature_importances_`: Tính bằng tổng weighted impurity decrease tại mỗi split mà feature đó tham gia
- Feature nào được dùng ở nhiều split quan trọng (gần gốc cây) → Importance cao
- Vẽ bar chart ngang Top 15 features

**Kết quả Scenario A:**
- `duration` = 50.9% → Áp đảo tuyệt đối
- `nr.employed` = 35.3%
- Các features khác bị "che lấp" bởi duration

**Output**: `report/03_feature_importance_A.png`

## Cell 11 [Markdown] — Tiêu đề Train Scenario B

## Cell 12 [Code] — Train DT Scratch trên Scenario B

```python
d_B = data["scenario_B"]
dt_B = ScratchDT(criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=1)
dt_B.fit(d_B["X_train"], d_B["y_train"], feature_names=d_B["feature_names"])
y_pred_B = dt_B.predict(d_B["X_test"])
```

**Giải thích:**
- Hoàn toàn giống Scenario A, chỉ khác dữ liệu đầu vào (đã loại bỏ cột duration)
- In classification report cho Scenario B

**Kết quả:**
- F1 giảm từ 0.57 (A) xuống 0.37 (B) → Chứng minh tác động của duration

## Cell 13 [Markdown] — Tiêu đề Feature Importance B

## Cell 14 [Code] — Feature Importance Scenario B

**Kết quả Scenario B (không duration):**
- `nr.employed` = 64.2% → Trở thành feature quan trọng nhất
- `pdays` = 13.6%
- `euribor3m` = 7.8%
- Các chỉ số kinh tế vĩ mô và lịch sử tiếp thị nổi lên → Đây mới là features mà ngân hàng BIẾT TRƯỚC cuộc gọi

**Output**: `report/03_feature_importance_B.png`

## Cell 15 [Markdown] — Tiêu đề So sánh A vs B

## Cell 16 [Code] — Bảng so sánh Scenario A vs B

- In bảng so sánh 4 metric (Acc, Prec, Rec, F1) giữa Scenario A và B
- Tính delta (chênh lệch) để thấy rõ tác động của việc loại bỏ duration

## Cell 17 [Markdown] — Tiêu đề Lưu kết quả

## Cell 18 [Code] — Lưu results_scratch.pkl

```python
results_scratch = {
    "scenario_A": {
        "y_pred": y_pred_A,
        "report": report_A,
        "feature_importance": importance_A,
        "feature_names": d_A["feature_names"],
    },
    "scenario_B": { ... }
}
with open("data/results_scratch.pkl", "wb") as f:
    pickle.dump(results_scratch, f)
```

**Giải thích:**
- Đóng gói toàn bộ kết quả dự đoán và feature importance
- File này sẽ được NB05 (Evaluation) và NB06 (Knowledge) load lên để so sánh với sklearn

**Output**: `data/results_scratch.pkl`
