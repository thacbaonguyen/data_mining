# NB04: Data Mining — Sklearn + Multi-Algorithm

> **Bước KDD**: Data Mining (Phần 2)
> **File**: `notebooks/04_sklearn.ipynb` — 24 cells
> **Mục đích**: Train sklearn DT (đối chứng), Naive Bayes, Random Forest; so sánh đa thuật toán

---

## Cell 0 [Markdown] — Tiêu đề notebook

## Cell 1 [Code] — Setup môi trường

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
```

**Giải thích:**
- `DecisionTreeClassifier`: Thuật toán cây quyết định của sklearn
- `plot_tree`: Hàm vẽ đồ họa cấu trúc cây
- `GaussianNB`: Thuật toán Naive Bayes (phân phối Gaussian cho biến liên tục)
- `RandomForestClassifier`: Rừng ngẫu nhiên (ensemble gồm nhiều cây)

## Cell 2 [Markdown] — Tiêu đề Load Data

## Cell 3 [Code] — Load dữ liệu

```python
with open("data/processed_data.pkl", "rb") as f:
    data = pickle.load(f)
```

Dùng chung file `processed_data.pkl` với NB03 → Đảm bảo so sánh công bằng tuyệt đối.

## Cell 4 [Markdown] — Tiêu đề Train Scenario A

## Cell 5 [Code] — Train sklearn DT trên Scenario A

```python
sk_A = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
sk_A.fit(d_A["X_train"], d_A["y_train"])
y_sk_A = sk_A.predict(d_A["X_test"])
```

**Giải thích:**
- **Cùng `criterion="gini"` và `max_depth=5`** như mô hình Scratch → So sánh trong cùng điều kiện
- `random_state=42`: Cố định kết quả ngẫu nhiên
- In classification report để so sánh với NB03

## Cell 6 [Markdown] — Tiêu đề Visualize Tree A

## Cell 7 [Code] — Vẽ đồ họa cây Scenario A

```python
fig, ax = plt.subplots(figsize=(40, 18))
plot_tree(sk_A, feature_names=d_A["feature_names"],
          class_names=["no", "yes"], filled=True, rounded=True,
          fontsize=7, ax=ax)
```

**Giải thích:**
- `plot_tree()`: Hàm riêng của sklearn để vẽ cấu trúc cây dạng đồ họa
- `filled=True`: Tô màu nút theo class chiếm ưu thế (xanh = yes, cam = no)
- `rounded=True`: Bo tròn góc nút cho đẹp
- `fontsize=7`: Cỡ chữ nhỏ vì cây depth=5 có tới 32 lá
- `figsize=(40, 18)`: Kích thước lớn để các nút không bị đè lên nhau

**Output**: `report/04_sklearn_tree_A.png`

## Cell 8 [Markdown] — Tiêu đề Feature Importance A

## Cell 9 [Code] — Feature Importance sklearn Scenario A

```python
imp_sk_A = sk_A.feature_importances_
```

- Vẽ bar chart ngang Top 15 features
- Kết quả gần như giống hệt NB03 (Scratch)

**Output**: `report/04_feature_importance_sk_A.png`

## Cell 10 [Markdown] — Tiêu đề Train Scenario B

## Cell 11 [Code] — Train sklearn DT trên Scenario B

Tương tự Cell 5 nhưng dùng dữ liệu Scenario B (không duration).

## Cell 12 [Markdown] — Tiêu đề Visualize Tree B

## Cell 13 [Code] — Vẽ đồ họa cây Scenario B

Tương tự Cell 7 nhưng cho Scenario B. Root split sẽ khác (không phải duration nữa).

**Output**: `report/04_sklearn_tree_B.png`

## Cell 14 [Markdown] — Tiêu đề Feature Importance B

## Cell 15 [Code] — Feature Importance sklearn Scenario B

**Output**: `report/04_feature_importance_sk_B.png`

## Cell 16 [Markdown] — Tiêu đề Naive Bayes

> **Naive Bayes** thuộc trường phái xác suất (Probabilistic). Thay vì xây cây luật, nó tính xác suất P(yes|X) dựa trên định lý Bayes. Đây là baseline mạnh cho bài toán phân loại.

## Cell 17 [Code] — Train Naive Bayes trên cả 2 Scenario

```python
nb_A = GaussianNB()
nb_A.fit(d_A["X_train"], d_A["y_train"])
y_nb_A = nb_A.predict(d_A["X_test"])
```

**Giải thích:**
- `GaussianNB()`: Giả định mỗi feature tuân theo phân phối chuẩn (Gaussian/Normal)
- Không cần hyperparameter tuning (không có max_depth, min_samples...)
- Cực kỳ nhanh vì chỉ cần tính mean và variance cho mỗi feature/class
- Train và in classification report cho cả Scenario A và B

**Tại sao dùng GaussianNB mà không CategoricalNB?**
- Dữ liệu đã qua One-Hot Encoding → Tất cả features đều là dạng số (0/1 hoặc float)
- GaussianNB xử lý được cả 2 loại này

## Cell 18 [Markdown] — Tiêu đề Random Forest

> **Random Forest** là phiên bản nâng cấp của Decision Tree: Xây dựng nhiều cây (n=100) trên các tập con ngẫu nhiên, sau đó bỏ phiếu đa số (majority vote).

## Cell 19 [Code] — Train Random Forest trên cả 2 Scenario

```python
rf_A = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_A.fit(d_A["X_train"], d_A["y_train"])
y_rf_A = rf_A.predict(d_A["X_test"])
```

**Giải thích từng tham số:**
- `n_estimators=100`: Xây 100 cây quyết định khác nhau
- `max_depth=5`: Mỗi cây tối đa 5 tầng (giống DT đơn lẻ để so sánh công bằng)
- `random_state=42`: Cố định kết quả
- `n_jobs=-1`: Dùng tất cả CPU cores để train song song (nhanh hơn)

**Cách Random Forest hoạt động:**
1. Lấy ngẫu nhiên 1 tập con dữ liệu (có hoàn lại) → Gọi là Bootstrap
2. Lấy ngẫu nhiên 1 tập con features (không phải tất cả features)
3. Xây 1 cây trên tập con đó
4. Lặp lại 100 lần → Có 100 cây khác nhau
5. Dự đoán: Cho 100 cây bỏ phiếu, kết quả cuối = class được bầu nhiều nhất

## Cell 20 [Markdown] — Tiêu đề Feature Importance DT vs RF

## Cell 21 [Code] — So sánh Feature Importance: DT (1 cây) vs RF (100 cây)

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Trái: DT importance — Phải: RF importance
```

**Giải thích:**
- Vẽ 2 bar chart cạnh nhau để so sánh trên Scenario B
- **DT (1 cây)**: Tập trung vào ít features → `nr.employed` chiếm ưu thế tuyệt đối
- **RF (100 cây)**: Importance phân tán đều hơn → Ổn định, tổng quát hơn
- Lý do: Mỗi cây trong RF chỉ nhìn subset features → Nhiều features được "đánh giá công bằng"

**Output**: `report/04_importance_dt_vs_rf.png`

## Cell 22 [Markdown] — Tiêu đề Lưu kết quả

## Cell 23 [Code] — Lưu results_sklearn.pkl (cả 3 thuật toán)

```python
results_sklearn = {
    "scenario_A": { ... },         # DT sklearn kết quả Scenario A
    "scenario_B": { ... },         # DT sklearn kết quả Scenario B
    "naive_bayes": {               # Kết quả Naive Bayes
        "scenario_A": {"y_pred": y_nb_A, "report": nb_report_A},
        "scenario_B": {"y_pred": y_nb_B, "report": nb_report_B},
    },
    "random_forest": {             # Kết quả Random Forest
        "scenario_A": {"y_pred": y_rf_A, "report": rf_report_A,
                       "feature_importance": rf_A.feature_importances_},
        "scenario_B": {"y_pred": y_rf_B, "report": rf_report_B,
                       "feature_importance": rf_B.feature_importances_},
    }
}
```

**Giải thích:**
- Lưu kết quả dự đoán của cả 3 thuật toán vào cùng 1 file
- NB05 sẽ load file này để vẽ biểu đồ so sánh đa thuật toán

**Output**: `data/results_sklearn.pkl`
