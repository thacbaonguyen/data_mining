# GIẢI THÍCH CHI TIẾT TỪNG BIỂU ĐỒ — THEO THỨ TỰ PIPELINE

> Tổng cộng **23 biểu đồ** trong thư mục `report/`, sắp xếp theo thứ tự pipeline KDD.

---

## GIAI ĐOẠN 1: EDA & PREPROCESSING (NB01)

### 1. `01_target_distribution.png`

**Loại biểu đồ:** Bar chart (trái) + Pie chart (phải)

**Dữ liệu:** Cột `y` (biến mục tiêu) của toàn bộ 41,176 dòng sau khi xóa duplicate.

**Cách đọc:**
- Bar chart bên trái: Trục X là 2 giá trị "no" và "yes", trục Y là số lượng khách hàng. Con số ghi trên đầu mỗi cột cho biết chính xác số lượng.
- Pie chart bên phải: Hiển thị tỷ lệ phần trăm. Màu đỏ = "no", Màu xanh = "yes".

**Kết quả cụ thể:**
- "no" = 36,548 khách (88.7%)
- "yes" = 4,640 khách (11.3%)
- Tỷ lệ chênh lệch: 7.9:1 (gần 8 người từ chối mới có 1 người đồng ý)

**Tại sao biểu đồ này quan trọng:**
Nó cho thấy bài toán bị **Class Imbalance** (mất cân bằng lớp) nghiêm trọng. Hệ quả trực tiếp là:
- Nếu một mô hình "ngu" chỉ đoán tất cả là "no", nó vẫn đạt Accuracy 88.7%.
- Do đó, Accuracy KHÔNG PHẢI là thước đo đáng tin cậy cho bài toán này.
- Pipeline phải dùng các metric khác: **Precision, Recall, F1 Score** cho class "yes".
- Mọi bước chia dữ liệu (split) phải dùng **Stratified** để giữ tỷ lệ 89/11 ở cả Train và Test.

---

### 2. `01_categorical_analysis.png`

**Loại biểu đồ:** 10 biểu đồ bar ngang xếp chồng (stacked horizontal bar), mỗi biểu đồ cho 1 biến categorical.

**Dữ liệu:** 10 biến phân loại: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome.

**Cách đọc:**
- Mỗi thanh ngang đại diện cho 1 giá trị của biến (ví dụ: job = "student", job = "retired"...)
- Phần đỏ = tỷ lệ % "no", phần xanh = tỷ lệ % "yes"
- Thanh nào có phần xanh DÀI HƠN = giá trị đó có yes-rate cao hơn
- Tổng mỗi thanh luôn = 100%

**Phát hiện quan trọng (phải nhớ khi thuyết trình):**
1. **`poutcome = success`** (kết quả chiến dịch trước = thành công): yes-rate ~65%. Đây là feature mạnh nhất — khách đã từng đăng ký thì có khả năng đăng ký lại rất cao.
2. **`contact = cellular`**: yes-rate cao hơn `telephone`. Gọi bằng di động hiệu quả hơn điện thoại bàn.
3. **`month = mar, sep, oct, dec`**: Yes-rate cao bất thường. Lý do: Những tháng này ngân hàng gọi ÍT cuộc hơn → Chọn lọc kỹ hơn → yes-rate cao (selection bias, KHÔNG phải nhân quả).
4. **`job = student, retired`**: Yes-rate cao hơn các nghề khác. Sinh viên (trẻ, tò mò) và người hưu trí (có thời gian, thu nhập ổn định) dễ chấp nhận tư vấn.

---

### 3. `01_numerical_distribution.png`

**Loại biểu đồ:** 10 histogram overlay (chồng lên nhau), mỗi biểu đồ cho 1 biến số.

**Dữ liệu:** 10 biến numerical: age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed.

**Cách đọc:**
- Trục X = giá trị của biến, Trục Y = mật độ (density) — đã chuẩn hóa để so sánh hình dạng.
- Màu đỏ (mờ) = phân phối của nhóm "no", Màu xanh (mờ) = phân phối của nhóm "yes".
- **Vùng 2 màu TÁCH BIỆT rõ ràng** = Feature có khả năng phân biệt tốt giữa yes và no.
- **Vùng 2 màu CHỒNG CHÉO nhiều** = Feature phân biệt yếu.

**Phát hiện:**
1. **`duration`**: Tách biệt rõ nhất. Nhóm "yes" có đuôi dài về phía phải (cuộc gọi dài). Nhóm "no" tập trung ở giá trị thấp.
2. **`nr.employed`** và **`euribor3m`**: Có sự dịch chuyển nhẹ giữa 2 nhóm. Khi số người được tuyển dụng thấp hoặc lãi suất Euribor thấp → yes-rate cao hơn.
3. **`age`**, **`campaign`**: Gần như chồng chéo hoàn toàn → Phân biệt rất yếu khi dùng đơn lẻ.
4. **`pdays`**: Phần lớn giá trị = 999 (nghĩa là chưa từng liên hệ trước đó). Nhóm "yes" có nhiều giá trị nhỏ hơn 999 → Khách đã từng được liên hệ gần đây có xu hướng đăng ký.

---

### 4. `01_correlation_heatmap.png`

**Loại biểu đồ:** Ma trận nhiệt (Heatmap) tam giác dưới.

**Dữ liệu:** Hệ số tương quan Pearson giữa 10 biến numerical.

**Cách đọc:**
- Mỗi ô = hệ số tương quan giữa 2 biến. Giá trị từ -1 đến +1.
- Đỏ đậm = tương quan dương mạnh (2 biến tăng/giảm cùng chiều).
- Xanh đậm = tương quan âm mạnh (2 biến ngược chiều).
- Trắng = không tương quan.
- Chỉ hiển thị nửa dưới vì ma trận đối xứng.

**Phát hiện:**
1. **`euribor3m` ↔ `nr.employed`**: r = 0.95 → **Multicollinearity cực cao**. Hai biến này mang gần như cùng một thông tin. Nếu dùng Logistic Regression sẽ có vấn đề, nhưng Decision Tree KHÔNG bị ảnh hưởng vì mỗi nút chỉ chọn 1 feature.
2. **`emp.var.rate` ↔ `euribor3m`**: r = 0.97 → Cũng multicollinear. Các chỉ số kinh tế vĩ mô thường di chuyển cùng nhau.
3. **Hệ quả cho Feature Importance:** Khi 2 biến tương quan cao, importance có thể bị "chia đôi" giữa chúng. Cần lưu ý khi diễn giải.

---

### 5. `01_boxplots.png`

**Loại biểu đồ:** 6 boxplot, mỗi biểu đồ tách theo nhóm yes/no.

**Dữ liệu:** 6 biến chọn lọc: age, duration, campaign, pdays, previous, euribor3m.

**Cách đọc:**
- **Hộp (Box)**: Cạnh dưới = Q1 (phân vị 25%), cạnh trên = Q3 (phân vị 75%). 50% dữ liệu nằm trong hộp.
- **Đường ngang giữa hộp**: Median (trung vị — giá trị ở giữa khi sắp xếp).
- **Râu (Whiskers)**: Kéo dài đến giá trị xa nhất trong khoảng 1.5×IQR (IQR = Q3 - Q1).
- **Chấm tròn ngoài râu**: Outlier (giá trị ngoại lai, bất thường).

**Phát hiện:**
1. **`duration`**: Median nhóm yes (~400s) cao hơn rõ rệt so với no (~180s). Hộp yes rộng hơn và dịch lên cao.
2. **`campaign`**: Có rất nhiều outlier (chấm tròn) ở cả 2 nhóm. Có khách bị gọi >30 lần mà vẫn từ chối.
3. **`euribor3m`**: Hộp yes nằm thấp hơn hộp no → Lãi suất Euribor thấp → yes-rate cao hơn.

---

### 6. `01_duration_analysis.png`

**Loại biểu đồ:** 3 subplot nằm ngang.

**Dữ liệu:** Biến `duration` (thời lượng cuộc gọi, đơn vị giây).

**Subplot 1 (trái) — Distribution:**
- Histogram overlay: đỏ = no, xanh = yes.
- Nhóm "no" tập trung mạnh ở 0-200 giây (cuộc gọi ngắn, bị từ chối nhanh).
- Nhóm "yes" trải dài từ 200 đến 1000+ giây.

**Subplot 2 (giữa) — Mean:**
- Bar chart so sánh mean duration: yes = **553 giây** (~9 phút), no = **220 giây** (~3.5 phút).
- Chênh lệch: 2.5 lần.

**Subplot 3 (phải) — Yes-rate theo khoảng duration:**
- Chia duration thành các khoảng (0-100, 100-200, ..., 600+).
- Duration > 600s → yes-rate vượt **50%**.
- Duration < 100s → yes-rate gần **0%**.

**Ý nghĩa cốt lõi:**
Duration là feature mạnh nhất nhưng **KHÔNG THỂ dùng cho dự đoán thực tế** vì chỉ biết SAU KHI cuộc gọi kết thúc. Đây là lý do pipeline tách thành Scenario A (có duration, benchmark) và Scenario B (không duration, thực tế).

---

## GIAI ĐOẠN 2: TRANSFORMATION (NB02)

### 7. `02_ablation_study.png`

**Loại biểu đồ:** Grouped bar chart — 3 nhóm × 4 metric.

**Dữ liệu:** Kết quả train cùng 1 mô hình sklearn DT (depth=5) trên 3 phiên bản data khác nhau.

**Cách đọc:**
- Trục X: 4 metric (Accuracy, Precision, Recall, F1).
- 3 nhóm bar: Xanh = Scenario A, Cam = Scenario B, Tím = B+ (Undersampling).
- Thanh cao hơn = metric tốt hơn.

**Kết quả cụ thể:**

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| A (có duration) | 0.9144 | 0.6534 | 0.5119 | **0.5740** |
| B (không duration) | 0.9009 | 0.6522 | 0.2586 | **0.3704** |
| B+ (undersample) | 0.8719 | 0.4480 | 0.5894 | **0.5091** |

**Phân tích (phải nói được khi thuyết trình):**
1. **A → B (bỏ duration):** F1 giảm 0.20. Recall giảm mạnh (0.51→0.26). Chứng minh duration là feature mạnh nhất.
2. **B → B+ (thêm undersampling):** Recall tăng mạnh (0.26→0.59) nhưng Precision giảm (0.65→0.45). Đây là trade-off: Tìm được nhiều khách tiềm năng hơn nhưng cũng "báo động nhầm" nhiều hơn.
3. **Kết luận Data-Centric:** Cùng 1 thuật toán, chỉ thay đổi DỮ LIỆU đầu vào đã khiến hiệu suất thay đổi 50%. Chứng minh: Chất lượng dữ liệu quan trọng hơn việc chọn thuật toán.

---

## GIAI ĐOẠN 3: DATA MINING (NB03 + NB04)

### 8-9. `03_feature_importance_A.png` & `03_feature_importance_B.png`

**Loại biểu đồ:** Bar chart ngang — Top 15 features quan trọng nhất (mô hình Scratch).

**Cách đọc:**
- Trục Y: Tên feature. Trục X: Importance score (tổng weighted impurity decrease).
- Thanh dài hơn = Feature đóng góp nhiều hơn vào các quyết định chia cắt của cây.

**Scenario A (có duration):**
- `duration` = **50.9%** → Áp đảo tuyệt đối, một mình chiếm hơn nửa tổng importance.
- `nr.employed` = 35.3%
- Tất cả features còn lại chia nhau phần nhỏ còn lại.

**Scenario B (không duration):**
- `nr.employed` = **64.2%** → Trở thành feature quan trọng nhất.
- `pdays` = 13.6% (số ngày kể từ lần liên hệ gần nhất).
- `euribor3m` = 7.8% (lãi suất liên ngân hàng châu Âu).

**Ý nghĩa:** Khi loại bỏ duration, các chỉ số kinh tế vĩ mô và lịch sử tiếp thị nổi lên. Đây mới là các features mà ngân hàng BIẾT TRƯỚC cuộc gọi → Có giá trị thực tiễn.

---

### 10-11. `04_feature_importance_sk_A.png` & `04_feature_importance_sk_B.png`

Tương tự 2 biểu đồ trên nhưng từ mô hình **sklearn**. Kết quả gần như trùng khớp với Scratch (sai khác < 0.001), chứng minh code tự viết hoạt động đúng.

---

### 12-13. `04_sklearn_tree_A.png` & `04_sklearn_tree_B.png`

**Loại biểu đồ:** Đồ họa cấu trúc cây quyết định (sklearn `plot_tree`).

**Cách đọc mỗi nút (node):**
- **Dòng 1:** Điều kiện chia cắt. Ví dụ: `nr.employed <= 5087.65`
- **Dòng 2:** `gini = 0.xxx` — Độ hỗn loạn tại nút đó. Gini càng nhỏ = nút càng thuần khiết.
- **Dòng 3:** `samples = xxx` — Số mẫu dữ liệu đi qua nút này.
- **Dòng 4:** `value = [xxx, yyy]` — Số mẫu class no (trái) và yes (phải).
- **Dòng 5:** `class = no/yes` — Class chiếm đa số tại nút.
- **Màu sắc:** Cam = đa số là "no", Xanh = đa số là "yes". Càng đậm = càng thuần khiết.

**Scenario A — Root split:** `duration <= 303.5` → Cuộc gọi ≤ 5 phút → rẽ trái (đa số "no").

**Scenario B — Root split:** `nr.employed <= 5087.65` → Số người có việc làm thấp → rẽ trái (kinh tế yếu, khách dễ chấp nhận đầu tư an toàn).

---

### 14. `04_importance_dt_vs_rf.png`

**Loại biểu đồ:** 2 bar chart ngang cạnh nhau — DT (1 cây) vs RF (100 cây), Scenario B.

**Cách đọc:** Giống feature importance ở trên. So sánh trực quan 2 thuật toán.

**Phát hiện:**
- **DT (1 cây):** Importance tập trung mạnh vào `nr.employed` (64%). Các features khác rất nhỏ.
- **RF (100 cây):** Importance phân tán đều hơn. Nhiều features đều đóng góp 5-15%.

**Tại sao RF phân tán hơn?** Mỗi cây trong RF chỉ nhìn một tập con features ngẫu nhiên. Feature `nr.employed` không phải lúc nào cũng có mặt → Các features khác có cơ hội "tỏa sáng" → Importance phân bố đều hơn → Mô hình tổng quát và ổn định hơn.

---

## GIAI ĐOẠN 4: EVALUATION (NB05)

### 15. `05_confusion_matrices.png`

**Loại biểu đồ:** 4 heatmap (2×2 grid) — confusion matrix.

**Bố cục:** Hàng trên = Scenario A, Hàng dưới = Scenario B. Cột trái = Scratch, Cột phải = Sklearn.

**Cách đọc mỗi ma trận 2×2:**

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | TN (True Negative) ✅ | FP (False Positive) ❌ |
| **Actual Yes** | FN (False Negative) ❌ | TP (True Positive) ✅ |

- **TN (góc trên-trái):** Đoán "no" đúng. Số này rất lớn vì 89% data là "no".
- **TP (góc dưới-phải):** Đoán "yes" đúng. Con số nhỏ nhưng là giá trị kinh doanh thực sự.
- **FP (góc trên-phải):** Đoán "yes" nhưng thực tế là "no" → Gọi điện lãng phí.
- **FN (góc dưới-trái):** Đoán "no" nhưng thực tế là "yes" → **Bỏ sót khách tiềm năng** (nguy hiểm nhất).

**Phát hiện:**
- Scratch ≈ Sklearn (số liệu gần trùng khớp).
- Scenario B: FN rất lớn → Recall thấp → Mô hình "thận trọng", ít dám predict "yes".

---

### 16. `05_depth_analysis.png`

**Loại biểu đồ:** 2 subplot — Trái: Metrics vs depth. Phải: Số lá vs depth.

**Subplot trái:** 4 đường (Acc, Prec, Rec, F1) thay đổi theo max_depth = [3, 5, 7, 10, None].
**Subplot phải:** Số lá tăng theo depth. depth=5 → 32 lá. depth=None → 4973 lá.

**Kết luận:** depth=5 là điểm cân bằng tốt nhất. Tăng depth thêm không cải thiện nhiều F1 nhưng cây phức tạp hơn rất nhiều (khó diễn giải, dễ overfit).

---

### 17. `05_pr_curve.png`

**Loại biểu đồ:** 2 Precision-Recall Curve — Scenario A (trái) và B (phải).

**Cách đọc:**
- Trục X = Recall. Trục Y = Precision.
- Đường cong xanh = Hiệu suất mô hình ở các ngưỡng xác suất khác nhau.
- Vùng tô mờ bên dưới = Diện tích AP (Average Precision). AP càng lớn = mô hình càng tốt.
- Đường ngang xám = Baseline (đoán ngẫu nhiên = tỷ lệ yes trong data ≈ 11.3%).

**Ý nghĩa:** PR Curve phù hợp hơn ROC Curve cho bài toán mất cân bằng vì nó tập trung vào class thiểu số ("yes").

---

### 18. `05_cv_boxplot.png`

**Loại biểu đồ:** 4 boxplot — F1 Score qua 5 folds.

**4 nhóm:** Scratch A, Sklearn A, Scratch B, Sklearn B.

**Cách đọc:** Hộp hẹp + giá trị mean±std nhỏ = Mô hình ổn định, không phụ thuộc cách chia data.

**Kết luận:** Scratch ≈ Sklearn qua mọi fold. Std rất thấp → Kết quả đáng tin cậy, reproducible.

---

### 19. `05_learning_curve.png`

**Loại biểu đồ:** 2 subplot — Scenario A (trái), B (phải). Mỗi subplot có 2 đường: Train F1 và Validation F1.

**Cách đọc:**
- Trục X = Số lượng mẫu training (tăng dần từ 10% đến 100% tổng data).
- Đường xanh = Train F1. Đường cam = Validation F1.
- Vùng tô mờ = ±1 standard deviation.
- **2 đường hội tụ** → Dữ liệu đã đủ, model không overfit.
- **2 đường còn cách xa** → Cần thêm data hoặc model quá phức tạp (overfit).

**Kết luận:** Khoảng cách Train-Validation hẹp → 41,000 dòng đã đủ cho depth=5.

---

### 20. `05_multi_model_f1.png`

**Loại biểu đồ:** Grouped bar chart — 4 thuật toán × 2 scenarios.

**Cách đọc:**
- 4 nhóm trên trục X: DT Scratch, DT Sklearn, Naive Bayes, Random Forest.
- Mỗi nhóm có 2 bar: Xanh = Scenario A, Cam = Scenario B.
- Số ghi trên mỗi bar = F1 Score chính xác.

**Ý nghĩa:** Biểu đồ này trả lời câu hỏi "Thuật toán nào tốt nhất?"
- Scenario A: Mọi thuật toán đều đạt F1 tốt nhờ duration.
- Scenario B: RF thường ổn định nhất. DT dễ diễn giải nhất. NB nhanh nhất nhưng có thể bị hạn chế bởi giả định independence.

---

### 21. `05_cm_extra_models.png`

**Loại biểu đồ:** 2 confusion matrix — Naive Bayes (trái) và Random Forest (phải) trên Scenario B.

**Cách đọc:** Giống biểu đồ #15. So sánh phân bố TP/FP/FN/TN giữa NB và RF với DT.

---

## GIAI ĐOẠN 5: KNOWLEDGE DISCOVERY (NB06)

### 22. `06_importance_comparison.png`

**Loại biểu đồ:** 2 bar chart ngang cạnh nhau — Scenario A (trái) vs B (phải).

**Mục đích:** Chứng minh trực quan rằng duration "che lấp" mọi features khác. Khi loại duration, bức tranh features thay đổi hoàn toàn.

**Tri thức rút ra:** Ngân hàng nên tập trung vào các features ở Scenario B (nr.employed, pdays, euribor3m, poutcome) vì đây là thông tin biết TRƯỚC cuộc gọi.

---

### 23. `06_segment_heatmap.png`

**Loại biểu đồ:** Heatmap 2 chiều — Hàng = Job, Cột = Education.

**Dữ liệu:** Yes-rate (%) cho mỗi nhóm (Job × Education). Chỉ hiển thị segments có support ≥ 1% tổng data.

**Cách đọc:**
- Màu càng đậm (đỏ/cam) = yes-rate càng cao.
- Số trong mỗi ô = tỷ lệ % khách đăng ký trong nhóm đó.

**Phát hiện quan trọng nhất:**
1. **`retired + basic.4y`**: yes-rate = **31%**, lift = 2.75× (gấp gần 3 lần trung bình 11.3%). Đây là nhóm khách tiềm năng nhất.
2. **`admin + university.degree`**: yes-rate = 14.3%.
3. **`blue-collar + basic.9y`**: yes-rate rất thấp (~5%) → Không nên ưu tiên gọi.

**Khuyến nghị kinh doanh:** Ưu tiên gọi nhóm hưu trí + học vấn basic.4y trước. Nhưng lưu ý: Đây là **pattern quan sát** (correlation), KHÔNG phải quan hệ nhân quả. Cần A/B testing để validate trước khi triển khai thật.
