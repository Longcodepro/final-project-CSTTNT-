# 🔊 Sonar Returns Classification

> **Binary Classification: Mine (M) vs Rock (R)**  
> Môn học: Machine Learning

---

## 📌 Giới thiệu

Dự án này xây dựng mô hình học máy để phân loại tín hiệu sonar phản xạ từ **mìn dưới nước (Mine)** và **đá (Rock)** dựa trên 60 đặc trưng năng lượng tần số. Pipeline được chia thành 3 giai đoạn liên tiếp: **EDA → Train → Test**, sử dụng chung một bộ dữ liệu `sonar.csv` xuyên suốt.

---

## 📂 Cấu trúc dự án

```
project/
├── data/
│   └── sonar.csv                    # Bộ dữ liệu gốc
├── exps_/sonar/                     # Thư mục kết quả thực nghiệm
│   ├── train_raw.csv                # Tập train (chưa chuẩn hóa)
│   ├── train_minmax.csv             # Tập train (MinMax Scaled)
│   ├── train_standard.csv           # Tập train (Standard Scaled)
│   ├── test_raw.csv                 # Tập test (chưa chuẩn hóa)
│   ├── test_minmax.csv              # Tập test (MinMax Scaled)
│   ├── test_standard.csv            # Tập test (Standard Scaled)
│   ├── split_metadata.json          # Thông tin chia dữ liệu
│   ├── custom_test_template.csv     # Template nhập dữ liệu tùy chỉnh
│   ├── eda_log.xlsx                 # Log kết quả EDA
│   ├── train_log.xlsx               # Log kết quả huấn luyện
│   └── sonar__test_sonar_log.xlsx   # Log kết quả test
├── model/sonar/
│   └── sonar__best_model.pkl        # Model tốt nhất (bundle)
├── 01_eda.ipynb                     # Notebook phân tích dữ liệu
├── 02_train.ipynb                   # Notebook huấn luyện mô hình
├── 03_test.ipynb                    # Notebook đánh giá mô hình
└── README.md
```

---

## 📊 Dataset

| Thuộc tính       | Giá trị                         |
|------------------|---------------------------------|
| Tên file         | `sonar.csv`                     |
| Số mẫu           | 208                             |
| Số đặc trưng     | 60 (F01 – F60, năng lượng sonar)|
| Nhãn phân loại   | `M` (Mine), `R` (Rock)          |
| Tỉ lệ nhãn       | Mine: 111, Rock: 97 (~cân bằng) |
| Missing values   | Không có                        |
| Duplicates       | Không có                        |

---

## 🔄 Pipeline thực nghiệm

```
01_eda.ipynb  ──►  02_train.ipynb  ──►  03_test.ipynb
   (EDA)              (Train)              (Test)
```

### Bước 1 – EDA (`01_eda.ipynb`)

- Thống kê mô tả, kiểm tra missing values và duplicates
- Phân tích phân phối nhãn (class distribution)
- Trực quan hóa: Histogram, Boxplot, Violin Plot, Correlation Heatmap
- Kiểm tra phân phối chuẩn (Shapiro-Wilk test)
- Chia dữ liệu **Train/Test theo tỉ lệ 70/30** (stratified)
- Chuẩn hóa và xuất 6 file CSV (raw / MinMax / Standard cho train và test)
- Ghi kết quả vào `eda_log.xlsx`

### Bước 2 – Train (`02_train.ipynb`)

- So sánh **11 thuật toán baseline** với tham số mặc định
- Đánh giá bằng **Stratified K-Fold Cross Validation** (K=5)
- Tinh chỉnh siêu tham số (**GridSearchCV**) cho 5 model mạnh:
  - SVM, Random Forest, Gradient Boosting, Logistic Regression, kNN
- Chọn model tốt nhất theo **Composite Score**:
  ```
  Composite = 0.4 × Accuracy + 0.3 × F1 + 0.3 × ROC_AUC
  ```
- Lưu model tốt nhất vào `sonar__best_model.pkl`
- Ghi kết quả vào `train_log.xlsx`

### Bước 3 – Test (`03_test.ipynb`)

- Load `best_model.pkl` từ bước train
- Đánh giá trên tập test chính thức tương ứng với scaler của model
- Xuất báo cáo: Confusion Matrix, ROC Curve, Classification Report
- Ghi kết quả vào `test_log.xlsx` và `predictions.csv`

---

## 🤖 Các mô hình được đánh giá

| Mô hình            | Ghi chú                              |
|--------------------|--------------------------------------|
| kNN                | Baseline + GridSearch (k, metric)    |
| Naive Bayes        | Baseline                             |
| SVM                | Baseline + GridSearch (C, kernel, γ) |
| Decision Tree      | Baseline                             |
| Random Forest      | Baseline + GridSearch                |
| AdaBoost           | Baseline                             |
| Gradient Boosting  | Baseline + GridSearch                |
| Extra Trees        | Baseline                             |
| LDA                | Baseline                             |
| MLP                | Baseline (128→64 hidden layers)      |
| Logistic Reg.      | Baseline + GridSearch                |

---

## ⚙️ Cài đặt & Chạy

### Yêu cầu

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

### Thứ tự thực thi

Chạy **tuần tự** từng notebook theo thứ tự:

```bash
# 1. Phân tích và chuẩn bị dữ liệu
jupyter notebook 01_eda.ipynb

# 2. Huấn luyện và chọn model
jupyter notebook 02_train.ipynb

# 3. Đánh giá model trên tập test
jupyter notebook 03_test.ipynb
```

> ⚠️ **Lưu ý:** `03_test.ipynb` yêu cầu `02_train.ipynb` đã chạy xong và file `sonar__best_model.pkl` đã tồn tại. `02_train.ipynb` tương tự yêu cầu `01_eda.ipynb` đã xuất đủ các file CSV.

---

## 📈 Tham số thực nghiệm chính

| Tham số           | Giá trị  |
|-------------------|----------|
| `TEST_SIZE`       | 0.30     |
| `RANDOM_STATE`    | 42       |
| `N_SPLITS` (KFold)| 5        |
| `W_ACC`           | 0.40     |
| `W_F1`            | 0.30     |
| `W_AUC`           | 0.30     |

---

## 📤 Output chính

| File | Mô tả |
|------|-------|
| `sonar__best_model.pkl` | Bundle model tốt nhất (model + tên + scaler + CV metrics) |
| `eda_log.xlsx` | Thống kê EDA, thông tin split |
| `train_log.xlsx` | Kết quả baseline, tuned, lịch sử chạy |
| `sonar__test_sonar_log.xlsx` | Kết quả test chính thức |
| `sonar__test_sonar_predictions.csv` | Dự đoán chi tiết từng mẫu |
| `plot_*.png` | Các biểu đồ EDA và đánh giá model |

---

## 👥 Thành viên nhóm

| STT | Họ và tên | MSSV | Phân công |
|-----|-----------|------|-----------|
| 1 | Nguyễn Thành Long | 3124410195 | Lên kế hoạch, quản lý tiến độ và phân chia công việc; Tổng hợp, kiểm tra và chỉnh sửa nội dung cuối; Quản lý source code chung trên GitHub/GitLab |
| 2 | Trần Đăng Khoa | 3124410162 | Quản lý và phân chia nội dung báo cáo; Tổng hợp báo cáo và PPT từ các thành viên; Viết nội dung phần giới thiệu (Báo cáo & PPT) |
| 3 | Hàng Vinh Quang | 3125720024 | Báo cáo phần II; PPT tương ứng; Code tương ứng |
| 4 | Vũ Văn Toàn | 3124410363 | Báo cáo phần III; PPT tương ứng; Code tương ứng |
| 5 | Lê Quốc Việt | 3124410405 | Báo cáo phần IV; PPT tương ứng; Code tương ứng |

---

## 📝 Ghi chú

- Tất cả các bước dùng `random_state=42` để đảm bảo tính tái lập (reproducibility).
- Scaler được fit **chỉ trên tập train** rồi mới transform tập test, tránh data leakage.
- Model cuối được **fit lại trên toàn bộ tập train** trước khi export.
- Định dạng bundle `.pkl` gồm: `model`, `model_name`, `scaler_name`, `cv_metrics`.

