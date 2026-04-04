# 🚢 Sonar Returns Classification – Final Project

**Môn học:** Trí Tuệ Nhân Tạo  
**Bài toán:** Phân loại nhị phân – Mine (M) vs Rock (R)  
**Dataset:** Sonar Returns (208 samples, 60 features)  

---

## 📌 Giới thiệu

Project này xây dựng một pipeline Machine Learning hoàn chỉnh để phân loại tín hiệu sonar thành:
- **M (Mine)** – vật thể kim loại  
- **R (Rock)** – đá  

Pipeline gồm: EDA → Preprocessing → Training → Evaluation → Testing → Tuning.

---

## 🗂️ Cấu trúc thư mục

```
final-project-CSTTNT/
├── bin/          
├── data/         
├── doc/          
├── exps_/        
├── model/        
└── prj/          
```

---

## ⚙️ Pipeline

### 1. 📊 EDA – `01_eda.ipynb`
- Thống kê dữ liệu
- Visualization
- Kiểm tra phân phối
- Chuẩn hóa (Raw, MinMax, Standard)

### 2. 🤖 Train – `02_train.ipynb`
- 10 models baseline
- 5-Fold Cross Validation
- GridSearch cho SVM & Random Forest
- Lưu model `.pkl`

### 3. 🧪 Test – `03_test.ipynb`
- Test độc lập (có thể cross-dataset)
- Metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion Matrix & ROC Curve

---

## 🚀 Cách chạy

```bash
pip install scikit-learn pandas numpy matplotlib seaborn openpyxl jupyter

# Download data
curl -o data/sonar.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv

# Run pipeline
python bin/run_all.py sonar
```

---

## 📊 Kết quả

- Best model: **SVM (tuned)**
- Accuracy: ~0.90
- Output:
  - `exps_/sonar_train_log.xlsx`
  - `exps_/sonar__test_sonar_log.xlsx`
  - `model/sonar/sonar__best_model.pkl`

---

## 💡 Điểm nổi bật

- Tách riêng EDA / Train / Test → dễ maintain
- Hỗ trợ **cross-test giữa datasets**
- Pipeline có thể chạy tự động bằng script
- Logging đầy đủ ra Excel

---

## 📎 Ghi chú

Project được phát triển theo đúng quy trình Machine Learning chuẩn và phục vụ mục đích học tập môn Trí Tuệ Nhân Tạo.
