# ==========================================
# PHÂN TÍCH PIMA INDIANS DIABETES
# CHƯƠNG 4.3 → 6.3
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sns.set(style="whitegrid")

# ========================
# LOAD DATA
# ========================

df = pd.read_csv("pima-indians-diabetes.data.csv", header=None)

df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

counts = df["Outcome"].value_counts()
percent = counts / len(df) * 100

plt.figure()
bars = plt.bar(["Không mắc","Mắc bệnh"],
               percent.values,
               color=["steelblue","orange"])

for i, v in enumerate(percent.values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')

plt.title("4.2.1 Phân bố kết quả bệnh")
plt.ylabel("Phần trăm (%)")
plt.show()

print("\n4.2.1 Phân bố:")
print(counts)
print(percent.round(2))

# -----------------------------------------------------
# 4.2.2 So sánh phân phối các chỉ số (KDE Model)
# -----------------------------------------------------

print("\n4.2.2 So sánh phân phối các chỉ số giữa 2 nhóm")
import seaborn as sns

features = ["Pregnancies","Glucose","BloodPressure",
            "SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age"]

plt.figure(figsize=(16,10))

for i, col in enumerate(features, 1):

    plt.subplot(2,4,i)

    # Vẽ KDE 2 nhóm
    sns.kdeplot(data=df[df["Outcome"]==0],
                x=col,
                fill=True,
                label="Không mắc",
                color="steelblue")

    sns.kdeplot(data=df[df["Outcome"]==1],
                x=col,
                fill=True,
                label="Mắc",
                color="orange")

    # Tính mean
    mean_0 = df[df["Outcome"]==0][col].mean()
    mean_1 = df[df["Outcome"]==1][col].mean()

    # Vẽ đường mean
    plt.axvline(mean_0, color="blue", linestyle="--",
                label=f"Mean Không mắc: {mean_0:.1f}")

    plt.axvline(mean_1, color="orange", linestyle="--",
                label=f"Mean Mắc: {mean_1:.1f}")

    plt.title(f"Phân phối {col}")
    plt.ylabel("Density")
    plt.legend(fontsize=8)

plt.tight_layout()
plt.show()
# ==========================================================
# 4.3 PHÂN TÍCH THEO NHÓM TUỔI
# ==========================================================

df["AgeGroup"] = pd.cut(df["Age"],
                        bins=[20,30,40,50,60,100],
                        labels=["20-30","30-40","40-50","50-60","60+"],
                        right=False)

# 4.3.1

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Age",
    hue="Outcome",
    bins=20,
    kde=True,
    stat="count",
    multiple="layer",
    palette={0:"steelblue", 1:"salmon"},
    alpha=0.6
)

plt.title("Phân bố bệnh theo độ tuổi")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Outcome")
plt.show()

import pandas as pd

bins = [20, 30, 40, 50, 100]
labels = ["21-30", "31-40", "41-50", ">50"]

df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

# Bảng số lượng
table = pd.crosstab(df["AgeGroup"], df["Outcome"])

print("Số lượng theo nhóm tuổi:")
print(table)

# Tính tỷ lệ %
rate = table.div(table.sum(axis=1), axis=0) * 100

print("\nTỷ lệ mắc bệnh theo nhóm tuổi (%):")
print(rate.round(2))

# 4.3.2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

fig, axes = plt.subplots(1,3, figsize=(15,4))

# 1️⃣ Scatter + Regression
sns.regplot(x="Age", y="BMI", data=df,
            scatter_kws={"alpha":0.5},
            line_kws={"color":"red"},
            ax=axes[0])
axes[0].set_title("BMI tăng dần theo tuổi")

# 2️⃣ Boxplot theo nhóm tuổi
sns.boxplot(x="AgeGroup", y="BMI", data=df, ax=axes[1])
axes[1].set_title("Phân bố BMI theo nhóm tuổi")
axes[1].tick_params(axis='x', rotation=45)

# 3️⃣ Line trung bình theo tuổi
mean_bmi = df.groupby("AgeGroup")["BMI"].mean()

axes[2].plot(mean_bmi.index, mean_bmi.values, marker="o")
axes[2].axhline(mean_bmi.mean(), color="red", linestyle="--")
axes[2].set_title("Xu hướng BMI trung bình theo tuổi")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,3, figsize=(15,4))

# 1️⃣ Scatter + Regression
sns.regplot(x="Age", y="Glucose", data=df,
            scatter_kws={"alpha":0.5},
            line_kws={"color":"red"},
            ax=axes[0])
axes[0].set_title("Glucose tăng theo tuổi")

# 2️⃣ Boxplot
sns.boxplot(x="AgeGroup", y="Glucose", data=df, ax=axes[1])
axes[1].set_title("Phân bố Glucose theo nhóm tuổi")
axes[1].tick_params(axis='x', rotation=45)

# 3️⃣ Line trung bình
mean_glucose = df.groupby("AgeGroup")["Glucose"].mean()

axes[2].plot(mean_glucose.index, mean_glucose.values, marker="o")
axes[2].axhline(126, color="red", linestyle="--")
axes[2].set_title("Xu hướng Glucose trung bình theo tuổi")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,3, figsize=(15,4))

# 1️⃣ Scatter + Regression
sns.regplot(x="Age", y="BloodPressure", data=df,
            scatter_kws={"alpha":0.5},
            line_kws={"color":"red"},
            ax=axes[0])
axes[0].set_title("Huyết áp tăng tuyến tính với tuổi")

# 2️⃣ Boxplot
sns.boxplot(x="AgeGroup", y="BloodPressure", data=df, ax=axes[1])
axes[1].set_title("Phân bố huyết áp theo nhóm tuổi")
axes[1].tick_params(axis='x', rotation=45)

# 3️⃣ Line trung bình
mean_bp = df.groupby("AgeGroup")["BloodPressure"].mean()

axes[2].plot(mean_bp.index, mean_bp.values, marker="o")
axes[2].axhline(80, color="red", linestyle="--")
axes[2].set_title("Xu hướng huyết áp trung bình theo tuổi")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 4.3.3
plt.figure(figsize=(6,4))

age_risk = df.groupby("AgeGroup")["Outcome"].mean()

plt.bar(age_risk.index, age_risk.values, color="tomato")
plt.title("Tỷ lệ mắc bệnh theo nhóm tuổi")
plt.ylabel("Tỷ lệ mắc")
plt.xticks(rotation=45)

plt.show()

# ==========================================================
# 4.4 PHÂN TÍCH BMI
# ==========================================================

df["BMI_Group"] = pd.cut(df["BMI"],
                         bins=[0,18.5,25,30,100],
                         labels=["Gầy","Bình thường","Thừa cân","Béo phì"],
                         right=False)

# 4.4.1
# Phân loại BMI theo WHO
def bmi_category(bmi):
    if bmi < 18.5:
        return "Gầy"
    elif bmi < 25:
        return "Bình thường"
    elif bmi < 30:
        return "Thừa cân"
    else:
        return "Béo phì"

df["BMI_Cat"] = df["BMI"].apply(bmi_category)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

sns.countplot(data=df,
              x="BMI_Cat",
              hue="Outcome",
              order=["Béo phì","Thừa cân","Bình thường","Gầy"])

plt.title("Tỷ lệ mắc bệnh theo nhóm BMI")
plt.xlabel("BMI_Cat")
plt.ylabel("Count")
plt.legend(title="Outcome")
plt.show()

# 4.4.2
import pandas as pd
import matplotlib.pyplot as plt

# Tạo bảng thống kê
order = ["Gầy", "Bình thường", "Thừa cân", "Béo phì"]

bmi_table = df.groupby("BMI_Cat").agg(
    Tổng_số=("Outcome","count"),
    Số_ca_mắc=("Outcome","sum"),
    BMI_trung_bình=("BMI","mean")
)

bmi_table["Tỷ lệ mắc (%)"] = round(
    bmi_table["Số_ca_mắc"] / bmi_table["Tổng_số"] * 100, 1
)

bmi_table = bmi_table.reindex(order)
bmi_table["BMI_trung_bình"] = round(bmi_table["BMI_trung_bình"],1)

# Reset index để hiện cột Nhóm BMI
bmi_table = bmi_table.reset_index()
bmi_table.rename(columns={"BMI_Cat":"Nhóm BMI"}, inplace=True)

# VẼ BẢNG
fig, ax = plt.subplots(figsize=(9,4))
ax.axis('off')

table = ax.table(
    cellText=bmi_table.values,
    colLabels=bmi_table.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

plt.title("Bảng 4.1: Tỷ lệ mắc bệnh theo nhóm BMI", pad=20)
plt.show()


# 4.4.3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

plt.figure(figsize=(7,5))

sns.scatterplot(data=df,
                x="BMI",
                y="Glucose",
                hue="Outcome",
                palette=["#1f77b4", "#ff7f0e"])

plt.title("BMI vs Glucose theo Outcome")
plt.xlabel("BMI")
plt.ylabel("Glucose")

plt.legend(title="Outcome")
plt.show()

plt.figure(figsize=(7,5))

sns.scatterplot(data=df,
                x="BMI",
                y="Age",
                hue="Outcome",
                palette=["#1f77b4", "#ff7f0e"])

plt.title("BMI vs Age theo Outcome")
plt.xlabel("BMI")
plt.ylabel("Age")

plt.legend(title="Outcome")
plt.show()

plt.figure(figsize=(7,5))

sns.scatterplot(data=df,
                x="BMI",
                y="Insulin",
                hue="Outcome",
                palette=["#1f77b4", "#ff7f0e"])

plt.title("BMI vs Insulin theo Outcome")
plt.xlabel("BMI")
plt.ylabel("Insulin")

plt.legend(title="Outcome")
plt.show()


# ==========================================================
# 4.5 GLUCOSE & INSULIN
# ==========================================================

# 4.5.1
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")  # nền xám giống PDF

plt.figure(figsize=(7,5))

# Outcome 0 (không bệnh)
sns.histplot(df[df["Outcome"]==0]["Glucose"],
             bins=20,
             kde=True,
             color="#1f77b4",
             alpha=0.5,
             label="0")

# Outcome 1 (bệnh)
sns.histplot(df[df["Outcome"]==1]["Glucose"],
             bins=20,
             kde=True,
             color="red",
             alpha=0.5,
             label="1")

plt.title("Phân bố nồng độ Glucose")
plt.xlabel("Glucose")
plt.ylabel("Count")

plt.legend(title="Outcome")
plt.show()

# Phân nhóm Glucose theo ADA
def glucose_group(x):
    if x < 100:
        return "Normal"
    elif x < 126:
        return "Prediabetes"
    else:
        return "Diabetes"

df["GlucoseGroup"] = df["Glucose"].apply(glucose_group)

# Tạo bảng thống kê
table = pd.crosstab(df["GlucoseGroup"], df["Outcome"])

table["Total"] = table[0] + table[1]
table["% mắc"] = round(table[1] / table["Total"] * 100, 1)

print(table)


# 4.5.2
plt.figure()
sns.scatterplot(x="Glucose", y="Insulin", hue="Outcome", data=df)
plt.title("4.5.2 Glucose và Insulin")
plt.show()

# ==========================================================
# 4.6 DI TRUYỀN
# ==========================================================

# 4.6.1
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")  # nền xám giống PDF

plt.figure(figsize=(7,5))

sns.histplot(df["DiabetesPedigreeFunction"],
             bins=20,
             kde=True,
             color="green",
             alpha=0.6)

plt.title("Phân bố Diabetes Pedigree Function")
plt.xlabel("DiabetesPedigreeFunction")
plt.ylabel("Count")

plt.xlim(0, 2.5)  # giống PDF
plt.show()

# 4.6.2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

plt.figure(figsize=(7,5))

sns.boxplot(data=df,
            x="Outcome",
            y="DiabetesPedigreeFunction",
            palette=["#66c2a5", "#fc8d62"])

plt.title("Tác động của yếu tố di truyền tới nguy cơ bệnh")
plt.xlabel("Outcome")
plt.ylabel("DiabetesPedigreeFunction")

plt.ylim(0, 2.5)
plt.show()

# 4.6.3
plt.figure()
sns.scatterplot(x="DiabetesPedigreeFunction",
                y="Glucose",
                hue="Outcome",
                data=df)
plt.title("4.6.3 Di truyền kết hợp Glucose")
plt.show()

# ==========================================================
# 4.7 SỐ LẦN CÓ THAI
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

plt.figure(figsize=(7,5))

sns.boxplot(data=df,
            x="Outcome",
            y="Pregnancies",
            order=[0,1],
            palette=["#66c2a5", "#fc8d62"])

plt.title("Số lần có thai theo Outcome")
plt.xlabel("Outcome")
plt.ylabel("Pregnancies")

plt.ylim(0, 18)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Tính tỷ lệ mắc bệnh theo số lần có thai
preg_risk = df.groupby("Pregnancies")["Outcome"].mean()

plt.figure(figsize=(8,5))

plt.bar(preg_risk.index,
        preg_risk.values,
        color="purple")

plt.title("Nguy cơ đái tháo đường theo số lần có thai")
plt.xlabel("Pregnancies")
plt.ylabel("Tỷ lệ mắc")

plt.ylim(0,1)
plt.show()

# ==========================================================
# CHƯƠNG 5
# ==========================================================

# 5.1 Ma trận tương quan
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

sns.set_style("darkgrid")

fig, axes = plt.subplots(2,2, figsize=(12,8))

# =====================
# 1️⃣ KDE Distribution
# =====================
sns.kdeplot(df[df["Outcome"]==0]["Glucose"],
            fill=True,
            color="#1f77b4",
            label="Không bệnh",
            ax=axes[0,0])

sns.kdeplot(df[df["Outcome"]==1]["Glucose"],
            fill=True,
            color="#ff7f0e",
            label="Mắc bệnh",
            ax=axes[0,0])

axes[0,0].axvline(100, color="orange", linestyle="--")
axes[0,0].axvline(126, color="red", linestyle="--")
axes[0,0].set_title("Phân bố Glucose theo Outcome")
axes[0,0].legend()

# =====================
# 2️⃣ Boxplot
# =====================
sns.boxplot(data=df,
            x="Outcome",
            y="Glucose",
            palette=["#1f77b4","#ff7f0e"],
            ax=axes[0,1])

axes[0,1].set_title("So sánh Glucose theo nhóm")

# =====================
# 3️⃣ Tỷ lệ mắc theo ngưỡng
# =====================
bins = [0,100,126,140,300]
labels = ["<100","100-126","126-140",">140"]

df["Glucose_group"] = pd.cut(df["Glucose"], bins=bins, labels=labels)

risk = df.groupby("Glucose_group")["Outcome"].mean()*100

axes[1,0].bar(risk.index, risk.values, color="salmon")
axes[1,0].set_title("Tỷ lệ mắc theo ngưỡng Glucose")
axes[1,0].set_ylabel("Tỷ lệ (%)")

# =====================
# 4️⃣ ROC Curve
# =====================
fpr, tpr, _ = roc_curve(df["Outcome"], df["Glucose"])
roc_auc = auc(fpr, tpr)

axes[1,1].plot(fpr, tpr)
axes[1,1].plot([0,1],[0,1],"k--")
axes[1,1].set_title(f"Đường cong ROC - Glucose\nAUC = {roc_auc:.2f}")
axes[1,1].set_xlabel("False Positive Rate")
axes[1,1].set_ylabel("True Positive Rate")

plt.tight_layout()
plt.show()


# 5.2.2 BMI - SkinThickness
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("darkgrid")

fig, axes = plt.subplots(2,2, figsize=(12,8))

# =========================
# 1️⃣ Scatter theo Outcome
# =========================
sns.scatterplot(data=df,
                x="BMI",
                y="SkinThickness",
                hue="Outcome",
                palette=["#1f77b4","#ff7f0e"],
                ax=axes[0,0])

axes[0,0].set_title("Mối quan hệ BMI - Độ dày da")

# =========================
# 2️⃣ Hồi quy tuyến tính
# =========================
sns.regplot(data=df,
            x="BMI",
            y="SkinThickness",
            scatter_kws={"alpha":0.4},
            line_kws={"color":"red"},
            ax=axes[0,1])

axes[0,1].set_title("Hồi quy tuyến tính BMI - Độ dày da")

# =========================
# 3️⃣ Boxplot theo nhóm BMI
# =========================
sns.boxplot(data=df,
            x="BMI_Cat",
            y="SkinThickness",
            order=["Bình thường","Thừa cân","Béo phì"],
            ax=axes[1,0])

axes[1,0].set_title("Độ dày da theo nhóm BMI")

# =========================
# 4️⃣ Heatmap bảng chéo
# =========================
# Chia SkinThickness thành 4 nhóm
# 4️⃣ Heatmap bảng chéo

df["SkinThickness"] = df["SkinThickness"].replace(0, np.nan)
df["Skin_group"] = pd.qcut(df["SkinThickness"], 4, duplicates="drop")


cross = pd.crosstab(df["Skin_group"], df["BMI_Cat"])

sns.heatmap(cross,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=axes[1,1])

axes[1,1].set_title("Phân bố kết hợp BMI - Độ dày da")

plt.tight_layout()
plt.show()


# 5.2.3 Age - Pregnancies
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("darkgrid")

fig, axes = plt.subplots(2,2, figsize=(12,8))

# =========================
# 1️⃣ Scatter + Regression
# =========================
sns.regplot(data=df,
            x="Age",
            y="Pregnancies",
            scatter_kws={"alpha":0.5},
            line_kws={"color":"red"},
            ax=axes[0,0])

axes[0,0].set_title("Mối quan hệ Tuổi - Số lần có thai")

# =========================
# 2️⃣ Trung bình theo nhóm tuổi
# =========================
bins = [20,30,40,50,60,100]
labels = ["20-30","30-40","40-50","50-60","60+"]

df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

mean_preg = df.groupby("AgeGroup")["Pregnancies"].mean()

axes[0,1].bar(mean_preg.index,
              mean_preg.values,
              color="red")

axes[0,1].set_title("Số lần có thai trung bình theo nhóm tuổi")
axes[0,1].set_ylabel("Pregnancies")

# =========================
# 3️⃣ Scatter theo Outcome
# =========================
sns.scatterplot(data=df,
                x="Age",
                y="Pregnancies",
                hue="Outcome",
                palette=["blue","red"],
                ax=axes[1,0])

axes[1,0].set_title("Phân tách theo Outcome")

# =========================
# 4️⃣ Heatmap tương quan
# =========================
corr = df[["Age","Pregnancies","Outcome"]].corr()

sns.heatmap(corr,
            annot=True,
            cmap="Reds",
            vmin=0,
            vmax=1,
            ax=axes[1,1])

axes[1,1].set_title("Ma trận tương quan nhỏ")

plt.tight_layout()
plt.show()


# 5.2.4 Glucose - Insulin
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("darkgrid")

fig, axes = plt.subplots(2,2, figsize=(12,8))

# =========================
# 1️⃣ Scatter theo Outcome
# =========================
sns.scatterplot(data=df,
                x="Glucose",
                y="Insulin",
                hue="Outcome",
                palette="coolwarm",
                ax=axes[0,0])

sns.regplot(data=df,
            x="Glucose",
            y="Insulin",
            scatter=False,
            color="red",
            ax=axes[0,0])

axes[0,0].set_title("Glucose - Insulin theo Outcome")

# =========================
# 2️⃣ Insulin trung bình theo nhóm Glucose
# =========================
bins = [0,100,126,200]
labels = ["Bình thường","Tiền ĐTĐ","ĐTĐ"]

df["Glu_group"] = pd.cut(df["Glucose"], bins=bins, labels=labels)

mean_ins = df.groupby("Glu_group")["Insulin"].mean()

axes[0,1].bar(mean_ins.index,
              mean_ins.values,
              color="orange")

axes[0,1].set_title("Insulin trung bình theo nhóm Glucose")
axes[0,1].set_ylabel("Insulin trung bình")

# =========================
# 3️⃣ Đường cong Glucose theo Insulin
# =========================
ins_bins = pd.qcut(df["Insulin"].replace(0,np.nan), 4, duplicates="drop")
mean_glu = df.groupby(ins_bins)["Glucose"].mean()

axes[1,0].plot(["Thấp","Trung bình","Cao","Rất cao"][:len(mean_glu)],
               mean_glu.values,
               marker="o",
               color="blue")

axes[1,0].set_title("Đường cong Glucose theo Insulin")
axes[1,0].set_ylabel("Glucose trung bình")

# =========================
# 4️⃣ Phát hiện suy tế bào beta
# =========================
axes[1,1].scatter(df["Glucose"],
                  df["Insulin"],
                  alpha=0.4,
                  color="gray")

axes[1,1].axvline(180, color="red", linestyle="--")
axes[1,1].axhline(100, color="red", linestyle="--")

axes[1,1].set_title("Phát hiện suy tế bào beta")
axes[1,1].set_xlabel("Glucose")
axes[1,1].set_ylabel("Insulin")

plt.tight_layout()
plt.show()


# ==========================================================
# 5.3 TẦM QUAN TRỌNG ĐẶC TRƯNG
# ==========================================================

X = df.drop(["Outcome","AgeGroup","BMI_Group"], axis=1, errors="ignore")
y = df["Outcome"]

# 5.3.1 Correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Chọn đặc trưng
features = ["Glucose","Age","Pregnancies","BMI",
            "DiabetesPedigreeFunction","Insulin",
            "SkinThickness","BloodPressure"]

X = df[features]
y = df["Outcome"]

# Chi-square yêu cầu dữ liệu >=0 → scale
X = X.fillna(X.median())
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi_scores, p_values = chi2(X_scaled, y)

chi_df = pd.DataFrame({
    "Đặc trưng": features,
    "Chi-square": chi_scores,
    "p-value": p_values
})

chi_df = chi_df.sort_values("Chi-square", ascending=False)

print(chi_df)


# 5.3.2 Mutual Information
from sklearn.feature_selection import mutual_info_classif

features = ["Glucose","Age","BMI","Pregnancies",
            "Insulin","DiabetesPedigreeFunction",
            "BloodPressure","SkinThickness"]

X = df[features].copy()
y = df["Outcome"]

# Điền NaN nếu có
X = X.fillna(X.median())

# Tính Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    "Đặc trưng": features,
    "Mutual Information": mi_scores
})

mi_df = mi_df.sort_values("Mutual Information", ascending=False)

print(mi_df)


# 5.3.3 Random Forest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features = ["Glucose","BMI","Age","DiabetesPedigreeFunction",
            "Insulin","BloodPressure","Pregnancies","SkinThickness"]

X = df[features].copy()
y = df["Outcome"]

# Xử lý NaN nếu có
X = X.fillna(X.median())

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X, y)

# Lấy importance
importances = rf.feature_importances_

rf_df = pd.DataFrame({
    "Đặc trưng": features,
    "Random Forest Importance": importances
})

rf_df = rf_df.sort_values("Random Forest Importance", ascending=False)

print(rf_df)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Random Forest Importance",
    y="Đặc trưng",
    hue="Đặc trưng",
    data=rf_df,
    palette="viridis",
    legend=False
)


plt.title("Tầm quan trọng của các đặc trưng (Random Forest)")
plt.xlabel("Random Forest Importance")
plt.ylabel("Đặc trưng")

plt.tight_layout()
plt.show()

# ==========================================================
# 5.4 ĐA CỘNG TUYẾN
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

features = ["Pregnancies","Glucose","BloodPressure",
            "SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age"]

X = df[features].copy()

# Xử lý NaN nếu có
X = X.fillna(X.median())

# Thêm constant (bắt buộc cho statsmodels)
from statsmodels.tools.tools import add_constant
X_const = add_constant(X)

# Tính VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [
    variance_inflation_factor(X_const.values, i+1)
    for i in range(len(features))
]

# Đánh giá
vif_data["Đánh giá"] = np.where(
    vif_data["VIF"] < 5,
    "Tốt",
    "Cảnh báo"
)

# Sắp xếp giảm dần
vif_data = vif_data.sort_values("VIF", ascending=False)

print(vif_data)

plt.figure(figsize=(8,5))

sns.barplot(
    x="VIF",
    y="Feature",
    hue="Feature",
    data=vif_data,
    palette="viridis",
    legend=False
)

plt.title("Biểu đồ hệ số phóng đại phương sai (VIF)")
plt.xlabel("VIF")
plt.ylabel("Đặc trưng")

plt.tight_layout()
plt.show()


# ==========================================================
# CHƯƠNG 6
# ==========================================================

# 6.1.1 KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chọn biến phân cụm
cluster_features = ["Glucose", "BMI", "Age"]

X_cluster = df[cluster_features].copy()
X_cluster = X_cluster.fillna(X_cluster.median())

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Glucose",
    y="BMI",
    hue="Cluster",
    palette="Set2",
    data=df,
    alpha=0.7
)

plt.title("Phân cụm bệnh nhân (k=4) theo Glucose và BMI")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.legend(title="Cụm")
plt.tight_layout()
plt.show()

summary_list = []

for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]

    count = len(sub)
    percent = round(count / len(df) * 100, 1)

    age_mean = sub["Age"].mean()
    age_std = sub["Age"].std()

    bmi_mean = sub["BMI"].mean()
    bmi_std = sub["BMI"].std()

    glu_mean = sub["Glucose"].mean()
    glu_std = sub["Glucose"].std()

    diabetes_rate = round(sub["Outcome"].mean() * 100, 1)

    summary_list.append([
        c,
        count,
        percent,
        age_mean, age_std,
        bmi_mean, bmi_std,
        glu_mean, glu_std,
        diabetes_rate
    ])

cluster_summary = pd.DataFrame(summary_list, columns=[
    "Cụm",
    "Số bệnh nhân",
    "Tỷ lệ (%)",
    "Tuổi TB", "Tuổi SD",
    "BMI TB", "BMI SD",
    "Glucose TB", "Glucose SD",
    "Tỷ lệ mắc bệnh (%)"
])

print(cluster_summary)

for _, row in cluster_summary.iterrows():
    print(f"\nCỤM {int(row['Cụm'])} ({int(row['Số bệnh nhân'])} bệnh nhân - {row['Tỷ lệ (%)']}%):")
    print(f" - Tuổi trung bình: {row['Tuổi TB']:.1f} ± {row['Tuổi SD']:.1f}")
    print(f" - BMI trung bình: {row['BMI TB']:.1f} ± {row['BMI SD']:.1f}")
    print(f" - Glucose trung bình: {row['Glucose TB']:.1f} ± {row['Glucose SD']:.1f}")
    print(f" - Tỷ lệ mắc bệnh: {row['Tỷ lệ mắc bệnh (%)']}%")


# 6.1.2 Đặc điểm cụm
plt.figure(figsize=(12,10))

# =========================
# 1️⃣ Scatter Age - Glucose
# =========================
plt.subplot(2,2,1)

sns.scatterplot(
    x="Age",
    y="Glucose",
    hue="Cluster",
    palette="Set2",
    data=df,
    alpha=0.7
)

plt.title("Phân bố cụm theo Tuổi và Glucose")
plt.xlabel("Tuổi")
plt.ylabel("Glucose")
plt.legend(title="Cụm")


# =========================
# 2️⃣ Scatter BMI - Glucose
# =========================
plt.subplot(2,2,2)

sns.scatterplot(
    x="BMI",
    y="Glucose",
    hue="Cluster",
    palette="Set2",
    data=df,
    alpha=0.7,
    legend=False
)

plt.title("Phân bố cụm theo BMI và Glucose")
plt.xlabel("BMI")
plt.ylabel("Glucose")


# =========================
# 3️⃣ Tỷ lệ mắc bệnh theo cụm
# =========================
plt.subplot(2,2,3)

diabetes_rate = df.groupby("Cluster")["Outcome"].mean() * 100

sns.barplot(
    x=diabetes_rate.index,
    y=diabetes_rate.values,
    palette="pastel"
)

plt.title("Tỷ lệ mắc bệnh theo từng cụm")
plt.xlabel("Cụm")
plt.ylabel("Tỷ lệ mắc bệnh (%)")

for i, v in enumerate(diabetes_rate.values):
    plt.text(i, v+1, f"{v:.1f}%", ha='center')


# =========================
# 4️⃣ Phân bố số lượng bệnh nhân
# =========================
plt.subplot(2,2,4)

cluster_counts = df["Cluster"].value_counts().sort_index()

plt.pie(
    cluster_counts,
    labels=[f"Cụm {i}" for i in cluster_counts.index],
    autopct="%1.1f%%"
)

plt.title("Phân bố số lượng bệnh nhân theo cụm")


plt.tight_layout()
plt.show()

# 6.2.1 Xếp hạng yếu tố nguy cơ
import statsmodels.api as sm
import numpy as np
import pandas as pd

features = ["Glucose","BMI","Age","Pregnancies",
            "DiabetesPedigreeFunction","BloodPressure",
            "Insulin","SkinThickness"]

X = df[features].copy()
y = df["Outcome"]

# Xử lý NaN
X = X.fillna(X.median())

# Thêm constant
X_const = sm.add_constant(X)

# Logistic regression
model = sm.Logit(y, X_const).fit()

params = model.params
conf = model.conf_int()
pvals = model.pvalues

or_df = pd.DataFrame({
    "Yếu tố nguy cơ": params.index,
    "OR": np.exp(params),
    "CI thấp": np.exp(conf[0]),
    "CI cao": np.exp(conf[1]),
    "P-value": pvals
})

# Bỏ constant
or_df = or_df.drop("const")

or_df = or_df.reset_index(drop=True)

def classify_or(or_value):
    if or_value >= 3:
        return "Rất cao"
    elif or_value >= 2:
        return "Vừa"
    else:
        return "Thấp"

or_df["Phân loại"] = or_df["OR"].apply(classify_or)

print(or_df)

# 6.2.2 Risk Score
# =============================
# 1️⃣ Tạo biến nhị phân
# =============================
df["Glu_high"] = (df["Glucose"] >= 126).astype(int)
df["BMI_high"] = (df["BMI"] >= 35).astype(int)
df["Age_high"] = (df["Age"] >= 45).astype(int)
df["Preg_high"] = (df["Pregnancies"] >= 4).astype(int)
df["DPF_high"] = (df["DiabetesPedigreeFunction"] >= 0.5).astype(int)
df["BP_high"] = (df["BloodPressure"] >= 90).astype(int)
df["Ins_high"] = (df["Insulin"] >= 166).astype(int)
df["Skin_high"] = (df["SkinThickness"] >= 30).astype(int)

# =============================
# 2️⃣ Tính RiskScore
# =============================
df["RiskScore"] = (
    df["Glu_high"] * 4 +
    df["BMI_high"] * 2 +
    df["Age_high"] * 2 +
    df["Preg_high"] * 2 +
    df["DPF_high"] * 2 +
    df["BP_high"] * 1 +
    df["Ins_high"] * 2 +
    df["Skin_high"] * 1
)

# =============================
# 3️⃣ Phân loại RiskLevel
# =============================
def classify_risk(score):
    if score >= 10:
        return "Nguy cơ rất cao (≥10)"
    elif score >= 7:
        return "Nguy cơ cao (7-9)"
    elif score >= 4:
        return "Nguy cơ vừa (4-6)"
    else:
        return "Nguy cơ thấp (0-3)"

df["RiskLevel"] = df["RiskScore"].apply(classify_risk)

# =============================
# 4️⃣ Tạo bảng thống kê
# =============================
risk_table = df.groupby("RiskLevel").agg(
    Tỉ_lệ_mắc_bệnh=("Outcome", lambda x: round(x.mean()*100,1)),
    Số_BN=("Outcome","count")
).reset_index()

# Sắp xếp đúng thứ tự
order = [
    "Nguy cơ thấp (0-3)",
    "Nguy cơ vừa (4-6)",
    "Nguy cơ cao (7-9)",
    "Nguy cơ rất cao (≥10)"
]

risk_table["RiskLevel"] = pd.Categorical(
    risk_table["RiskLevel"],
    categories=order,
    ordered=True
)

risk_table = risk_table.sort_values("RiskLevel")

print(risk_table)

# =============================
# 5️⃣ Vẽ biểu đồ (KHÔNG warning)
# =============================
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

sns.barplot(
    x="RiskLevel",
    y="Tỉ_lệ_mắc_bệnh",
    hue="RiskLevel",        # bắt buộc thêm
    data=risk_table,
    palette="Reds",
    legend=False
)

plt.xticks(rotation=30)
plt.ylabel("Tỉ lệ mắc bệnh (%)")
plt.title("Tỉ lệ mắc bệnh theo mức nguy cơ")

plt.tight_layout()
plt.show()


# 6.2.3 Phân tích ngưỡng
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd

def roc_analysis(y_true, predictor, name):
    
    fpr, tpr, thresholds = roc_curve(y_true, predictor)
    auc_score = roc_auc_score(y_true, predictor)
    
    # Youden Index
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    
    best_threshold = thresholds[best_idx]
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]
    
    return {
        "Biến": name,
        "Ngưỡng tối ưu": round(best_threshold,1),
        "AUC": round(auc_score,2),
        "Độ nhạy": round(sensitivity*100,1),
        "Độ đặc hiệu": round(specificity*100,1)
    }

results = []

results.append(roc_analysis(df["Outcome"], df["Glucose"], "Glucose"))
results.append(roc_analysis(df["Outcome"], df["BMI"], "BMI"))
results.append(roc_analysis(df["Outcome"], df["Age"], "Age"))
results.append(roc_analysis(df["Outcome"], df["RiskScore"], "RiskScore"))

roc_table = pd.DataFrame(results)

print(roc_table)

# 6.3.1 WHO/ADA cutoff
df["WHO_Predict"] = (df["Glucose"] >= 126).astype(int)

print("\n6.3.1 So sánh với tiêu chí WHO:")
print(pd.crosstab(df["WHO_Predict"], df["Outcome"]))

# 6.3.2 Accuracy
import numpy as np
import pandas as pd

def evaluate_cutoff(y_true, predictor, threshold):
    
    y_pred = (predictor >= threshold).astype(int)
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    sensitivity = TP / (TP + FN) if (TP+FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN+FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP+FP) > 0 else 0
    npv = TN / (TN + FN) if (TN+FN) > 0 else 0
    accuracy = (TP + TN) / len(y_true)
    
    return [
        threshold,
        round(sensitivity*100,1),
        round(specificity*100,1),
        round(ppv*100,1),
        round(npv*100,1),
        round(accuracy*100,1)
    ]

cutoffs = [100,110,120,130,140,150,160]

results = []

for c in cutoffs:
    results.append(
        evaluate_cutoff(df["Outcome"], df["Glucose"], c)
    )

cutoff_table = pd.DataFrame(results, columns=[
    "Ngưỡng Glucose (mg/dL)",
    "Độ nhạy (%)",
    "Độ đặc hiệu (%)",
    "PPV (%)",
    "NPV (%)",
    "Độ chính xác (%)"
])

print(cutoff_table)
