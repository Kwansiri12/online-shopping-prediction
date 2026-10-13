
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------
# 1. กำหนดตำแหน่งไฟล์
# --------------------------------

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "C05_online_purchase_classification.xlsx"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

NUMERIC = [
    "pages_viewed",
    "session_minutes",
    "items_viewed",
]

CATEGORICAL = [
    "discount_seen",
    "returning_visitor",
    "device",
]

FEATURES = NUMERIC + CATEGORICAL
TARGET = "purchased"

# --------------------------------
# 2. โหลดข้อมูล Excel
# --------------------------------

df = pd.read_excel(DATA_FILE, sheet_name="Raw_Data")

print("จำนวนข้อมูลเดิม:", df.shape)
print("\nตัวอย่างข้อมูล:")
print(df.head())

print("\nข้อมูลค่าว่าง:")
print(df.isnull().sum())

print("\nจำนวนข้อมูลซ้ำทั้งแถว:")
print(df.duplicated().sum())

print("\nจำนวนรหัสผู้เข้าชมที่ซ้ำ:")
print(df["visitor_id"].duplicated().sum())

# --------------------------------
# 3. ทำความสะอาดข้อมูล
# --------------------------------

# ลบเฉพาะแถวที่ซ้ำกันทุกคอลัมน์
df = df.drop_duplicates().copy()

# ทำข้อความให้เป็นตัวพิมพ์เล็ก
# และลบช่องว่างส่วนเกิน
for col in CATEGORICAL:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

# จัดการค่า yes/no
for col in ["discount_seen", "returning_visitor"]:
    df[col] = df[col].where(
        df[col].isin(["yes", "no"]),
        pd.NA
    )

# จัดการประเภทอุปกรณ์
df["device"] = df["device"].where(
    df["device"].isin(["mobile", "desktop", "tablet"]),
    pd.NA
)

# แปลงค่าว่างให้ scikit-learn จัดการ
for col in CATEGORICAL:
    df[col] = (
        df[col]
        .astype(object)
        .where(df[col].notna(), np.nan)
    )

# แปลงข้อมูลตัวเลข
for col in NUMERIC:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

    # ค่าติดลบถือว่าไม่สมเหตุสมผล
    df.loc[df[col] < 0, col] = np.nan

# ตรวจสอบ Target
df[TARGET] = pd.to_numeric(
    df[TARGET], errors="coerce"
)

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

print("\nจำนวนข้อมูลหลังทำความสะอาด:", len(df))

# --------------------------------
# 4. วิเคราะห์ข้อมูลด้วยกราฟ
# --------------------------------

# กราฟที่ 1 จำนวนผู้ซื้อ / ไม่ซื้อ
plt.figure(figsize=(7, 4))

df[TARGET].map({
    0: "Not Purchased",
    1: "Purchased"
}).value_counts().plot(kind="bar")

plt.title("Purchase Outcomes")
plt.xlabel("Purchase Result")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_purchase_outcomes.png",
    dpi=160
)
plt.close()

# กราฟที่ 2 อัตราการซื้อตามอุปกรณ์
purchase_by_device = (
    df.groupby("device", dropna=False)[TARGET]
    .mean()
    .sort_values()
)

plt.figure(figsize=(7, 4))
purchase_by_device.plot(kind="bar")

plt.title("Purchase Rate by Device")
plt.ylabel("Purchase Rate")
plt.xlabel("Device")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_purchase_by_device.png",
    dpi=160
)
plt.close()

# กราฟที่ 3 ระยะเวลาที่ใช้ในเว็บไซต์
plt.figure(figsize=(7, 4))

df.boxplot(
    column="session_minutes",
    by=TARGET
)

plt.suptitle("")
plt.title("Session Duration by Purchase")
plt.xlabel("Purchased (0 or 1)")
plt.ylabel("Minutes")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_session_duration.png",
    dpi=160
)
plt.close()

# --------------------------------
# 5. กำหนด Features และ Target
# --------------------------------

X = df[FEATURES]
y = df[TARGET]

# --------------------------------
# 6. แบ่งข้อมูล Train / Test
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain:", len(X_train))
print("Test:", len(X_test))

# --------------------------------
# 7. เตรียมข้อมูลสำหรับโมเดล
# --------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, NUMERIC),
    ("categorical", categorical_transformer, CATEGORICAL)
])

# --------------------------------
# 8. สร้างโมเดลสองประเภท
# --------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42
    )
}

results = []
trained_models = {}

# --------------------------------
# 9. ฝึกและทดสอบโมเดล
# --------------------------------

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    result = {
        "model": name,
        "accuracy": round(
            float(accuracy_score(y_test, y_pred)), 4
        ),
        "precision": round(
            float(precision_score(y_test, y_pred)), 4
        ),
        "recall": round(
            float(recall_score(y_test, y_pred)), 4
        ),
        "f1": round(
            float(f1_score(y_test, y_pred)), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist()
    }

    results.append(result)
    trained_models[name] = pipeline

    print("\n======================")
    print("Model:", name)
    print("======================")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(result["confusion_matrix"])

# --------------------------------
# 10. เลือกโมเดลที่ดีที่สุด
# --------------------------------

best = max(
    results,
    key=lambda r: (r["f1"], r["accuracy"])
)

best_model = trained_models[best["model"]]

# บันทึกโมเดล
joblib.dump(
    best_model,
    MODEL_DIR / "shopping_model.joblib"
)

# บันทึกผลประเมิน
metrics = {
    "best_model": best["model"],
    "results": results,
    "rows_after_cleaning": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test)
}

with open(
    MODEL_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n======================")
print("BEST MODEL:", best["model"])
print("ACCURACY:", best["accuracy"])
print("F1-SCORE:", best["f1"])
print("======================")
print("บันทึกโมเดลเรียบร้อยแล้ว")
