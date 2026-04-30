"""
Experiment 7: Iris Flower Classification using Multiple Models

Dataset Information:
    - We have uploaded the dataset on the GitHub repository.
    - You can download it from here: https://www.kaggle.com/datasets/uciml/iris/data
    - I have downloaded the dataset, which is in the folder 'IRIS' within 'EXP_7'.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. Load Iris Dataset from Local CSV File
# ============================================================
print("=" * 70)
print(" Iris Flower Classification - Experiment 7")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "IRIS" / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs_exp7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset file not found at: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("\n[INFO] Dataset loaded successfully.")
print(f"[INFO] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"[INFO] Columns: {list(df.columns)}")

print("\n[INFO] First 5 rows:")
print(df.head().to_string(index=False))

print("\n[INFO] Dataset Summary:")
print(df.describe().to_string())

print("\n[INFO] Missing values per column:")
print(df.isnull().sum().to_string())

# ============================================================
# 2. Data Preprocessing
# ============================================================
# Drop Id column as it is not useful for prediction.
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
target_col = "Species"

X = df[feature_cols].copy()
y = df[target_col].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = list(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[INFO] Train/Test split completed.")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing samples  : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")
print(f"  Classes          : {class_names}")

# ============================================================
# 3. Define and Train 5 Classification Models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = []

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
fig.suptitle("Confusion Matrices - Iris Dataset (5 Models)", fontsize=15, fontweight="bold")

for idx, (name, model) in enumerate(models.items()):
    print("\n" + "=" * 70)
    print(f" Model {idx + 1}: {name}")
    print("=" * 70)

    # Scale-sensitive models use standardized features.
    if name in {"Logistic Regression", "SVM (RBF)", "KNN"}:
        X_train_used, X_test_used = X_train_scaled, X_test_scaled
    else:
        X_train_used, X_test_used = X_train, X_test

    model.fit(X_train_used, y_train)
    y_pred = model.predict(X_test_used)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    ax = axes[idx] if len(models) > 1 else axes
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    results.append(
        {
            "Model": name,
            "Accuracy": round(acc, 4),
        }
    )

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUTPUT_DIR / "confusion_matrices_iris.png", dpi=160)
plt.close()

print("\n[INFO] Confusion matrix figure saved.")

# ============================================================
# 4. Comparison Table and Bar Chart
# ============================================================
print("\n" + "=" * 70)
print(" MODEL COMPARISON TABLE")
print("=" * 70)

df_results = pd.DataFrame(results)
df_results.index = range(1, len(df_results) + 1)
df_results.index.name = "Sr No."
print(df_results.to_string())

df_results.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=True)

best_model = df_results.loc[df_results["Accuracy"].idxmax(), "Model"]
best_acc = df_results["Accuracy"].max()
print(f"\n>> Best Model: {best_model} with Accuracy = {best_acc:.4f}")

metrics = ["Accuracy"]
x = np.arange(len(df_results))
width = 0.5

fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = ["#1b9e77"]

for i, metric in enumerate(metrics):
    bars = ax2.bar(x + i * width, df_results[metric], width, label=metric, color=colors[i])
    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax2.set_xlabel("Model", fontsize=12)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("Model Comparison (Accuracy) - Iris", fontsize=14, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(df_results["Model"], rotation=15, ha="right")
ax2.set_ylim(0, 1.10)
ax2.grid(axis="y", alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_iris.png", dpi=160)
plt.close()

print("[INFO] Comparison chart and CSV table saved.")
print("\n[DONE] Experiment 7 complete!")
