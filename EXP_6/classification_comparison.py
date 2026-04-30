"""
EXP 6: Comparison of 5 Basic Classification Models on MNIST
=============================================================
Models: Logistic Regression, Decision Tree, Random Forest, SVM, k-NN
 Dataset: MNIST (from local dataset folder)
 Dataset link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/code
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# 1. Load MNIST from Local IDX Files
# ============================================================

def load_idx_images(filepath):
    """Read IDX image file and return numpy array."""
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images

def load_idx_labels(filepath):
    """Read IDX label file and return numpy array."""
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

print("=" * 60)
print(" Loading MNIST Dataset from Local Files")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

X_train_full = load_idx_images(DATASET_DIR / "train-images.idx3-ubyte")
y_train_full = load_idx_labels(DATASET_DIR / "train-labels.idx1-ubyte")
X_test_full  = load_idx_images(DATASET_DIR / "t10k-images.idx3-ubyte")
y_test_full  = load_idx_labels(DATASET_DIR / "t10k-labels.idx1-ubyte")

print(f"Full Training set : {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
print(f"Full Test set     : {X_test_full.shape[0]} samples")

# --- Use a subset for faster training ---
# Using 10,000 training and 2,000 test samples for speed
TRAIN_SIZE = 10000
TEST_SIZE = 2000

np.random.seed(42)
train_idx = np.random.choice(len(X_train_full), TRAIN_SIZE, replace=False)
test_idx  = np.random.choice(len(X_test_full), TEST_SIZE, replace=False)

X_train = X_train_full[train_idx]
y_train = y_train_full[train_idx]
X_test  = X_test_full[test_idx]
y_test  = y_test_full[test_idx]

print(f"\nUsing subset:")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing samples  : {X_test.shape[0]}")
print(f"  Features (pixels): {X_train.shape[1]}")

# Feature Scaling (normalize pixel values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

target_names = [str(i) for i in range(10)]

# ============================================================
# 2. Define the 5 Classification Models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "SVM": SVC(kernel='rbf', random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
}

# ============================================================
# 3. Train, Evaluate, and Collect Results
# ============================================================
results = []  # For comparison table

fig, axes = plt.subplots(1, 5, figsize=(30, 5))
fig.suptitle("Confusion Matrices – MNIST (5 Models)", fontsize=16, fontweight='bold')

for idx, (name, model) in enumerate(models.items()):
    print("\n" + "=" * 60)
    print(f" Model {idx + 1}: {name}")
    print("=" * 60)

    # Train
    print("  Training...", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("Done.")

    # Predict
    y_pred = model.predict(X_test)

    # Performance Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    # Classification Report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names,
        ax=axes[idx]
    )
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

    # Store results for comparison
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    })

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(BASE_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
print("\n[INFO] Confusion matrices saved to 'confusion_matrices.png'")

# ============================================================
# 4. Comparison Table
# ============================================================
print("\n" + "=" * 60)
print(" MODEL COMPARISON TABLE")
print("=" * 60)

df_results = pd.DataFrame(results)
df_results.index = range(1, len(df_results) + 1)
df_results.index.name = "Sr No."
print(df_results.to_string())

# Highlight the best model
best_model = df_results.loc[df_results['Accuracy'].idxmax(), 'Model']
best_acc = df_results['Accuracy'].max()
print(f"\n>> Best Model: {best_model} with Accuracy = {best_acc:.4f}")

# ============================================================
# 5. Bar Chart Comparison
# ============================================================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(df_results))
width = 0.18

fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, metric in enumerate(metrics):
    bars = ax2.bar(x + i * width, df_results[metric], width, label=metric, color=colors[i])
    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7
        )

ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Performance Comparison of 5 Classification Models (MNIST)', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(df_results['Model'], rotation=15, ha='right')
ax2.set_ylim(0, 1.15)
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
print("[INFO] Bar chart saved to 'model_comparison.png'")

plt.show()
print("\n[DONE] Experiment 6 complete!")
