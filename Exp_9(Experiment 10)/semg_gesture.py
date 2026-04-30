# Dataset note:
# We have uploaded the dataset to the GitHub repository.
# Dataset link: https://www.kaggle.com/datasets/kyr7plus/emg-4
# The dataset used locally is available in the Exp_9 folder as emg_gestures.csv or in the dataset/ subfolder.

import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


CLASS_NAMES: Dict[int, str] = {
    0: "Rock",
    1: "Scissors",
    2: "Paper",
    3: "OK",
}

PROSTHETIC_MAPPING: Dict[int, str] = {
    0: "Grip",
    1: "Wrist Extension",
    2: "Open Palm",
    3: "Pinch",
}


def find_dataset_files(base_dir: str = ".") -> List[str]:
    """Find dataset files. Prefer emg_gestures.csv, else use all CSV files in dataset/."""
    preferred = os.path.join(base_dir, "emg_gestures.csv")
    if os.path.exists(preferred):
        return [preferred]

    dataset_csvs = sorted(glob.glob(os.path.join(base_dir, "dataset", "*.csv")))
    if dataset_csvs:
        return dataset_csvs

    fallback_csvs = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    if fallback_csvs:
        return fallback_csvs

    raise FileNotFoundError("No CSV files found. Expected emg_gestures.csv or dataset/*.csv")


def load_emg_data(file_paths: List[str]) -> pd.DataFrame:
    """Load and combine CSV files into one DataFrame, ensuring a numeric label column exists."""
    frames: List[pd.DataFrame] = []

    for path in file_paths:
        df = pd.read_csv(path, header=None)

        # If the source file doesn't include labels, infer from filename (e.g., 0.csv -> label 0).
        if df.shape[1] == 0:
            continue

        last_col = df.iloc[:, -1]
        if pd.api.types.is_numeric_dtype(last_col):
            # Keep as-is if the final column already looks like labels.
            pass
        else:
            filename = os.path.splitext(os.path.basename(path))[0]
            if filename.isdigit():
                df[len(df.columns)] = int(filename)
            else:
                raise ValueError(f"Could not infer label column for {path}")

        frames.append(df)

    if not frames:
        raise ValueError("No data loaded from CSV files.")

    data = pd.concat(frames, ignore_index=True)
    data = data.apply(pd.to_numeric, errors="coerce").dropna(axis=0)
    data.iloc[:, -1] = data.iloc[:, -1].astype(int)
    return data


def infer_signal_shape(data: pd.DataFrame, n_channels: int = 8) -> int:
    """Infer timesteps per channel from flattened signal columns."""
    feature_count = data.shape[1] - 1
    if feature_count % n_channels != 0:
        raise ValueError(
            f"Feature count ({feature_count}) is not divisible by {n_channels} channels."
        )
    return feature_count // n_channels


def row_to_signal_matrix(row_features: np.ndarray, n_channels: int, timesteps: int) -> np.ndarray:
    """Convert a flattened feature row into [n_channels, timesteps]."""
    return row_features.reshape(n_channels, timesteps)


def plot_raw_signals_per_class(
    data: pd.DataFrame,
    n_channels: int,
    timesteps: int,
    class_names: Dict[int, str],
) -> None:
    """Plot raw EMG signals (all 8 channels) for one sample from each class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for i, class_id in enumerate(sorted(class_names.keys())):
        subset = data[data.iloc[:, -1] == class_id]
        if subset.empty:
            axes[i].set_title(f"{class_names[class_id]} ({class_id}) - no sample")
            continue

        sample = subset.iloc[0, :-1].to_numpy(dtype=float)
        signal_matrix = row_to_signal_matrix(sample, n_channels=n_channels, timesteps=timesteps)

        for ch in range(n_channels):
            axes[i].plot(signal_matrix[ch], label=f"Ch{ch + 1}", linewidth=1.2)

        axes[i].set_title(f"{class_names[class_id]} ({class_id})")
        axes[i].set_xlabel("Time Index")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Raw sEMG Signals (8 Channels) for One Sample per Gesture", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.96))
    plt.show()


def extract_window_features(
    signal_matrix: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Extract MAV and RMS per channel from sliding windows of one signal matrix."""
    n_channels, timesteps = signal_matrix.shape
    features: List[List[float]] = []

    for start in range(0, timesteps - window_size + 1, step_size):
        window = signal_matrix[:, start : start + window_size]
        mav = np.mean(np.abs(window), axis=1)
        rms = np.sqrt(np.mean(np.square(window), axis=1))
        feature_vector = np.concatenate([mav, rms]).tolist()
        features.append(feature_vector)

    return np.array(features)


def build_feature_dataset(
    data: pd.DataFrame,
    n_channels: int,
    timesteps: int,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create model-ready feature matrix X and target vector y using windowed MAV + RMS."""
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, row in data.iterrows():
        signal = row.iloc[:-1].to_numpy(dtype=float)
        label = int(row.iloc[-1])
        signal_matrix = row_to_signal_matrix(signal, n_channels=n_channels, timesteps=timesteps)
        sample_features = extract_window_features(signal_matrix, window_size=window_size, step_size=step_size)

        if sample_features.size == 0:
            continue

        X_list.append(sample_features)
        y_list.extend([label] * sample_features.shape[0])

    if not X_list:
        raise ValueError("No features were extracted. Check window_size/step_size and data shape.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Create the requested classifier pipelines with StandardScaler."""
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        max_features="sqrt",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
            ]
        ),
    }
    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str, np.ndarray]:
    """Train all models and return metrics, per-model predictions, and best-model predictions."""
    results = []
    predictions_by_model: Dict[str, np.ndarray] = {}
    best_name = ""
    best_score = -np.inf
    best_pred: Optional[np.ndarray] = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test))
        predictions_by_model[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "F1_Macro": f1,
            }
        )

        # Select best by F1 macro, then accuracy as tie-breaker.
        if (f1 > best_score) or (
            np.isclose(f1, best_score)
            and acc > next((r["Accuracy"] for r in results if r["Model"] == best_name), -np.inf)
        ):
            best_score = f1
            best_name = name
            best_pred = y_pred

    if best_pred is None:
        raise RuntimeError("No model produced predictions.")

    results_df = pd.DataFrame(results).sort_values(by=["F1_Macro", "Accuracy"], ascending=False)
    return results_df, predictions_by_model, best_name, best_pred


def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    title: str,
) -> None:
    """Plot confusion matrix as a heatmap for the selected best-performing model."""
    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([class_names[i] for i in labels], rotation=30, ha="right")
    ax.set_yticklabels([class_names[i] for i in labels])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()


def save_model_comparison_chart(results_df: pd.DataFrame, output_path: str) -> None:
    """Save a bar chart image comparing model Accuracy and F1_Macro."""
    chart_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    x = np.arange(len(chart_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    acc_bars = ax.bar(x - width / 2, chart_df["Accuracy"], width, label="Accuracy")
    f1_bars = ax.bar(x + width / 2, chart_df["F1_Macro"], width, label="F1 Macro")

    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["Model"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (acc_bars, f1_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # 1) Data loading
    csv_files = find_dataset_files(base_dir=".")
    print(f"Using CSV file(s): {csv_files}")
    data = load_emg_data(csv_files)

    # 2) Signal metadata
    n_channels = 8
    timesteps = infer_signal_shape(data, n_channels=n_channels)
    print(f"Loaded data shape: {data.shape} | channels: {n_channels} | timesteps/channel: {timesteps}")

    # 3) Raw signal visualization for one sample per class
    plot_raw_signals_per_class(data, n_channels=n_channels, timesteps=timesteps, class_names=CLASS_NAMES)

    # 4) Feature engineering
    # For very short recordings (e.g., 8 timesteps/channel), sliding windows collapse
    # into one tiny MAV/RMS vector and lose discriminative information.
    if timesteps <= 8:
        X = data.iloc[:, :-1].to_numpy(dtype=float)
        y = data.iloc[:, -1].to_numpy(dtype=int)
        print("Using full raw-signal features (short recording detected).")
    else:
        window_size = min(32, timesteps)
        step_size = max(1, window_size // 2)
        X, y = build_feature_dataset(
            data,
            n_channels=n_channels,
            timesteps=timesteps,
            window_size=window_size,
            step_size=step_size,
        )
        print(f"Using window features: window_size={window_size}, step_size={step_size}")

    print(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")

    # 5) Train-test split (70/30) and model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    models = build_models(random_state=42)
    results_df, predictions_by_model, best_model_name, best_predictions = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 6) Model comparison table (Accuracy and F1)
    print("\nModel Performance Comparison (70/30 split):")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    comparison_image_path = os.path.join("result", "model_comparison.png")
    save_model_comparison_chart(results_df, comparison_image_path)
    print(f"Saved model comparison image to: {comparison_image_path}")

    # 7) Visual evaluation: confusion heatmap for every model
    print("\nGenerating confusion matrix heatmaps for all models...")
    for model_name, y_pred in predictions_by_model.items():
        plot_confusion_heatmap(
            y_true=y_test,
            y_pred=y_pred,
            class_names=CLASS_NAMES,
            title=f"Confusion Matrix - {model_name}",
        )

    # Also report the best model by F1 macro.
    print(f"\nBest model by F1_Macro: {best_model_name}")

    # 8) Prosthetic action mapping
    print("\nProsthetic Mapping Dictionary:")
    print(PROSTHETIC_MAPPING)


if __name__ == "__main__":
    main()