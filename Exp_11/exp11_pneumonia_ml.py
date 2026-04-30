# Dataset note:
# We have uploaded the dataset to the GitHub repository.
# Dataset link: https://www.kaggle.com/datasets/aadigupta1601/chest-x-ray-pneumonia-numerical-feature-dataset
# The dataset used locally is available in the Exp_11/dataset folder.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
EPS = 1e-8

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIGS_DIR = OUT_DIR / "figures"

GLOBAL_FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "p10",
    "p90",
    "skewness",
    "kurtosis",
    "entropy",
]
GLCM_FEATURES = [
    "glcm_contrast",
    "glcm_energy",
    "glcm_homogeneity",
    "glcm_correlation",
]
FFT_FEATURES = ["fft_mean", "fft_std", "fft_energy"]
EDGE_FEATURES = ["edge_density", "edge_count", "edge_strength"]
LBP_FEATURES = [f"lbp_{i}" for i in range(10)]

FEATURE_GROUPS = {
    "global_intensity": GLOBAL_FEATURES,
    "glcm_texture": GLCM_FEATURES,
    "fft_frequency": FFT_FEATURES,
    "edge_metrics": EDGE_FEATURES,
    "lbp_texture": LBP_FEATURES,
}


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def derive_label(image_name: str) -> int:
    name = str(image_name).lower()
    pneumonia_tokens = ("person", "pneumonia", "bacteria", "virus")
    return int(any(token in name for token in pneumonia_tokens))


def load_split(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename)
    df["label"] = df["image"].apply(derive_label)
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["image", "label"])
    y = df["label"]
    return x, y


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["intensity_range"] = out["max"] - out["min"]
    out["upper_lower_ratio"] = (out["p90"] + EPS) / (out["p10"] + EPS)
    out["contrast_homogeneity_ratio"] = (out["glcm_contrast"] + EPS) / (
        out["glcm_homogeneity"] + EPS
    )
    out["edge_to_fft_ratio"] = (out["edge_strength"] + EPS) / (out["fft_mean"] + EPS)
    out["texture_entropy_interaction"] = out["entropy"] * out["glcm_contrast"]
    out["lbp_uniformity_proxy"] = out["lbp_8"] + out["lbp_9"]
    return out


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray | None) -> float:
    if scores is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def get_score_signal(model: Pipeline, x: pd.DataFrame) -> np.ndarray | None:
    estimator = model
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x)
    return None


def evaluate_model(
    model: Pipeline,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    split_name: str,
) -> Dict[str, float]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    y_score = get_score_signal(model, x_eval)
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall": recall_score(y_eval, y_pred, zero_division=0),
        "f1": f1_score(y_eval, y_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_eval, y_score),
    }


def build_models() -> Dict[str, Pipeline]:
    return {
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
            ]
        ),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=9))]),
        "RandomForest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]
        ),
    }


def run_eda(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df), "features": train_df.shape[1] - 2},
            {"split": "val", "rows": len(val_df), "features": val_df.shape[1] - 2},
            {"split": "test", "rows": len(test_df), "features": test_df.shape[1] - 2},
        ]
    )
    split_summary.to_csv(TABLES_DIR / "split_summary.csv", index=False)

    class_dist = pd.DataFrame(
        [
            {"split": "train", "class": "normal", "count": int((train_df["label"] == 0).sum())},
            {"split": "train", "class": "pneumonia", "count": int((train_df["label"] == 1).sum())},
            {"split": "val", "class": "normal", "count": int((val_df["label"] == 0).sum())},
            {"split": "val", "class": "pneumonia", "count": int((val_df["label"] == 1).sum())},
            {"split": "test", "class": "normal", "count": int((test_df["label"] == 0).sum())},
            {"split": "test", "class": "pneumonia", "count": int((test_df["label"] == 1).sum())},
        ]
    )
    class_dist.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=class_dist, x="split", y="count", hue="class")
    plt.title("Class distribution across splits")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "class_distribution.png", dpi=180)
    plt.close()

    feature_df = train_df.drop(columns=["image", "label"])
    missing_summary = feature_df.isna().sum().sort_values(ascending=False).reset_index()
    missing_summary.columns = ["feature", "missing_count"]
    missing_summary.to_csv(TABLES_DIR / "missing_values_train.csv", index=False)

    corr_subset = feature_df.corr().abs()
    top_corr = (
        corr_subset.where(np.triu(np.ones(corr_subset.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_corr.columns = ["feature_1", "feature_2", "abs_corr"]
    top_corr.head(50).to_csv(TABLES_DIR / "top_feature_correlations.csv", index=False)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, cmap="mako", center=0.5)
    plt.title("Train feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "feature_correlation_heatmap.png", dpi=180)
    plt.close()


def analyze_feature_groups(train_df: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        for feature in columns:
            normal_vals = train_df.loc[train_df["label"] == 0, feature]
            pneumonia_vals = train_df.loc[train_df["label"] == 1, feature]
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "normal_mean": float(normal_vals.mean()),
                    "pneumonia_mean": float(pneumonia_vals.mean()),
                    "mean_difference_abs": float(abs(pneumonia_vals.mean() - normal_vals.mean())),
                    "normal_std": float(normal_vals.std()),
                    "pneumonia_std": float(pneumonia_vals.std()),
                }
            )
    group_df = pd.DataFrame(rows).sort_values("mean_difference_abs", ascending=False)
    group_df.to_csv(TABLES_DIR / "feature_group_statistics.csv", index=False)

    top_features = group_df.groupby("group").head(2)["feature"].tolist()
    plot_df = train_df[["label"] + top_features].melt(id_vars=["label"], var_name="feature", value_name="value")
    plot_df["class"] = plot_df["label"].map({0: "normal", 1: "pneumonia"})
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=plot_df, x="feature", y="value", hue="class", showfliers=False)
    plt.title("Representative feature-group separability (train)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "feature_group_boxplots.png", dpi=180)
    plt.close()


def evaluate_model_suite(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_results = []
    test_results = []
    for name, model in build_models().items():
        val_results.append(
            evaluate_model(model, name, x_train, y_train, x_val, y_val, "val")
        )
        test_results.append(
            evaluate_model(model, name, x_train, y_train, x_test, y_test, "test")
        )

    val_df = pd.DataFrame(val_results).sort_values("f1", ascending=False)
    test_df = pd.DataFrame(test_results).sort_values("f1", ascending=False)
    val_df.to_csv(TABLES_DIR / "baseline_model_results_val.csv", index=False)
    test_df.to_csv(TABLES_DIR / "baseline_model_results_test.csv", index=False)
    return val_df, test_df


def evaluate_feature_group_models(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
) -> None:
    rows = []
    for group_name, cols in FEATURE_GROUPS.items():
        model = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        rows.append(
            evaluate_model(
                model,
                f"RandomForest_{group_name}",
                train_x[cols],
                train_y,
                val_x[cols],
                val_y,
                "val",
            )
        )
        rows.append(
            evaluate_model(
                model,
                f"RandomForest_{group_name}",
                train_x[cols],
                train_y,
                test_x[cols],
                test_y,
                "test",
            )
        )
    pd.DataFrame(rows).sort_values(["split", "f1"], ascending=[True, False]).to_csv(
        TABLES_DIR / "feature_group_model_results.csv", index=False
    )


def run_feature_selection_and_pca(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    eng_train = add_engineered_features(x_train)
    eng_val = add_engineered_features(x_val)
    eng_test = add_engineered_features(x_test)

    k = min(20, eng_train.shape[1])
    selection_model = Pipeline(
        [
            ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=800,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pca_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=800,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    results = [
        evaluate_model(
            selection_model,
            "FeatureSelection_MI+RandomForest",
            eng_train,
            y_train,
            eng_val,
            y_val,
            "val",
        ),
        evaluate_model(
            selection_model,
            "FeatureSelection_MI+RandomForest",
            eng_train,
            y_train,
            eng_test,
            y_test,
            "test",
        ),
        evaluate_model(
            pca_model,
            "PCA95+RandomForest",
            eng_train,
            y_train,
            eng_val,
            y_val,
            "val",
        ),
        evaluate_model(
            pca_model,
            "PCA95+RandomForest",
            eng_train,
            y_train,
            eng_test,
            y_test,
            "test",
        ),
    ]

    selection_model.fit(eng_train, y_train)
    selector = selection_model.named_steps["select"]
    selected_features = eng_train.columns[selector.get_support()].tolist()
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        TABLES_DIR / "selected_features_mutual_info.csv", index=False
    )

    pca_model.fit(eng_train, y_train)
    pca_step = pca_model.named_steps["pca"]
    explained = np.cumsum(pca_step.explained_variance_ratio_)
    pca_df = pd.DataFrame(
        {"component_index": np.arange(1, len(explained) + 1), "cumulative_explained_variance": explained}
    )
    pca_df.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(pca_df["component_index"], pca_df["cumulative_explained_variance"], marker="o")
    plt.axhline(0.95, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance curve")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "pca_explained_variance_curve.png", dpi=180)
    plt.close()

    out = pd.DataFrame(results)
    out.to_csv(TABLES_DIR / "feature_engineering_selection_pca_results.csv", index=False)
    return out


def save_best_model_diagnostics(
    val_results: pd.DataFrame,
    fs_pca_results: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    val_candidates = pd.concat(
        [val_results[["model", "f1"]], fs_pca_results[fs_pca_results["split"] == "val"][["model", "f1"]]],
        ignore_index=True,
    ).sort_values("f1", ascending=False)
    best_model_name = val_candidates.iloc[0]["model"]

    if best_model_name in build_models():
        best_model = build_models()[best_model_name]
        train_x = x_train
        test_x = x_test
    elif best_model_name == "FeatureSelection_MI+RandomForest":
        train_x = add_engineered_features(x_train)
        test_x = add_engineered_features(x_test)
        k = min(20, train_x.shape[1])
        best_model = Pipeline(
            [
                ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=800,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        train_x = add_engineered_features(x_train)
        test_x = add_engineered_features(x_test)
        best_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=800,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    best_model.fit(train_x, y_train)
    y_pred = best_model.predict(test_x)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["normal", "pneumonia"], output_dict=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Best: {best_model_name})")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "best_model_confusion_matrix.png", dpi=180)
    plt.close()

    with open(TABLES_DIR / "best_model_test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(TABLES_DIR / "best_model_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_model_by_val_f1": best_model_name}, f, indent=2)

    return {
        "best_model_name": best_model_name,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def print_final_outcomes(
    test_results: pd.DataFrame,
    fs_pca_results: pd.DataFrame,
    diagnostics: Dict[str, object],
) -> None:
    baseline_test_best = test_results.sort_values("f1", ascending=False).iloc[0]
    fs_pca_test = fs_pca_results[fs_pca_results["split"] == "test"].sort_values("f1", ascending=False)
    fs_pca_best = fs_pca_test.iloc[0]

    report = diagnostics["classification_report"]
    cm = np.array(diagnostics["confusion_matrix"])

    print("\n" + "=" * 72)
    print("FINAL OUTCOMES (Objective 5 & 6)")
    print("=" * 72)

    print("\n[Objective 5] Feature Engineering, Selection, and Dimensionality Reduction")
    print(
        f"- Best baseline (test): {baseline_test_best['model']} | "
        f"F1={baseline_test_best['f1']:.4f}, Accuracy={baseline_test_best['accuracy']:.4f}"
    )
    print(
        f"- Best from FE/FS/PCA (test): {fs_pca_best['model']} | "
        f"F1={fs_pca_best['f1']:.4f}, Accuracy={fs_pca_best['accuracy']:.4f}"
    )
    f1_gain = fs_pca_best["f1"] - baseline_test_best["f1"]
    print(f"- F1 change vs baseline best: {f1_gain:+.4f}")

    print("\n[Objective 6] Model Evaluation Metrics (Best model chosen by validation F1)")
    print(f"- Selected best model: {diagnostics['best_model_name']}")
    print(
        "- Test metrics: "
        f"Accuracy={report['accuracy']:.4f}, "
        f"Precision={report['weighted avg']['precision']:.4f}, "
        f"Recall={report['weighted avg']['recall']:.4f}, "
        f"F1={report['weighted avg']['f1-score']:.4f}"
    )
    print("- Confusion Matrix [rows=actual(normal,pneumonia), cols=predicted(normal,pneumonia)]:")
    print(cm)
    print("=" * 72 + "\n")


def main() -> None:
    ensure_dirs()

    train_df = load_split("train_features.csv")
    val_df = load_split("val_features.csv")
    test_df = load_split("test_features.csv")

    run_eda(train_df, val_df, test_df)
    analyze_feature_groups(train_df)

    x_train, y_train = split_xy(train_df)
    x_val, y_val = split_xy(val_df)
    x_test, y_test = split_xy(test_df)

    val_results, test_results = evaluate_model_suite(
        x_train, y_train, x_val, y_val, x_test, y_test
    )
    evaluate_feature_group_models(x_train, y_train, x_val, y_val, x_test, y_test)
    fs_pca_results = run_feature_selection_and_pca(
        x_train, y_train, x_val, y_val, x_test, y_test
    )
    diagnostics = save_best_model_diagnostics(
        val_results, fs_pca_results, x_train, y_train, x_test, y_test
    )
    print_final_outcomes(test_results, fs_pca_results, diagnostics)

    print("Experiment completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
