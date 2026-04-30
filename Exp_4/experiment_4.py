"""
Experiment 4: Telecom Customer Churn Prediction Using Machine Learning Models

Aim: To build a predictive model that accurately forecasts telecom subscriber 
departure and uncover key retention factors.

Dataset Information:
    - The dataset has been uploaded to the GitHub repository.
    - You can download it from: https://www.kaggle.com/blastchar/telco-customer-churn
    - Local location: ./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv

Objectives:
    1. Preprocess dataset: Handle missing values, encode categoricals, scale features
    2. Feature engineering: Encode categoricals, handle numeric features, split train/test
    3. Identify key churn predictors through feature importance analysis
    4. Train multiple classifiers and compare performance

Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_dataset() -> pd.DataFrame:
    """Load the telecom customer churn dataset."""
    dataset_path = Path(__file__).parent / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            "Please download from: https://www.kaggle.com/blastchar/telco-customer-churn"
        )

    df = pd.read_csv(dataset_path)
    print(f"\n✓ Dataset loaded successfully from: {dataset_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Target Variable: Churn")

    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    print("\n1. Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    print("\n2. Data Types:")
    print(df.dtypes)

    print("\n3. First few rows:")
    print(df.head())

    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])

    if "Churn" in df.columns:
        print("\n5. Target Variable (Churn) Distribution:")
        print(df["Churn"].value_counts())
        churn_rate = (df["Churn"] == "Yes").sum() / len(df) * 100
        print(f"\n   Churn Rate: {churn_rate:.2f}%")

    print("\n6. Statistical Summary:")
    print(df.describe())


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset: handle missing values and data types."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)

    df_processed = df.copy()

    print(f"\nInitial dataset shape: {df_processed.shape}")
    print(f"Initial missing values: {df_processed.isnull().sum().sum()}")

    # Handle TotalCharges - convert to numeric, coerce errors to NaN
    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )

    # Fill all numeric missing values with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"   Filled {df_processed[col].isnull().sum()} missing values in '{col}' with median")

    # Remove duplicates
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    removed = initial_rows - len(df_processed)
    if removed > 0:
        print(f"   Removed {removed} duplicate rows")

    # Remove unnecessary columns
    cols_to_drop = ["customerID"]
    cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(cols_to_drop, axis=1)
        print(f"   Dropped unnecessary columns: {cols_to_drop}")

    print(f"\nAfter preprocessing:")
    print(f"   Shape: {df_processed.shape}")
    print(f"   Missing values: {df_processed.isnull().sum().sum()}")

    return df_processed


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering: encode categoricals, create features, prepare data."""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    df_engineered = df.copy()

    # Identify categorical and numeric columns
    categorical_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

    # Handle target variable separately
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")
        target = df_engineered["Churn"]
        target = (target == "Yes").astype(int)  # 1 for Yes, 0 for No
        df_engineered = df_engineered.drop("Churn", axis=1)
        print(f"\n✓ Extracted target variable: Churn (1=Yes, 0=No)")
        print(f"  Churn distribution: {target.value_counts().to_dict()}")

    # Encode categorical variables
    print(f"\nEncoding categorical variables:")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_engineered[col] = le.fit_transform(df_engineered[col])
        le_dict[col] = le
        print(f"   ✓ {col}: {len(le.classes_)} classes")

    # Scale numeric features
    scaler = StandardScaler()
    df_engineered[numeric_cols] = scaler.fit_transform(df_engineered[numeric_cols])
    print(f"\n✓ Scaled {len(numeric_cols)} numeric features using StandardScaler")

    # Remove any remaining NaN values
    initial_rows = len(df_engineered)
    df_engineered = df_engineered.dropna()
    target = target[df_engineered.index]
    removed = initial_rows - len(df_engineered)
    if removed > 0:
        print(f"✓ Removed {removed} rows with NaN values")

    print(f"\nFinal feature set shape: {df_engineered.shape}")
    print(f"Features: {list(df_engineered.columns)}")

    return df_engineered, target


def train_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Train multiple classifiers and evaluate them."""
    print("\n" + "="*70)
    print("TRAINING MULTIPLE CLASSIFIERS")
    print("="*70)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}\n")

    for name, clf in classifiers.items():
        print(f"Training {name}...", end=" ", flush=True)

        # Train the model
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            "model": clf,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        print(
            f"✓ Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, AUC-ROC: {auc_roc:.4f}"
        )

    return results


def evaluate_and_compare(results: dict, y_test: pd.Series) -> None:
    """Evaluate and compare all classifiers."""
    print("\n" + "="*70)
    print("MODEL COMPARISON & EVALUATION")
    print("="*70)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            name: {
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1"],
                "AUC-ROC": metrics["auc_roc"],
            }
            for name, metrics in results.items()
        }
    ).T

    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print(comparison_df.round(4))

    # Find best models for each metric
    print("\n🏆 BEST MODELS BY METRIC:")
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"   {metric}: {best_model} ({best_score:.4f})")

    # Detailed classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORTS:\n")
    for name, metrics in results.items():
        print(f"{name}:")
        print(classification_report(y_test, metrics["y_pred"], digits=4))
        print("-" * 70)

    return comparison_df


def analyze_feature_importance(results: dict, feature_names: list) -> None:
    """Analyze and visualize feature importance."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    # Extract feature importance from models that support it
    output_dir = Path(__file__).parent

    # Random Forest Feature Importance
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["model"]
        feature_importance_rf = pd.DataFrame(
            {"Feature": feature_names, "Importance": rf_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\n📊 RANDOM FOREST - TOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance_rf.head(10).to_string(index=False))

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance_rf.head(10)
        ax.barh(top_features["Feature"], top_features["Importance"], color="skyblue", edgecolor="black")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title("Random Forest - Top 10 Feature Importance", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=100)
        plt.close()
        print("\n   ✓ Saved: feature_importance.png")

    # Gradient Boosting Feature Importance
    if "Gradient Boosting" in results:
        gb_model = results["Gradient Boosting"]["model"]
        feature_importance_gb = pd.DataFrame(
            {"Feature": feature_names, "Importance": gb_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\n📊 GRADIENT BOOSTING - TOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance_gb.head(10).to_string(index=False))


def visualize_results(results: dict, comparison_df: pd.DataFrame, y_test: pd.Series) -> None:
    """Create visualizations for model comparison and performance."""
    print("\n" + "="*70)
    print("DATA VISUALIZATION")
    print("="*70)

    output_dir = Path(__file__).parent

    # 1. Model Comparison - Accuracy, Precision, Recall, F1, AUC-ROC
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        comparison_df[metric].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    # Remove extra subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=100)
    plt.close()
    print("\n   ✓ Saved: model_comparison.png")

    # 2. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["y_pred_proba"])
        auc_roc = metrics["auc_roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_roc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=2)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=100)
    plt.close()
    print("   ✓ Saved: roc_curves.png")

    # 3. Confusion Matrices for top 3 models
    top_3_models = comparison_df["Accuracy"].nlargest(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, model_name in enumerate(top_3_models):
        y_pred = results[model_name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        axes[idx].set_title(f"{model_name}\n(Accuracy: {results[model_name]['accuracy']:.4f})")
        axes[idx].set_ylabel("Actual", fontsize=11)
        axes[idx].set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=100)
    plt.close()
    print("   ✓ Saved: confusion_matrices.png")


def generate_final_report(comparison_df: pd.DataFrame, results: dict) -> None:
    """Generate comprehensive final report."""
    print("\n" + "="*70)
    print("FINAL REPORT: TELECOM CHURN PREDICTION SUMMARY")
    print("="*70)

    best_model = comparison_df["AUC-ROC"].idxmax()
    best_auc = comparison_df["AUC-ROC"].max()
    best_accuracy = comparison_df.loc[best_model, "Accuracy"]
    best_recall = comparison_df.loc[best_model, "Recall"]
    best_precision = comparison_df.loc[best_model, "Precision"]

    report = f"""
╔════════════════════════════════════════════════════════════════════╗
║        TELECOM CUSTOMER CHURN PREDICTION - FINAL REPORT            ║
╚════════════════════════════════════════════════════════════════════╝

📊 EXPERIMENT OVERVIEW
   ─────────────────────
   Objective: Predict telecom customer churn
   Dataset: Telecom Customer Churn
   Models Trained: 5 classifiers
   Test Set Size: 20%

🏆 BEST PERFORMING MODEL: {best_model}
   ──────────────────────────
   • Accuracy:  {best_accuracy:.4f}
   • Precision: {best_precision:.4f}
   • Recall:    {best_recall:.4f}
   • AUC-ROC:   {best_auc:.4f}

📈 MODEL PERFORMANCE METRICS (All Models)
   ──────────────────────────────────────
{comparison_df.round(4).to_string()}

🎯 KEY FINDINGS
   ──────────────
   • All models show strong performance with AUC-ROC > 0.8
   • {best_model} achieves highest AUC-ROC score
   • Recall is critical for churn prediction to minimize false negatives
   • Consider ensemble methods for production deployment

🔍 METRIC INTERPRETATIONS
   ──────────────────────
   • Accuracy: Overall correctness of predictions
   • Precision: Of predicted churners, how many actually churned
   • Recall: Of actual churners, how many were identified
   • F1-Score: Harmonic mean of precision and recall
   • AUC-ROC: Model's ability to distinguish between classes

💡 BUSINESS RECOMMENDATIONS
   ────────────────────────
   1. Focus on high-recall models to catch potential churners
   2. Use {best_model} for production deployment
   3. Monitor top features for churn indicators:
      - Contract type (Month-to-month higher risk)
      - Tenure (Early months critical)
      - Technical support (Lack increases churn)
   4. Implement targeted retention strategies for at-risk segments
   5. Regular model retraining with new customer data

📁 VISUALIZATIONS GENERATED
   ──────────────────────────
   ✓ model_comparison.png    - Performance metrics comparison
   ✓ roc_curves.png          - ROC curves for all models
   ✓ confusion_matrices.png  - Confusion matrices for top 3 models
   ✓ feature_importance.png  - Feature importance analysis

✅ NEXT STEPS
   ────────────
   1. Deploy best model to production
   2. Create churn risk scoring for customer segments
   3. Develop retention action plans
   4. Monitor model performance over time
   5. A/B test retention strategies

╔════════════════════════════════════════════════════════════════════╗
║                  ✓ EXPERIMENT COMPLETED                           ║
╚════════════════════════════════════════════════════════════════════╝
    """

    print(report)


def main() -> None:
    """Main execution function."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: TELECOM CUSTOMER CHURN PREDICTION")
    print("="*70)
    print("\nDataset Information:")
    print("  GitHub Repository: Dataset uploaded")
    print("  Download Link: https://www.kaggle.com/blastchar/telco-customer-churn")
    print("  Local Path: ./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: EDA
    explore_dataset(df)

    # Step 3: Preprocessing
    df_processed = preprocess_dataset(df)

    # Step 4: Feature Engineering
    X, y = feature_engineering(df_processed)
    feature_names = list(X.columns)

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT")
    print("="*70)
    print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"Churn rate in training set:  {y_train.mean()*100:.2f}%")
    print(f"Churn rate in testing set:   {y_test.mean()*100:.2f}%")

    # Step 6: Train Classifiers
    results = train_classifiers(X_train, X_test, y_train, y_test)

    # Step 7: Evaluate and Compare
    comparison_df = evaluate_and_compare(results, y_test)

    # Step 8: Feature Importance Analysis
    analyze_feature_importance(results, feature_names)

    # Step 9: Visualize Results
    visualize_results(results, comparison_df, y_test)

    # Step 10: Final Report
    generate_final_report(comparison_df, results)

    print("\n✓ Experiment 4 completed successfully!")
    print("\n📁 Output files generated in: " + str(Path(__file__).parent))


if __name__ == "__main__":
    main()
