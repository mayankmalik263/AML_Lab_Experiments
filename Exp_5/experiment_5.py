"""
Experiment 5: Stock Prices Prediction using Regression Models

Aim: To develop and evaluate regression-based machine learning models for predicting
stock prices using historical market data.

Dataset Information:
    - We have uploaded the dataset on the GitHub repository.
    - You can download it from here: https://www.kaggle.com/datasets/dgawlik/tesla-stock-price
    - I have downloaded the dataset, which is in the folder 'dataset' within 'Exp_5'.

Objectives:
    1. Understand and analyze historical stock market data
    2. Preprocess data and perform feature engineering for regression modeling
    3. Implement regression models for stock price prediction
    4. Evaluate model performance using MSE, MAE, and R²
    5. Compare predicted values with actual stock prices for analysis

Technologies Used:
    Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_dataset() -> pd.DataFrame:
    """Load the stock prices dataset."""
    dataset_path = Path(__file__).parent / "dataset" / "Tesla.csv - Tesla.csv.csv"

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            "Please download it from: https://www.kaggle.com/datasets/dgawlik/tesla-stock-price"
        )

    df = pd.read_csv(dataset_path)
    print(f"\n✓ Dataset loaded successfully from: {dataset_path}")
    print(f"  Shape: {df.shape}")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform basic exploratory data analysis."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print("\n1. Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Data Types:\n{df.dtypes}")

    print("\n2. First few rows:")
    print(df.head())

    print("\n3. Statistical Summary:")
    print(df.describe(include="all"))

    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])

    output_dir = Path(__file__).parent

    # Price trend over time
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], color="darkred", linewidth=1)
    plt.title("Tesla Close Price Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Observation Index")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.savefig(output_dir / "price_trend.png", dpi=100)
    plt.close()

    # Correlation heatmap
    numeric_df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=100)
    plt.close()

    print("\n   Visualizations saved: price_trend.png, correlation_heatmap.png")


def preprocess_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and create time-series features."""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    df_processed = df.copy()

    # Parse dates and sort chronologically
    df_processed["Date"] = pd.to_datetime(df_processed["Date"])
    df_processed = df_processed.sort_values("Date").reset_index(drop=True)

    # Convert numeric columns if needed
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    for column in numeric_columns:
        df_processed[column] = pd.to_numeric(df_processed[column], errors="coerce")

    print(f"\nInitial missing values: {df_processed.isnull().sum().sum()}")

    # Fill any missing values using forward fill then backward fill
    df_processed[numeric_columns] = df_processed[numeric_columns].ffill().bfill()

    # Date-based features
    df_processed["Year"] = df_processed["Date"].dt.year
    df_processed["Month"] = df_processed["Date"].dt.month
    df_processed["Day"] = df_processed["Date"].dt.day
    df_processed["DayOfWeek"] = df_processed["Date"].dt.dayofweek
    df_processed["WeekOfYear"] = df_processed["Date"].dt.isocalendar().week.astype(int)

    # Market-derived features
    df_processed["High_Low_Spread"] = df_processed["High"] - df_processed["Low"]
    df_processed["Open_Close_Spread"] = df_processed["Close"] - df_processed["Open"]
    df_processed["Daily_Return"] = df_processed["Close"].pct_change().fillna(0)

    # Forecast the next trading day close price
    df_processed["Target_Close"] = df_processed["Close"].shift(-1)

    # Lag/rolling features based on historical close price
    df_processed["Prev_Close"] = df_processed["Close"].shift(1)
    df_processed["MA_5"] = df_processed["Close"].shift(1).rolling(window=5).mean()
    df_processed["MA_10"] = df_processed["Close"].shift(1).rolling(window=10).mean()
    df_processed["Volatility_5"] = df_processed["Close"].shift(1).rolling(window=5).std()

    # Drop rows with NaN values created by lag/rolling features
    before_drop = len(df_processed)
    df_processed = df_processed.dropna().reset_index(drop=True)
    dropped_rows = before_drop - len(df_processed)

    print(f"   Dropped {dropped_rows} rows created by rolling features")
    print(f"   Final shape after preprocessing: {df_processed.shape}")

    return df_processed


def perform_feature_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze feature correlations with the target price."""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    feature_columns = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear",
        "High_Low_Spread",
        "Open_Close_Spread",
        "Daily_Return",
        "Prev_Close",
        "MA_5",
        "MA_10",
        "Volatility_5",
    ]

    target = "Target_Close"
    corr_matrix = df[feature_columns + [target]].corr()[target].sort_values(ascending=False)

    print("\nCorrelation with Next-Day Close price:")
    print(corr_matrix)

    selected_features = corr_matrix.drop(target).index.tolist()
    print(f"\nSelected features ({len(selected_features)}): {selected_features}")

    feature_importance_df = corr_matrix.drop(target).abs().sort_values(ascending=False).reset_index()
    feature_importance_df.columns = ["Feature", "Importance"]

    output_dir = Path(__file__).parent
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(10), y="Feature", x="Importance", color="steelblue")
    plt.title("Top Feature Correlations with Close Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_importance.png", dpi=100)
    plt.close()

    print("\n   Visualization saved: feature_correlation_importance.png")
    return feature_importance_df


def build_and_evaluate_models(df: pd.DataFrame, selected_features: list[str]) -> dict:
    """Train multiple regression models and evaluate them."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    X = df[selected_features]
    y = df["Target_Close"]

    # Time-based split to respect stock market chronology
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"\nTime-based split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000)),
        ]),
        "Support Vector Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
        ]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
    }

    results: dict[str, dict] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        print(f"   MAE:  {mae:.4f}")
        print(f"   MSE:  {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")

    return results


def visualize_results(results: dict, feature_importance_df: pd.DataFrame) -> None:
    """Visualize predicted vs actual values and model comparison."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    output_dir = Path(__file__).parent

    comparison_df = pd.DataFrame(
        {
            name: {
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
            }
            for name, metrics in results.items()
        }
    ).T

    # Comparison of model metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["MAE", "MSE", "RMSE", "R2"]
    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 2, idx % 2]
        comparison_df[metric].plot(kind="bar", ax=ax, color="teal", edgecolor="black")
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "model_metrics_comparison.png", dpi=100)
    plt.close()
    print("\n   ✓ Saved: model_metrics_comparison.png")

    # Actual vs predicted for best model
    best_model_name = comparison_df["R2"].idxmax()
    best_results = results[best_model_name]
    y_test = best_results["y_test"]
    y_pred = best_results["y_pred"]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="navy", edgecolors="k")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.title(f"Actual vs Predicted Next-Day Close Prices ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Next-Day Close Price")
    plt.ylabel("Predicted Next-Day Close Price")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=100)
    plt.close()
    print("   ✓ Saved: actual_vs_predicted.png")

    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, color="darkorange", edgecolors="k")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Close Price")
    axes[0].set_ylabel("Residuals")
    axes[0].grid(True, alpha=0.3)

    sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "residual_analysis.png", dpi=100)
    plt.close()
    print("   ✓ Saved: residual_analysis.png")

    # Feature importance bar chart
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(10)
    sns.barplot(data=top_features, y="Feature", x="Importance", color="royalblue")
    plt.title("Top Feature Relationships with Close Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "top_feature_importance.png", dpi=100)
    plt.close()
    print("   ✓ Saved: top_feature_importance.png")

    return comparison_df, best_model_name


def generate_final_report(comparison_df: pd.DataFrame, best_model_name: str) -> None:
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 70)
    print("FINAL REPORT: STOCK PRICE PREDICTION SUMMARY")
    print("=" * 70)

    best_r2 = comparison_df.loc[best_model_name, "R2"]
    best_mae = comparison_df.loc[best_model_name, "MAE"]
    best_mse = comparison_df.loc[best_model_name, "MSE"]
    best_rmse = comparison_df.loc[best_model_name, "RMSE"]

    report = f"""
╔════════════════════════════════════════════════════════════════════╗
║         STOCK PRICE PREDICTION USING REGRESSION MODELS            ║
╚════════════════════════════════════════════════════════════════════╝

📊 EXPERIMENT OVERVIEW
   ─────────────────────
    Objective: Predict Tesla next-day close prices from historical data
   Dataset: Tesla stock price dataset
   Models Trained: 5 regression models
   Split Strategy: Time-based 80-20 split

🏆 BEST PERFORMING MODEL: {best_model_name}
   ──────────────────────────
   • MAE:  {best_mae:.4f}
   • MSE:  {best_mse:.4f}
   • RMSE: {best_rmse:.4f}
   • R²:   {best_r2:.4f}

📈 MODEL COMPARISON
   ─────────────────
{comparison_df.round(4).to_string()}

🔍 KEY FINDINGS
   ─────────────
    • Historical price features strongly influence the next-day close price
    • Lagged and rolling features help capture market momentum
   • Time-aware splitting avoids leakage and better reflects real forecasting
   • Linear and regularized models remain competitive for this dataset

💡 INTERPRETATION
   ───────────────
   • Lower MAE/MSE indicates better prediction accuracy
   • Higher R² indicates better variance explained by the model
   • Actual vs predicted plots show how closely predictions follow market movement

📁 VISUALIZATIONS GENERATED
   ──────────────────────────
   ✓ price_trend.png
   ✓ correlation_heatmap.png
   ✓ feature_correlation_importance.png
   ✓ model_metrics_comparison.png
   ✓ actual_vs_predicted.png
   ✓ residual_analysis.png
   ✓ top_feature_importance.png

✅ NEXT STEPS
   ───────────
   1. Try additional lag features or technical indicators
   2. Compare with LSTM or other sequence models
   3. Test the model on newer stock data
   4. Tune hyperparameters for SVR and Random Forest

╔════════════════════════════════════════════════════════════════════╗
║                  ✓ EXPERIMENT COMPLETED                           ║
╚════════════════════════════════════════════════════════════════════╝
    """

    print(report)


def main() -> None:
    """Main execution function."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: STOCK PRICE PREDICTION USING REGRESSION MODELS")
    print("=" * 70)
    print("\nDataset Information:")
    print("  We have uploaded the dataset on the GitHub repository.")
    print("  You can download it from here: https://www.kaggle.com/datasets/dgawlik/tesla-stock-price")
    print("  I have downloaded the dataset, which is in the folder 'dataset' within the folder 'Exp_5'.")

    df = load_dataset()
    explore_dataset(df)

    df_processed = preprocess_and_engineer_features(df)
    feature_importance_df = perform_feature_analysis(df_processed)

    selected_features = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear",
        "High_Low_Spread",
        "Open_Close_Spread",
        "Daily_Return",
        "Prev_Close",
        "MA_5",
        "MA_10",
        "Volatility_5",
    ]

    results = build_and_evaluate_models(df_processed, selected_features)
    comparison_df, best_model_name = visualize_results(results, feature_importance_df)
    generate_final_report(comparison_df, best_model_name)

    print("\n✓ Experiment 5 completed successfully!")
    print(f"\n📁 Output files generated in: {Path(__file__).parent}")


if __name__ == "__main__":
    main()
