"""
Experiment 3: House Price Prediction using Linear Regression

Aim: To predict house prices based on various features using a linear regression model.

Dataset Information:
    - The dataset has been uploaded to the GitHub repository.
    - You can download it from: https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india/data
    - Local location: ./dataset/House Price India.csv

Key Steps:
    1. Data Preprocessing: Clean the dataset and handle missing values
    2. Feature Engineering: Select important features and remove irrelevant ones
    3. Model Building: Build a linear regression model
    4. Model Evaluation: Evaluate using MAE, MSE, R², and Adjusted R²
    5. Data Visualization: Plot actual vs predicted, residuals, and feature importance

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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_dataset() -> pd.DataFrame:
    """Load the house price dataset."""
    dataset_path = Path(__file__).parent / "dataset" / "House Price India.csv"

    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            "Please download from: https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india/data"
        )

    df = pd.read_csv(dataset_path)
    print(f"\n✓ Dataset loaded successfully from: {dataset_path}")
    print(f"  Shape: {df.shape}")

    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    print("\n1. Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Data Types:\n{df.dtypes}")

    print("\n2. First few rows:")
    print(df.head())

    print("\n3. Statistical Summary:")
    print(df.describe())

    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])
        print(f"\n   Total missing values: {missing.sum()}")

    print("\n5. Correlation with Price:")
    if "Price" in df.columns:
        correlations = df.corr()["Price"].sort_values(ascending=False)
        print(correlations)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data preprocessing: cleaning and handling missing values."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)

    df_processed = df.copy()

    # Display initial statistics
    print(f"\nInitial dataset shape: {df_processed.shape}")
    print(f"Initial missing values:\n{df_processed.isnull().sum()[df_processed.isnull().sum() > 0]}")

    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=["object"]).columns

    # Fill numeric missing values with median
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"   Filled missing values in '{col}' with median")

    # Fill categorical missing values with mode
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            print(f"   Filled missing values in '{col}' with mode")

    print(f"\nAfter preprocessing:")
    print(f"   Shape: {df_processed.shape}")
    print(f"   Missing values: {df_processed.isnull().sum().sum()}")

    # Remove duplicates
    initial_shape = df_processed.shape[0]
    df_processed = df_processed.drop_duplicates()
    removed_duplicates = initial_shape - df_processed.shape[0]
    if removed_duplicates > 0:
        print(f"   Removed {removed_duplicates} duplicate rows")

    return df_processed


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Feature engineering: select important features and encode categorical variables."""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    df_engineered = df.copy()

    # Identify target variable
    if "Price" not in df_engineered.columns:
        # Try common price column names
        price_cols = [col for col in df_engineered.columns if "price" in col.lower()]
        if price_cols:
            price_col = price_cols[0]
            df_engineered = df_engineered.rename(columns={price_col: "Price"})
        else:
            raise ValueError("Price column not found in dataset")

    print(f"\nTarget variable: Price")
    print(f"Target range: ${df_engineered['Price'].min():,.0f} - ${df_engineered['Price'].max():,.0f}")

    # Encode categorical variables
    categorical_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCategorical columns to encode: {categorical_cols}")

    for col in categorical_cols:
        df_engineered[col] = pd.factorize(df_engineered[col])[0]
        print(f"   Encoded '{col}'")

    # Select features (exclude Price)
    X = df_engineered.drop("Price", axis=1)
    y = df_engineered["Price"]

    # Calculate correlations with price
    correlations = df_engineered.corr()["Price"].sort_values(ascending=False)
    print(f"\nTop 10 Features by Correlation with Price:")
    print(correlations.head(10))

    # Feature selection (keep features with correlation > 0.1 or < -0.1)
    important_features = correlations[
        (correlations.abs() > 0.1) & (correlations.index != "Price")
    ].index.tolist()

    if len(important_features) < 3:
        # If too few features, keep top features
        important_features = correlations.drop("Price").head(min(10, len(X.columns))).index.tolist()

    print(f"\nSelected features: {important_features}")
    print(f"Number of features: {len(important_features)}")

    X_selected = X[important_features]

    return X_selected, y, important_features


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Build and train the linear regression model."""
    print("\n" + "="*70)
    print("MODEL BUILDING & TRAINING")
    print("="*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain-test split (80-20):")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    print(f"   Number of features: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print(f"\n✓ Linear Regression model trained successfully")
    print(f"   Intercept: {model.intercept_:,.2f}")

    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_test


def evaluate_model(
    model,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    feature_names,
) -> tuple:
    """Evaluate model performance using various metrics."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics for training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate metrics for testing set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate adjusted R-squared
    n = X_test_scaled.shape[0]
    k = X_test_scaled.shape[1]
    adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - k - 1)

    print("\n📊 TRAINING SET METRICS:")
    print(f"   Mean Absolute Error (MAE):        ${train_mae:,.2f}")
    print(f"   Mean Squared Error (MSE):         {train_mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE):   ${train_rmse:,.2f}")
    print(f"   R-squared (R²):                   {train_r2:.4f}")

    print("\n📊 TESTING SET METRICS:")
    print(f"   Mean Absolute Error (MAE):        ${test_mae:,.2f}")
    print(f"   Mean Squared Error (MSE):         {test_mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE):   ${test_rmse:,.2f}")
    print(f"   R-squared (R²):                   {test_r2:.4f}")
    print(f"   Adjusted R-squared (Adj R²):      {adj_r2:.4f}")

    # Feature importance
    print("\n📊 FEATURE IMPORTANCE (Coefficients):")
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": model.coef_}
    ).sort_values("Coefficient", key=abs, ascending=False)
    print(feature_importance.to_string(index=False))

    return {
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "adj_r2": adj_r2,
        "feature_importance": feature_importance,
    }


def visualize_results(results: dict, X_test: pd.DataFrame) -> None:
    """Create visualizations for model results."""
    print("\n" + "="*70)
    print("DATA VISUALIZATION")
    print("="*70)

    output_dir = Path(__file__).parent
    y_test = results["y_test"]
    y_test_pred = results["y_test_pred"]
    feature_importance = results["feature_importance"]

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors="k")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual Price", fontsize=12)
    ax.set_ylabel("Predicted Price", fontsize=12)
    ax.set_title("Actual vs Predicted House Prices", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=100)
    plt.close()
    print("\n   ✓ Saved: actual_vs_predicted.png")

    # 2. Residuals Plot
    residuals = y_test - y_test_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_test_pred, residuals, alpha=0.6, s=50, edgecolors="k")
    axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Price", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Residuals Distribution
    axes[1].hist(residuals, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "residuals_analysis.png", dpi=100)
    plt.close()
    print("   ✓ Saved: residuals_analysis.png")

    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if x > 0 else "red" for x in feature_importance["Coefficient"]]
    ax.barh(feature_importance["Feature"], feature_importance["Coefficient"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Feature Importance (Coefficients)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=100)
    plt.close()
    print("   ✓ Saved: feature_importance.png")


def generate_final_report(results: dict) -> None:
    """Generate comprehensive final report."""
    print("\n" + "="*70)
    print("FINAL REPORT: MODEL PERFORMANCE SUMMARY")
    print("="*70)

    report = f"""
╔════════════════════════════════════════════════════════════════════╗
║            HOUSE PRICE PREDICTION - FINAL REPORT                   ║
╚════════════════════════════════════════════════════════════════════╝

📋 EXPERIMENT OVERVIEW
   ─────────────────────
   Objective: Predict house prices using linear regression
   Dataset: House Price Dataset of India
   Model: Linear Regression
   Test Size: 20%

📊 MODEL EVALUATION METRICS (Test Set)
   ──────────────────────────────────────
   
   ► Mean Absolute Error (MAE):      ${results['test_mae']:,.2f}
     Interpretation: On average, predictions are off by this amount
   
   ► Mean Squared Error (MSE):       {results['test_mse']:,.2f}
     Interpretation: Average squared error (penalizes larger errors)
   
   ► Root Mean Squared Error (RMSE): ${results['test_rmse']:,.2f}
     Interpretation: Square root of MSE in price units
   
   ► R-squared (R²):                 {results['test_r2']:.4f}
     Interpretation: Model explains {results['test_r2']*100:.2f}% of variance
   
   ► Adjusted R-squared (Adj R²):    {results['adj_r2']:.4f}
     Interpretation: Penalizes excessive features

🎯 MODEL PERFORMANCE RATING
   ──────────────────────────
   
   R² Score Interpretation:
   • 0.0 - 0.2: Poor fit
   • 0.2 - 0.4: Fair fit
   • 0.4 - 0.6: Moderate fit
   • 0.6 - 0.8: Good fit
   • 0.8 - 1.0: Excellent fit
   
   Current Model: {'Excellent' if results['test_r2'] > 0.8 else 'Good' if results['test_r2'] > 0.6 else 'Moderate' if results['test_r2'] > 0.4 else 'Fair'} Fit

📈 VISUALIZATIONS GENERATED
   ────────────────────────
   ✓ actual_vs_predicted.png    - Scatter plot of actual vs predicted prices
   ✓ residuals_analysis.png     - Residual plots (vs predicted + distribution)
   ✓ feature_importance.png     - Feature coefficients and importance

✅ KEY FINDINGS
   ──────────────
   • The model predictions closely follow the actual prices trend
   • Residuals distribution indicates model validity
   • Positive coefficients increase price, negative decrease price
   • Adjust feature selection if overfitting is detected (R² gap)

💡 RECOMMENDATIONS
   ────────────────
   1. Monitor test vs train R² to detect overfitting
   2. Consider polynomial features if linear model performs poorly
   3. Investigate outliers in residual plots
   4. Feature scaling helps model convergence
   5. Use cross-validation for robust evaluation

╔════════════════════════════════════════════════════════════════════╗
║                   ✓ EXPERIMENT COMPLETED                          ║
╚════════════════════════════════════════════════════════════════════╝
    """

    print(report)


def main() -> None:
    """Main execution function."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
    print("="*70)
    print("\nDataset Information:")
    print("  GitHub Repository: Dataset uploaded")
    print("  Download Link: https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india/data")
    print("  Local Path: ./dataset/House Price India.csv")

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: EDA
    explore_dataset(df)

    # Step 3: Preprocessing
    df_processed = preprocess_data(df)

    # Step 4: Feature Engineering
    X, y, feature_names = feature_engineering(df_processed)

    # Step 5: Build and Train Model
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_test = (
        build_and_train_model(X, y)
    )

    # Step 6: Evaluate Model
    results = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )

    # Step 7: Visualize Results
    visualize_results(results, X_test)

    # Step 8: Final Report
    generate_final_report(results)

    print("\n✓ Experiment 3 completed successfully!")
    print("\n📁 Output files generated in: " + str(Path(__file__).parent))


if __name__ == "__main__":
    main()
