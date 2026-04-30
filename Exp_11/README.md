# Exp. 11: Pneumonia Diagnosis from Chest X-Ray Numerical Features

This project implements a complete numerical feature-based machine learning pipeline for pneumonia diagnosis from chest X-ray derived features.

## Aim
Develop a machine learning based disease diagnosis system for pneumonia detection from Chest X-Ray images using a validated numerical feature dataset.

## What is implemented (Objective-wise)
1. Medical image diagnosis context through radiomics-style numerical features.
2. Data preprocessing + EDA over train/val/test feature splits.
3. Feature-group analysis for:
   - Global intensity statistics
   - Texture descriptors (GLCM)
   - Frequency features (FFT)
   - Edge metrics
   - LBP features
4. Classical ML model training and comparison.
5. Feature engineering + feature selection + dimensionality reduction.
6. Performance evaluation with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report.

## Project structure
- `dataset\train_features.csv`
- `dataset\val_features.csv`
- `dataset\test_features.csv`
- `exp11_pneumonia_ml.py` (main experiment script)
- `outputs\figures\` (plots)
- `outputs\tables\` (metrics/tables/json reports)

## Label mapping used
Since CSV files are feature-only plus image filename, labels are derived from the `image` name:
- **Pneumonia (1):** filename contains `person`, `pneumonia`, `bacteria`, or `virus`
- **Normal (0):** otherwise

## Steps performed by the script
1. Load train/val/test CSVs and derive labels.
2. Run EDA:
   - split summary
   - class distribution
   - missing values
   - feature correlation analysis
3. Analyze validated feature groups (global/GLCM/FFT/edge/LBP).
4. Train and compare baseline models:
   - Logistic Regression
   - SVM (RBF)
   - KNN
   - Random Forest
   - Gradient Boosting
5. Evaluate per-feature-group logistic models.
6. Apply feature engineering and evaluate:
   - Mutual-information based SelectKBest + Logistic Regression
   - PCA (95% variance) + Logistic Regression
7. Save best-model confusion matrix and classification report.

## How to run
From this folder:

```bash
python exp11_pneumonia_ml.py
```

## Key produced files
In `outputs\tables\`:
- `split_summary.csv`
- `class_distribution.csv`
- `feature_group_statistics.csv`
- `baseline_model_results_val.csv`
- `baseline_model_results_test.csv`
- `feature_group_model_results.csv`
- `feature_engineering_selection_pca_results.csv`
- `selected_features_mutual_info.csv`
- `best_model_summary.json`
- `best_model_test_classification_report.json`

In `outputs\figures\`:
- `class_distribution.png`
- `feature_correlation_heatmap.png`
- `feature_group_boxplots.png`
- `pca_explained_variance_curve.png`
- `best_model_confusion_matrix.png`

## Current run highlights
- Baseline best on validation F1: **Logistic Regression** (F1 = 0.8421)
- Best test F1 among baseline models: **SVM_RBF** (F1 = 0.8618)
- Best model by validation selection logic was saved as Logistic Regression.

Note: validation set is very small (`16` samples), so model ranking can fluctuate; test-set comparison is more stable.
