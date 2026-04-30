# Dataset note:
# We have uploaded the dataset to the GitHub repository.
# Dataset link: https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data
# The dataset used locally is available in the Exp_8 folder as train.csv and test.csv.

# Fix for UnicodeDecodeError in subprocess on Windows
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Fix for UnicodeDecodeError in subprocess on Windows
import subprocess
original_run = subprocess.run
def patched_run(*args, **kwargs):
    # Check if it's the problematic wmic command
    if args and len(args) > 0 and isinstance(args[0], list) and 'wmic' in args[0] and 'CPU' in args[0]:
        # Force text=False for wmic CPU command
        kwargs['text'] = False
        result = original_run(*args, **kwargs)
        # Decode manually with error handling
        if result.stdout:
            result.stdout = result.stdout.decode('utf-8', errors='replace')
        if result.stderr:
            result.stderr = result.stderr.decode('utf-8', errors='replace')
        return result
    if 'text' in kwargs and kwargs['text']:
        kwargs['encoding'] = 'utf-8'
        try:
            return original_run(*args, **kwargs)
        except UnicodeDecodeError:
            # Fallback: use text=False and decode manually
            kwargs['text'] = False
            result = original_run(*args, **kwargs)
            result.stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else None
            result.stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else None
            return result
    return original_run(*args, **kwargs)
subprocess.run = patched_run

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Objective 1: Understand the structure and key features of a credit dataset
print("Objective 1: Understanding the dataset structure")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("Train columns:", list(train_df.columns))
print("Data types:\n", train_df.dtypes)
print("First 5 rows of train data:\n", train_df.head())
print("Summary statistics:\n", train_df.describe())

# Objective 2: Preprocessing and exploratory data analysis
print("\nObjective 2: Preprocessing and EDA")

# Handle missing values
train_df.replace('_', np.nan, inplace=True)
train_df.replace('', np.nan, inplace=True)

# Convert numeric columns
numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

for col in numeric_cols:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

# Fill missing values
for col in train_df.columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        med = train_df[col].median()
        if pd.isna(med):
            train_df[col].fillna(0, inplace=True)
        else:
            train_df[col].fillna(med, inplace=True)
    else:
        mode_series = train_df[col].mode()
        if not mode_series.empty:
            train_df[col].fillna(mode_series[0], inplace=True)
        else:
            train_df[col].fillna('Unknown', inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']

for col in categorical_cols:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        label_encoders[col] = le

# EDA: Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='Credit_Score', data=train_df)
plt.title('Distribution of Credit Scores')
plt.savefig('credit_score_distribution.png')
plt.show()

# Objective 3: Identify important features using PCA
print("\nObjective 3: Feature importance using PCA")

# Prepare features
features = [col for col in train_df.columns if col not in ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Credit_History_Age', 'Credit_Score']]
X = train_df[features]
y = train_df['Credit_Score']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.92)
X_pca = pca.fit_transform(X_scaled)

print("Number of components selected:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

plt.figure(figsize=(8, 6))
plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.savefig('pca_variance.png')
plt.show()

# Feature importance from PCA
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(pca.components_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("Top 10 important features:\n", feature_importance.head(10))

# Objective 4 & 5: Implement classifiers and evaluate
print("\nObjectives 4 & 5: Classification models and evaluation")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=1),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Comparison
print("\nModel Comparison:")
comparison_df = pd.DataFrame(results).T.drop('Confusion Matrix', axis=1)
print(comparison_df)

# Plot comparison
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Determine the best classifier
best_model = comparison_df['Accuracy'].idxmax()
best_accuracy = comparison_df['Accuracy'].max()
print(f"\nBest Classifier for this task: {best_model} with Accuracy: {best_accuracy:.4f}")

print("\nExperiment completed. Plots saved as PNG files.")