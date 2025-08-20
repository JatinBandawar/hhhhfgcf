# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

# Preprocessing tools
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from scipy.stats import uniform, randint

import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Data cleaning and preprocessing...")
# Data Cleaning and Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Convert 'Yes'/'No' in Churn to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert SeniorCitizen to object type for consistent preprocessing
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

# Visualizing categorical features (if needed)
# print("Visualizing categorical features...")
# colors = {1: 'red', 0: 'blue'}
# for col in df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure']).select_dtypes(include='object'):
#     plt.figure(figsize=(5, 3))
#     sns.countplot(data=df, x=col, hue='Churn', palette=colors)
#     plt.title(col)
#     plt.tight_layout()
#     plt.show()

# Feature Engineering
print("Performing feature engineering...")

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Define preprocessing for numerical and categorical data
numerical_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# Apply SMOTEENN (uncomment if needed)
# print("Applying SMOTEENN for class balancing...")
# sm = SMOTEENN()
# X_res, y_res = sm.fit_resample(X, y)

# Use original data without SMOTEENN for simplicity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definitions
models = [
    ('Random Forest', RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5]}),
    ('Support Vector Machine', SVC(random_state=42, probability=True),
        {'C': [1, 10], 'gamma': ['scale', 'auto']}),
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000),
        {'C': [0.1, 1], 'penalty': ['l2']}),
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {'max_depth': [None, 10], 'min_samples_split': [2, 5]})
]

best_model = None
best_accuracy = 0.0
model_scores = []

print("Training models...")
# Training models
for name, model, params in models:
    print(f"Training {name}...")
    
    # Create and fit pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Use GridSearchCV for hyperparameter tuning
    search = GridSearchCV(pipeline, param_grid={'classifier__' + key: value for key, value in params.items()}, cv=3)
    search.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores.append({'Model': name, 'Accuracy': acc})
    
    print(f"Model: {name}\nTest Accuracy: {round(acc, 3)}\n")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = search

print(f"Best model: {best_model.best_estimator_.named_steps['classifier'].__class__.__name__}")
print(f"Best accuracy: {best_accuracy:.3f}")

# Save best model
print("Saving model...")
joblib.dump(best_model, 'churn_model_pipeline.pkl')

# Save column information for reference
with open('model_columns.pkl', 'wb') as f:
    pickle.dump({
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'preprocessor': preprocessor
    }, f)

# Plot model comparison (uncomment if needed)
# scores_df = pd.DataFrame(model_scores)
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Accuracy', data=scores_df, palette='pastel')
# plt.xticks(rotation=45)
# plt.ylim(0, 1)
# plt.title('Model Accuracy Comparison')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

print("Training and evaluation complete!")