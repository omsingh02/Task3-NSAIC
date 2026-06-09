"""
1. What was the single most impactful change you made?
Addressing the severe class imbalance (1906:334) and non-linear feature relationships by switching from LogisticRegression to RandomForestClassifier with balanced class weights.

2. Why does it work — mathematically or statistically?
The dataset has an ~85% negative class imbalance. Logistic Regression optimizes for overall accuracy, heavily favoring the majority class, resulting in poor minority class recall and a low F1-Score. Additionally, Logistic Regression assumes linear decision boundaries. RandomForest handles class imbalance intrinsically using recursive partitioning and `class_weight='balanced'` (which applies inverse frequency weights). It also inherently handles varying scales among features without needing explicit scaling.

3. Table: baseline metric | improved metric | delta
| Metric | Baseline | Improved | Delta |
| --- | --- | --- | --- |
| F1-Score | 0.1882 | 0.5189 | +175.68% |
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def get_baseline_f1(df):
    # Baseline logic matching the intern's implementation
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.fillna(0)
    X = df_numeric.drop(['ID', 'Response'], axis=1)
    y = df_numeric['Response']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

def main():
    # 1. Load Data
    df = pd.read_csv('marketing_campaign.csv', sep='\t')
    
    # Run and capture baseline
    baseline_f1 = get_baseline_f1(df)
    
    # 2. Improved Preprocessing
    # Drop non-predictive or redundant columns
    df_clean = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'])
    
    # Impute missing values (Income) with median
    df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())
    
    # Encode categoricals using pd.get_dummies
    df_clean = pd.get_dummies(df_clean, columns=['Education', 'Marital_Status'])
    
    # Define Target and Features
    X = df_clean.drop('Response', axis=1)
    y = df_clean['Response']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Improved Modeling with Small Hyperparameter Search
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # <=8 combinations: 2 n_estimators * 3 max_depths = 6 combinations
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    improved_f1 = f1_score(y_test, y_pred)
    
    # Print results
    print("--- AUDIT SUMMARY ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values: Income ({df['Income'].isnull().sum()})")
    print(f"Class Imbalance: 0: {df['Response'].value_counts()[0]}, 1: {df['Response'].value_counts()[1]}")
    print("\n--- RESULTS ---")
    print(f"Baseline F1-Score: {baseline_f1:.4f}")
    print(f"Improved F1-Score: {improved_f1:.4f}")
    print(f"Best Params: {grid_search.best_params_}")
    
    delta = (improved_f1 - baseline_f1) / baseline_f1 * 100
    print(f"Relative Improvement: +{delta:.2f}%")

if __name__ == "__main__":
    main()
