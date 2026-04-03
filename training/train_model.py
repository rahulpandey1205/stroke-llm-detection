import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# -----------------------------------------
# 1. Load Main Dataset
# -----------------------------------------
data_path = "data/processed/processed_stroke_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Processed data not found at {data_path}")

data = pd.read_csv(data_path)
print(f"Original dataset size: {len(data)}")

# -----------------------------------------
# 2. Features and Target Split
# -----------------------------------------
X = data.drop("stroke", axis=1)
y = data["stroke"]
feature_names = X.columns.tolist()

# -----------------------------------------
# 3. Train/Test Split (Stratified)
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------------
# 4. Handle Class Imbalance with SMOTETomek
# -----------------------------------------
print("Applying SMOTETomek to balance classes...")
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
print(f"Resampled training size: {len(X_train_res)}")

# -----------------------------------------
# 5. Scaling
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 6. Hyperparameter Tuning with RandomizedSearchCV
# -----------------------------------------
print("Starting Hyperparameter Tuning...")

param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'scale_pos_weight': [1, 3, 5]
}

xgb_base = XGBClassifier(eval_metric='aucpr', random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    xgb_base, 
    param_distributions=param_dist, 
    n_iter=20, 
    scoring='f1', # Better balance than just recall
    cv=cv, 
    verbose=1, 
    n_jobs=-1, 
    random_state=42
)

random_search.fit(X_train_scaled, y_train_res)

print(f"Best Parameters: {random_search.best_params_}")
best_xgb = random_search.best_estimator_

# -----------------------------------------
# 7. Final Evaluation
# -----------------------------------------
probs = best_xgb.predict_proba(X_test_scaled)[:, 1]
# Using a sensitivity-optimized threshold
threshold = 0.3 
custom_predictions = (probs >= threshold).astype(int)

accuracy = accuracy_score(y_test, custom_predictions)
precision = precision_score(y_test, custom_predictions)
recall = recall_score(y_test, custom_predictions)
f1 = f1_score(y_test, custom_predictions)
roc_auc = roc_auc_score(y_test, probs)

print("\n--- Optimized XGBoost Model Training Completed ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f} (Crucial for Stroke Detection)")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report (Threshold 0.3):")
print(classification_report(y_test, custom_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, custom_predictions))

# -----------------------------------------
# 8. Save Model Package
# -----------------------------------------
model_package = {
    "model": best_xgb,
    "scaler": scaler,
    "features": feature_names,
    "threshold": threshold,
    "metrics": {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
}

joblib.dump(model_package, "models/stroke_prediction_model.pkl")
print("\nModel saved successfully at models/stroke_prediction_model.pkl")
