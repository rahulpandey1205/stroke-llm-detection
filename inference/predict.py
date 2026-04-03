import joblib
import pandas as pd
import shap
import os
import numpy as np

# Define model path
MODEL_PATH = os.path.join("models", "stroke_prediction_model.pkl")

# Cache for the loaded model and metadata
_model_package = None
_explainer = None

def load_model_package():
    """Lazy loads the model package."""
    global _model_package
    if _model_package is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model_package = joblib.load(MODEL_PATH)
    return _model_package

def get_explainer():
    """Lazy loads the SHAP explainer."""
    global _explainer
    if _explainer is None:
        package = load_model_package()
        model = package["model"]
        
        # XGBoost models work directly with TreeExplainer
        _explainer = shap.TreeExplainer(model)
    return _explainer

def predict_risk(patient_data):
    """
    Predicts stroke risk level and returns probability with feature importance.
    """
    package = load_model_package()
    model = package["model"]
    features = package["features"]
    scaler = package.get("scaler")
    
    # Create copy of patient data to avoid modifying original
    data = patient_data.copy()
    
    # --- Feature Engineering (Replicated from preprocessing) ---
    
    # 1. Age Groups
    age = data.get("age", 50)
    if age <= 30: data["age_group"] = 0
    elif age <= 50: data["age_group"] = 1
    elif age <= 70: data["age_group"] = 2
    else: data["age_group"] = 3
    
    # 2. BMI Categories
    bmi = data.get("bmi", 25)
    if bmi <= 18.5: data["bmi_cat"] = 0
    elif bmi <= 25: data["bmi_cat"] = 1
    elif bmi <= 30: data["bmi_cat"] = 2
    else: data["bmi_cat"] = 3
    
    # 3. Glucose Risk Levels
    glucose = data.get("avg_glucose_level", 100)
    if glucose <= 90: data["glucose_risk"] = 0
    elif glucose <= 140: data["glucose_risk"] = 1
    else: data["glucose_risk"] = 2
    
    # 4. Interaction Features
    data["age_hypertension"] = age * data.get("hypertension", 0)
    data["age_heart_disease"] = age * data.get("heart_disease", 0)

    # Convert input to dataframe
    df = pd.DataFrame([data])
    
    # Ensure all required features exist (fill with 0 if missing)
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
            
    # Reorder columns to match training
    df = df[features]
    
    # Scale data if scaler exists
    X_scaled = df
    if scaler:
        X_scaled = scaler.transform(df)
    
    # Predict probability
    probability = float(model.predict_proba(X_scaled)[0][1]) * 100
    
    # Get custom threshold from package or use default 50
    # For the new model, we might want to lower it again if we want higher recall
    threshold_val = 20.0 # Adjusted for the new balanced model
    
    # Risk classification
    if probability > 50:
        risk = "High Stroke Risk"
    elif probability > threshold_val:
        risk = "Moderate Stroke Risk"
    else:
        risk = "Low Stroke Risk"
        
    # Get SHAP contributions
    explainer = get_explainer()
    shap_contributors = []
    
    try:
        # We pass the scaled data for SHAP as the model was trained on scaled data
        shap_values = explainer.shap_values(X_scaled)
        
        # XGBoost SHAP values might be a single array for binary classification
        if isinstance(shap_values, list):
            values = shap_values[-1][0]
        else:
            values = shap_values[0]
            
        contributions = {}
        for i, feature in enumerate(features):
            contributions[feature] = float(values[i])
            
        # Sort by absolute contribution value
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature, value in sorted_features[:4]:
            sign = "+" if value > 0 else "-"
            shap_contributors.append(f"{feature} ({sign})")
            
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        shap_contributors = ["Explanation unavailable"]

    return {
        "probability": round(probability, 2),
        "risk": risk,
        "shap_contributors": shap_contributors
    }

if __name__ == "__main__":
    # Test case (must match feature set)
    sample_patient = {
        "gender": 1,
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": 1,
        "work_type": 2,
        "Residence_type": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": 1
    }
    
    print("\nAdvanced Stroke Risk Prediction (Test)")
    print("---------------------------------------")
    try:
        result = predict_risk(sample_patient)
        print(f"Risk Level: {result['risk']}")
        print(f"Probability: {result['probability']}%")
        print(f"Top Factors: {result['shap_contributors']}")
    except Exception as e:
        print(f"Prediction failed: {e}. Ensure you have trained the new model first.")
