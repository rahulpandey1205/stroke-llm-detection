import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")

print("Original Shape:", data.shape)

# remove ID column
if 'id' in data.columns:
    data = data.drop("id", axis=1)

# handle missing BMI values
data["bmi"].fillna(data["bmi"].mean(), inplace=True)

# --- Feature Engineering ---

# 1. Age Groups
data["age_group"] = pd.cut(
    data["age"],
    bins=[0, 30, 50, 70, 120],
    labels=[0, 1, 2, 3] # Young, Middle, Senior, Elderly
).astype(int)

# 2. BMI Categories
data["bmi_cat"] = pd.cut(
    data["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3] # Underweight, Normal, Overweight, Obese
).astype(int)

# 3. Glucose Risk Levels
data["glucose_risk"] = pd.cut(
    data["avg_glucose_level"],
    bins=[0, 90, 140, 300],
    labels=[0, 1, 2] # Normal, Prediabetes, Diabetes
).astype(int)

# 4. Interaction Features
data["age_hypertension"] = data["age"] * data["hypertension"]
data["age_heart_disease"] = data["age"] * data["heart_disease"]

# convert categorical columns to numbers
label_encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

print("\nCleaned Dataset Preview with New Features:")
print(data.head())

# save processed data
data.to_csv("data/processed/processed_stroke_data.csv", index=False)

print("\nPreprocessing Completed Successfully")