import pandas as pd

# load dataset
data = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")

print("Dataset Loaded Successfully")

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())