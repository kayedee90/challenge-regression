import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder

# === Load data from parent directory ===
input_path = "../data/raw_data.csv"
output_path = "../data/cleaned_data_mvg.csv"

print(f"📂 Loading data from: {input_path}")
df = pd.read_csv(input_path)
print(f"✅ Data loaded. Shape: {df.shape}")

# === Get to know and visualize our data ===
# Data Overview
print("Data Overview:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nData types:")
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicates: {duplicates}")

# Display basic info about the dataset
print("\nBasic Information:")
print(df.info())

# Show the first few rows of the dataset to inspect the data
print("\nFirst few rows of the dataset:")
print(df.head())


# === Drop unused or problematic columns ===
columns_to_drop = ['id', 'url', 'MunicipalityCleanName', 'price_square_meter', 'locality']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"🧹 Dropped columns: {columns_to_drop}")

# === Remove duplicates ===
before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"🧹 Removed {before - after} duplicate rows")

# === Fill missing numeric values with median ===
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"🧮 Filled NaNs in numeric columns with median")

# === Fill missing binary/object values with default ===
object_cols = df.select_dtypes(include='object').columns
df[object_cols] = df[object_cols].fillna("Unknown")
print(f"🧮 Filled NaNs in object columns with 'Unknown'")

# === Encode categorical columns using OrdinalEncoder ===
categorical_cols = ['type', 'subtype', 'province', 'region', 'buildingCondition', 'epcScore']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

if not set(categorical_cols).issubset(df.columns):
    missing = list(set(categorical_cols) - set(df.columns))
    raise ValueError(f"Missing expected categorical columns: {missing}")

df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
print(f"🏷️ Encoded categorical columns: {categorical_cols}")

# Recompute price_per_m2 for EDA
df['price_square_meter'] = df['price'] / df['habitableSurface']

# Remove negative or zero values
df = df[df['price'] > 0]
df = df[df['habitableSurface'] > 0]
df = df[df['price_square_meter'] > 0]

# Save encodings 
# Create output directory if it doesn't exist
os.makedirs("encoding_maps", exist_ok=True)

# Define your mappings
type_map = {"house": 0, "apartment": 1}
subtype_map = {v: i for i, v in enumerate(df["subtype"].dropna().unique())}
condition_map = {
    "TO_RENOVATE": 0,
    "TO_RESTORE": 1,
    "GOOD": 2,
    "JUST_RENOVATED": 3,
    "AS_NEW": 4,
    "NEW": 5,
}
region_map = {v: i for i, v in enumerate(df["region"].dropna().unique())}

# Save each mapping as Excel
pd.DataFrame(type_map.items(), columns=["label", "code"]).to_excel("encoding_maps/type_encoding.xlsx", index=False)
pd.DataFrame(subtype_map.items(), columns=["label", "code"]).to_excel("encoding_maps/subtype_encoding.xlsx", index=False)
pd.DataFrame(condition_map.items(), columns=["label", "code"]).to_excel("encoding_maps/condition_encoding.xlsx", index=False)
pd.DataFrame(region_map.items(), columns=["label", "code"]).to_excel("encoding_maps/region_encoding.xlsx", index=False)

# === Final check ===
print(f"✅ Final shape: {df.shape}")
print("✅ Sample of cleaned data:")
print(df.head())

# === Save cleaned dataset ===
df.to_csv(output_path, index=False)
print(f"💾 Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")

