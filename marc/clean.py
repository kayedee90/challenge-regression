import pandas as pd
import numpy as np
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
columns_to_drop = ['id', 'url', 'MunicipalityCleanName', 'price_square_meter']
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

# === Final check ===
print(f"✅ Final shape: {df.shape}")
print("✅ Sample of cleaned data:")
print(df.head())

# === Save cleaned dataset ===
df.to_csv(output_path, index=False)
print(f"💾 Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")

