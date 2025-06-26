import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("immoweb-dataset-clean.csv")  # Update with your actual file name

# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Remove rows with NaN values
df.dropna(inplace=True)

# 3. Remove text (non-numerical) columns
# Keep only numeric columns for ML
df_numeric = df.select_dtypes(include=[np.number])

# OPTIONAL: if you want to encode categorical vars instead of dropping them:
# df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Remove highly correlated features (correlation > 0.9)
corr_matrix = df_numeric.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find columns with correlation > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop those columns
df_numeric.drop(columns=to_drop, inplace=True)

# Final cleaned dataset ready for ML
df_ml_ready = df_numeric.copy()

# Save it to CSV if needed
df_ml_ready.to_csv("dataset_ready_for_ml.csv", index=False)

print("✅ Dataset ready for machine learning.")
print(f"Final shape: {df_ml_ready.shape}")
