import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Base Models
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Final Estimator
from sklearn.linear_model import Lasso

# Load Dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\raw_data.csv")

# 🔧 Map condition and EPC scores
condition_map = {
    "TO_RESTORE": 0, "TO_BE_DONE_UP": 1, "TO_RENOVATE": 1,
    "GOOD": 2, "JUST_RENOVATED": 3, "AS_NEW": 4
}
epc_map = {
    'G': 0, 'F': 1, 'E': 2, 'D': 3, 'C': 4,
    'B': 5, 'A': 6, 'A+': 7, 'A++': 8
}

df['buildingCondition'] = df['buildingCondition'].map(condition_map)
df['epcScore'] = df['epcScore'].map(epc_map)

# Define Features
num_features = ['habitableSurface', 'bedroomCount', 'hasGarden', 'gardenSurface', 'hasTerrace', 'hasParking']
cat_features = ['postCode', 'subtype', 'epcScore', 'buildingCondition']
features = num_features + cat_features

X = df[features].copy()
y = np.log1p(df['price'])  # Apply log transformation to stabilize variance

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Build Stacked Model
stacked_model = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=15, n_jobs=-1, random_state=1)),
        ('svr', SVR(kernel='rbf', C=10, epsilon=0.1))
    ],
    final_estimator=Lasso(alpha=0.05),
    passthrough=False,
    n_jobs=-1
)

# Train Model
stacked_model.fit(X_train_processed, y_train)

# Predictions
y_pred = stacked_model.predict(X_test_processed)
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

# Evaluate Performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original)

print("\nStacked Ensemble Performance")
print("================================")
print(f"R² Score: {r2 * 100:.2f}%")
print(f"RMSE: €{rmse:,.0f}")
print(f"MAE: €{mae:,.0f}")
print(f"MAPE: {mape * 100:.2f}%")

# Save Model & Preprocessor
joblib.dump((preprocessor, stacked_model), "models/stacked_model_bundle2.pkl")
print("\n✅ Model bundle saved to 'models/stacked_model_bundle2.pkl'")