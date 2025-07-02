
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
# Import Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\raw_data.csv")

# Assign values to buildingCondition and epcScore
condition_map = {
    "TO_RESTORE": 0,
    "TO_BE_DONE_UP": 1,
    "TO_RENOVATE": 1,
    "GOOD": 2,
    "JUST_RENOVATED": 3,
    "AS_NEW": 4
}

epc_map = {
    'G': 0,
    'F': 1,
    'E': 2,
    'D': 3,
    'C': 4,
    'B': 5,
    'A': 6,
    'A+': 7,
    'A++': 8
}

# Apply the mappings
df['epcScore'] = df['epcScore'].map(epc_map)
df['buildingCondition'] = df['buildingCondition'].map(condition_map)

# Feature setup
num_features = ['habitableSurface', 'bedroomCount', 'hasGarden', 'gardenSurface', 'hasTerrace', 'hasParking']
cat_features = ['postCode', 'subtype', 'epcScore', 'buildingCondition']

features = num_features + cat_features
X = df[features].copy()

y = np.log1p(df['price'])  # log-transformed target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#create preprocessor to separate approaches for each model
preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_features)
])

# Fit preprocessor and transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define stacked model
stacked_model = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1)),
        ('hist', HistGradientBoostingRegressor(max_iter=500, max_depth=10, random_state=1)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate= 0.03,max_depth=15, n_jobs=-1, random_state=1))
    ],
    final_estimator=LinearRegression(),
    n_jobs=-1
)

# Train stacked model
stacked_model.fit(X_train_processed, y_train)
y_pred = stacked_model.predict(X_test_processed)

# Convert predictions back to original scale
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original)

# Display results
print("\nStacked Ensemble Performance:")
print("=========")
print(f"R²: {r2 * 100:.1f}%")
print(f"RMSE: €{rmse:,.0f}")
print(f"MAE: €{mae:,.0f}")
print(f"MAPE: {mape * 100:.2f}%")