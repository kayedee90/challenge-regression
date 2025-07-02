import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
from catboost import CatBoostRegressor

"""
Model hit:
  R² Score: 75.1%
  RMSE: €93,181
  MAE: €63,517
  MAPE: 18.62%
------

Top Features:
             Feature  Importance
0           postCode   34.225737
1   habitableSurface   30.735052
2            subtype    9.017687
3  buildingCondition    8.464933
4           epcScore    8.192663
5       bedroomCount    4.395673
6         hasParking    2.114078
7      gardenSurface    1.543669
8         hasTerrace    1.084918
9          hasGarden    0.225591
"""
# Load the dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\raw_data.csv")

# Map condition values
condition_map = {
    "TO_RESTORE": 0,
    "TO_BE_DONE_UP": 1,
    "TO_RENOVATE": 1,
    "GOOD": 2,
    "JUST_RENOVATED": 3,
    "AS_NEW": 4
}

# Map EPC scores
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
df['epcScore'] = df['epcScore'].map(epc_map)
df['buildingCondition'] = df['buildingCondition'].map(condition_map)

# Define features
num_features = [
    'habitableSurface',
    'bedroomCount',
    'hasGarden',
    'gardenSurface',
    'hasTerrace',
    'hasParking'
]
cat_features = ['postCode', 'subtype', 'epcScore', 'buildingCondition']
features = num_features + cat_features

# Feature matrix and target
X = df[features]
y = np.log1p(df['price'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# CatBoost model
model_name = CatBoostRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    cat_features=cat_features,
    verbose=0,
    random_state=1
)

# Fit model
model_name.fit(X_train, y_train)

# Predict in log-scale, convert back to real € scale
y_prediction = np.expm1(model_name.predict(X_test))
y_test_real = np.expm1(y_test) 

# Feature importance
importance_df = pd.Series(model_name.feature_importances_, index=X_train.columns) \
    .sort_values(ascending=False) \
    .reset_index()
importance_df.columns = ['Feature', 'Importance']
print(importance_df.head(15))

# Metrics
rmse = root_mean_squared_error(y_test_real, y_prediction)
r2 = r2_score(y_test_real, y_prediction)
mae = mean_absolute_error(y_test_real, y_prediction)
mape = mean_absolute_percentage_error(y_test_real, y_prediction)

# Output results
print(model_name)
print(f"  R² Score: {r2 * 100:.1f}%")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
print(f"  MAPE: {mape * 100:.2f}%")
print("------")
