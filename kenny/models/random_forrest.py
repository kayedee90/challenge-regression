import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt

"""
REMOVED PPM² FROM DATASET TO FIX 100% ACCURACY
Model explains about 59.0% of the variation in property prices.
Squared difference between predicted and actual prices is €119,698.
"""

# Load the dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\raw_data.csv")

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
df['epcScore'] = df['epcScore'].map(epc_map)
#print(df['epcScore'].value_counts().sort_index())
#print(df['buildingCondition'].value_counts().sort_index())

# Create a list of numerical features to compute from
num_features = [
    'habitableSurface', 
    'bedroomCount',
    'hasGarden',
    'gardenSurface',
    'hasTerrace',
    'hasParking'
    ]
# Create a list of categorical features to compute from
cat_features = ['postCode','subtype','epcScore','buildingCondition']
# Combine into 1 feature list
features = num_features + cat_features

# X: feed the features to the model
X = df[features]
# y: the feature to predict
y = df['price']

# Split the dataset 20/80 for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create an impution strategy for missing values
num_imputer = SimpleImputer(strategy='mean') # Take the mean for missing values
cat_imputer = OneHotEncoder(handle_unknown='ignore') # Ignore missing values

# Combine impution strategies
preprocessor = ColumnTransformer([
    ('num', num_imputer, num_features),
    ('cat', cat_imputer, cat_features)
])


# Create a pipeline for the impution strategy
model_pipeline = Pipeline([
    ('imputer', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,        # fewer trees = faster
        max_depth=15,           # restrict how deep they grow
        n_jobs=-1,              # use all your CPU cores
        random_state=1
    ))
])

# Define the model pipeline
model_pipeline.fit(X_train, y_train)
# Define the prediction param
y_prediction = model_pipeline.predict(X_test)
# Grab model from pipeline
model = model_pipeline.named_steps['model']

# Get feature names after preprocessing
feature_names = model_pipeline.named_steps['imputer'].get_feature_names_out()
feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
importances = pd.Series(model.feature_importances_, index=feature_names)
importance_df = importances.sort_values(ascending=False).reset_index()
print(importance_df.head(15))
print("------")
print(f"  R² Score: {r2 * 100:.1f}%")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
print(f"  MAPE: {mape * 100:.2f}%")
print("------")

