import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error

"""
Model explains about 82.0% of the variation in property prices.
Squared difference between predicted and actual prices is €79,012.

dropping price per square meter reduced accuracy to 67.9%
"""

# Load the dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\cleaned_data_mvg.csv")

# Create a list of numerical features to compute from
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
    X, y, test_size=0.2, random_state=1
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
    ('model', linear_model.LinearRegression())
])

model_pipeline.fit(X_train, y_train)



y_prediction = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
mape = mean_absolute_percentage_error(y_test, y_prediction)

print(f"  R² Score: {r2 * 100:.1f}%")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
print(f"  MAPE: {mape * 100:.2f}%")
print("------")