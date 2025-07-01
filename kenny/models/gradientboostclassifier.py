import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor

"""
                            index         0
0                habitableSurface  0.434860
1                   postCode_8300  0.098624
2                   subtype_HOUSE  0.062749
3        buildingCondition_AS_NEW  0.051260
4   buildingCondition_TO_RENOVATE  0.025742
5                    bedroomCount  0.023855
6                      epcScore_6  0.022242
7                   postCode_1050  0.020994
8                   postCode_1180  0.015340
9                   subtype_VILLA  0.011655
10                     epcScore_5  0.010025
11                  postCode_2000  0.009932
12                  postCode_1150  0.009924
13              subtype_PENTHOUSE  0.009539
14                     hasParking  0.008042
<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>
  R² Score: 65.6%
  RMSE: €110,026
  MAE: €80,012
  MAPE: 26.42%
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
    X, y, test_size=0.5, random_state=1
)

# Create an impution strategy for missing values
num_imputer = SimpleImputer(strategy='mean') # Take the mean for missing values
cat_imputer = OneHotEncoder(handle_unknown='ignore') # Ignore missing values

# Combine impution strategies
preprocessor = ColumnTransformer([
    ('num', num_imputer, num_features),
    ('cat', cat_imputer, cat_features)
])



# Loop through each model
model_name = GradientBoostingRegressor

model_pipeline = Pipeline([
    ('imputer', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=50,
                                        learning_rate=0.1,
                                        max_depth=8,
                                        random_state=1))
])

model_pipeline.fit(X_train, y_train)
y_prediction = model_pipeline.predict(X_test)

# Grab model from pipeline
model = model_pipeline.named_steps['model']

# Get rankings of most important features
feature_names = model_pipeline.named_steps['imputer'].get_feature_names_out()
feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
importances = pd.Series(model.feature_importances_, index=feature_names)
importance_df = importances.sort_values(ascending=False).reset_index()
print(importance_df.head(15))

mse = mean_squared_error(y_test, y_prediction)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
mape = mean_absolute_percentage_error(y_test, y_prediction)

print(f"{model_name}")
print(f"  R² Score: {r2 * 100:.1f}%")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
print(f"  MAPE: {mape * 100:.2f}%")
print("------")
