import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
    ('model', RandomForestRegressor(
        n_estimators=20,        # fewer trees = faster
        max_depth=10,           # restrict how deep they grow
        n_jobs=-1,              # use all your CPU cores
        random_state=1
    ))
])


model_pipeline.fit(X_train, y_train)

y_prediction = model_pipeline.predict(X_test)


# Grab the actual model from the pipeline
forest = model_pipeline.named_steps['model']

# Get feature names after preprocessing
feature_names = model_pipeline.named_steps['imputer']\
    .get_feature_names_out()

# Get importances and pair them with names
importances = pd.Series(forest.feature_importances_, index=feature_names)

top_features = importances.sort_values(ascending=False).head(15)

# Sort and plot
top_features.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()



mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)
rmse = mse ** 0.5
print(f"Model explains about {r2 * 100:.1f}% of the variation in property prices.") #test variation
print(f"Squared difference between predicted and actual prices is €{rmse:,.0f}.") #test mean squared

