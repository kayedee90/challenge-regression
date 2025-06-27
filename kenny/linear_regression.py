import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
Model explains about 82.0% of the variation in property prices.
Squared difference between predicted and actual prices is €79,012.
"""

# Load the dataset
df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\cleaned_data_mvg.csv")

# Create a list of numerical features to compute from
num_features = [
    'habitableSurface', 
    'bedroomCount',
    'buildingCondition',
    'hasGarden',
    'gardenSurface',
    'hasTerrace',
    'epcScore',
    'hasParking',
    'price_square_meter'
    ]
# Create a list of categorical features to compute from
cat_features = ['postCode', 'subtype']
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
r2 = r2_score(y_test, y_prediction)

rmse = mse ** 0.5
print(f"Model explains about {r2 * 100:.1f}% of the variation in property prices.") #test variation
print(f"Squared difference between predicted and actual prices is €{rmse:,.0f}.") #test mean squared



"""
# Feed feature values for prediction
# use np.nan in case of missing values for mean imputing
new_data = pd.DataFrame({
    'subtype': ['HOUSE'],
    'habitableSurface': [120],
    'bedroomCount': [2],
    'buildingCondition': [3],
    'postCode': [2570],
    'hasGarden': [np.nan],
    'gardenSurface': [np.nan],
    'hasTerrace': [np.nan],
    'epcScore': [np.nan],
    'hasParking': [np.nan],
    'price_square_meter': [np.nan]
    })

predicted_price = model_pipeline.predict(new_data)


"""#Structured print returns
"""
# Turn the list into rows
row = new_data.iloc[0]
# Create a list for the sentence structure
parts = []
# Checks if value was given and mentions it, if given
if pd.notna(row.get('subtype')):
    parts.append(f"a {row['subtype'].lower()}")
else:
    parts.append("a property")
    
if pd.notna(row.get('habitableSurface')):
    parts.append(f"of {int(row['habitableSurface'])} m²")

if pd.notna(row.get('bedroomCount')):
    bedrooms = int(row['bedroomCount'])
    parts.append(f"with {bedrooms} bedroom{'s' if bedrooms != 1 else ''}")

if pd.notna(row.get('buildingCondition')):
    parts.append(f"in condition {int(row['buildingCondition'])}")

if pd.notna(row.get('epcScore')):
    parts.append(f"with an EPC score of {int(row['epcScore'])}")

if pd.notna(row.get('hasGarden')) and row['hasGarden']:
    parts.append("that has a garden")

if pd.notna(row.get('gardenSurface')):
    parts.append(f"of {int(row['gardenSurface'])} m²")

if pd.notna(row.get('hasTerrace')) and row['hasTerrace']:
    parts.append("and a terrace")

if pd.notna(row.get('hasParking')) and row['hasParking']:
    parts.append("with parking")

if pd.notna(row.get('postCode')) and row['postCode']:
     postalcode = int(row['postCode'])
     parts.append(f"in postcode area {postalcode}")

# Join parts into one sentence
description = "Estimated price for " + " ".join(parts)

#Print
print(f"{description} is: €{predicted_price[0]:,.2f}")

"""

