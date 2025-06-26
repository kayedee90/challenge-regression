import pandas as pd
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\difke\becode\Projects\challenge-regression\data\raw_data.csv")

X = df[['habitableSurface', 'bedroomCount']]
y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

new_data = pd.DataFrame({
    'habitableSurface': [100],
    'bedroomCount': [3]
})

predicted_price = regr.predict(new_data)
print(f"Predicted price: €{predicted_price[0]:,.2f}")

