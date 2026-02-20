import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

honey = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Return some data
print(honey.head())

# Find mean for production of honey per year
prod_per_year = honey.groupby('year').totalprod.mean().reset_index()

# Reshape years column in prod_per_year
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Access to totalprod column
y = prod_per_year['totalprod']

# Plot current values
plt.scatter(X, y)
plt.show()

# Create linear regression with Sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Predict values for current data
y_predict = regr.predict(X)

plt.plot(X, y_predict)
plt.show()

# Create range to predict future values
X_future = np.array(range(2013, 2050))
# Reshape X_future to be a 2D array
X_future = X_future.reshape(-1, 1)
# Predict future values
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()
