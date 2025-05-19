from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True)

# and a single depended variable: 'GPA'
x = df.data.iloc[::10]
y = df.target.iloc[::10]
print(x)
print(y)

reg = LinearRegression().fit(x.values, y)

# Getting the coefficients of the regression
print(f'Coeff: {reg.coef_}') #print all coefficients

# Getting the intercept of the regression
yIntercept = reg.intercept_
print(f'Intercept: {yIntercept}')

Ypred = reg.predict([[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]])
print(f'Predicted: {Ypred}')


