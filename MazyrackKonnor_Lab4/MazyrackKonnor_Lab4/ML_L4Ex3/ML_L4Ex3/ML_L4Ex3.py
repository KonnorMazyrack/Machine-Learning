from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

df = fetch_california_housing(as_frame=True)

# and a single depended variable: 'GPA'
x = df.data.iloc[::10]
y = df.target.iloc[::10]
print(df.feature_names)
print(x)
print(y)


xScaledData = StandardScaler()
xScaled = xScaledData.fit_transform(x.values)
reg = LinearRegression().fit(xScaled, y)#between scaledX and y

print(f'Scaled data:\n{xScaled}')

print(f'Weighted Coeff (scaled X): {reg.coef_}')
Mweight = 0
ind = 0
for i in range(len(reg.coef_)):
    if abs(reg.coef_[i]) > Mweight:
        Mweight = abs(reg.coef_[i])
        ind = i
print(f'Coeeficient that carries the most weight is: {df.feature_names[ind]} =  {reg.coef_[ind]}')


