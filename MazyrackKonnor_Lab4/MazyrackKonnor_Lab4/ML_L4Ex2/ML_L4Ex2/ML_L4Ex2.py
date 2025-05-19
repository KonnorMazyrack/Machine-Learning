from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True)

# and a single depended variable: 'GPA'
x = df.data[['MedInc', 'HouseAge']]
x = x.iloc[::10]
MedInc = x.iloc[:, 0]
HouseAge = x.iloc[:, 1]
y = df.target.iloc[::10]
print(x)
print(y)

reg = LinearRegression().fit(x.values, y)

# Getting the coefficients of the regression
print(f'Coeff: {reg.coef_}') #print all coefficients
b1 = reg.coef_[0]
b2 = reg.coef_[1]

# Getting the intercept of the regression
yIntercept = reg.intercept_
print(f'Intercept: {yIntercept}')

slope, intercept, r, p, std_error = stats.linregress(MedInc, y)
print(f'MedInc and MedHouseValue:\nSlope: {slope}, y-intercept: {intercept}, r: {r}')
slope2, intercept2, r2, p2, std_error2 = stats.linregress(HouseAge, y)
print(f'\nHouseAge and MedHouseValue:\nSlope: {slope2}, y-intercept: {intercept2}, r: {r2}\n')
X1, X2 = np.meshgrid(MedInc, HouseAge)

print(f'X1:\n{X1}')
print(f'X2:\n{X2}')
Z = yIntercept + b1*X1 + b2*X2
#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(MedInc, HouseAge, y, c=y, cmap='Greens')
ax.set_title('3D Graph')
plt.show()






