from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('materials.csv')

X = data.drop(['Strength'], axis=1)
y = data['Strength']


x1_time = X['Time']

x2_pressure = X['Pressure']

x3_temp = X['Temperature']

Xall = np.vstack((x1_time, x2_pressure, x3_temp)).T

X = Xall
reg = LinearRegression().fit(X, y)

Rsquared = reg.score(X, y)
print(f'Rsquared for all IV is: {Rsquared}')
print(f'Weighted Coeff: {reg.coef_}')

for i in range(X.shape[1]):
    reg = LinearRegression().fit(X[:, i].reshape(-1, 1), y)
    R2 = reg.score(X[:, i].reshape(-1, 1), y)
    print(f'R2 with IV{i}: {R2}')

reg = LinearRegression().fit(X[:, [0, 1]], y)
R2 = reg.score(X[:, [0, 1]], y)
print(f'R2 (0, 1): {R2}')
reg = LinearRegression().fit(X[:, [1, 0]], y)
R2 = reg.score(X[:, [1, 0]], y)
print(f'R2 (1, 0): {R2}')
reg = LinearRegression().fit(X[:, [0, 2]], y)
R2 = reg.score(X[:, [0, 2]], y)
print(f'R2 (0, 2): {R2}')
reg = LinearRegression().fit(X[:, [2, 0]], y)
R2 = reg.score(X[:, [2, 0]], y)
print(f'R2 (2, 0): {R2}')
reg = LinearRegression().fit(X[:, [1, 2]], y)
R2 = reg.score(X[:, [1, 2]], y)
print(f'R2 (1, 2): {R2}')
reg = LinearRegression().fit(X[:, [2, 1]], y)
R2 = reg.score(X[:, [2, 1]], y)
print(f'R2 (2, 1): {R2}')

X = data.drop(['Strength', 'Time'], axis=1)

slope, intercept, r, p, std_error = stats.linregress(x2_pressure, y)
print(f'Pressure and Strength:\nSlope: {slope}, y-intercept: {intercept}, r: {r}')

slope, intercept, r, p, std_error = stats.linregress(x3_temp, y)
print(f'Time and Strength:\nSlope: {slope}, y-intercept: {intercept}, r: {r}')

reg = LinearRegression().fit(X, y)

X1, X2 = np.meshgrid(x2_pressure, x3_temp)
print(f'X1:\n{X1}')
print(f'X2:\n{X2}')
b1 = reg.coef_[0]
b2 = reg.coef_[1]
yIntercept = reg.intercept_
Z = yIntercept + b1*X1 + b2*X2

#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(x2_pressure, x3_temp, y, c=y, cmap='Greens')
ax.set_xlabel('Pressure')
ax.set_ylabel('Temperature')
ax.set_zlabel('Strength')
ax.set_title('3D Graph')
plt.show()

