from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
np.set_printoptions(legacy='1.25')

data = pd.read_csv('materials.csv')

X = data.drop(['Strength'], axis=1)
y = data['Strength']
print(X)
print(y)
featureNames = list(X.columns)
X = np.array(X)
y = np.array(y)

slope, intercept, r, p, std_error = stats.linregress(X[:, 0], y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)
slope, intercept, r, p, std_error = stats.linregress(X[:, 1], y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)
slope, intercept, r, p, std_error = stats.linregress(X[:, 2], y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)

model = LinearRegression().fit(X, y) #lec13.0

# Predictions to test values
x_pred = np.array([[32.1, 37.5, 128.95], [36.9, 35.37, 130.03]])
y_pred = model.predict(x_pred)  
print(f'y_pred: {y_pred}') #used previously in the class

X = np.column_stack((np.ones(X.shape[0]), X))
transX = X.T
xTx = np.dot(transX, X)
invxTx = np.linalg.inv(xTx)
xTy = np.dot(transX, y)
result = np.dot(invxTx, xTy) 

y_pred1 = result[0]  
y_pred2 = result[0]
for j in range(1, len(result)):
    y_pred1 += result[j] * x_pred[0][j-1]  
    y_pred2 += result[j] * x_pred[1][j-1]
print(f'y_pred1: {y_pred1:.2f}')
print(f'y_pred2: {y_pred2:.2f}')