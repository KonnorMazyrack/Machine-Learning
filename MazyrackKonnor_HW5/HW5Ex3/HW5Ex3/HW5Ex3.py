from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('vehicles.csv')
data = data.drop(['make'], axis=1)

X = data.loc[:, 'cyl':]
y = data['mpg']
print(X)
print(y)
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
featureNames = list(X.columns)
X = np.array(X)
y = np.array(y)


xScaledData = StandardScaler()
xScaled = xScaledData.fit_transform(X)

reg = LinearRegression().fit(xScaled, y)
print(f'Weighted Coeff: {reg.coef_}')

x_pred = np.array([[6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4]])
xpredScaled = xScaledData.transform(x_pred)
y_pred = reg.predict(xpredScaled)  
print(f'Predicted mpg: {y_pred}') #used previously in the class

