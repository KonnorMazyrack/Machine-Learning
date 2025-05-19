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


xScaled = StandardScaler().fit_transform(X)
reg = LinearRegression().fit(xScaled, y)
print(f'Weighted Coeff: {reg.coef_}')

results = np.empty(5)
for i in range(len(reg.coef_)):
    if abs(reg.coef_[i]) > min(results):
        results[np.argmin(results)] = abs(reg.coef_[i])
results= np.sort(results)[::-1]
for a in range(len(results)):
    for b in range(len(reg.coef_)):
        if results[a] == abs(reg.coef_[b]):
            print(f'{a+1} most important: {featureNames[b]}, {reg.coef_[b]}')



