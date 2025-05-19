import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision = 2, suppress = True)

df = pd.read_csv('Bank-data.csv')
df = df.drop(df.columns[0], axis=1)

X = np.array(df.iloc[:, :6])
y = np.array(df.iloc[:, 6])

scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
logReg = linear_model.LogisticRegression().fit(Xscaled, y)
print(f'Coef: {logReg.coef_}')
print(f'Intercept: {logReg.intercept_}')

log_odds = np.exp(logReg.coef_)
print(f'Odds: {log_odds}')

pred = [[1.335, 0, 1, 0, 0, 109], [1.25, 0, 0, 1, 0, 279]]
predValue = np.array(pred)
dataPoints_scaled = scaler.transform(predValue)

yPred = logReg.predict(dataPoints_scaled)
print(f'yPred: {yPred}')
yProb = logReg.predict_proba(dataPoints_scaled)[:, 1]
print(f'yProb: {yProb}')

for c in range(yPred.shape[0]):
    if yPred[c] == 'yes':
        print(f'Client {c+1} WILL subscribe a term deposit with Probability: {yProb[c]:.2f}')
    else:
        print(f'Client {c+1} WILL NOT subscribe a term deposit with Probability: {yProb[c]:.2f}')

