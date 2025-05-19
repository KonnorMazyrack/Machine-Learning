import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision = 2, suppress = True)

df = pd.read_csv('Student-Pass-Fail.csv')

X = np.array(df.iloc[:, :2])
y = np.array(df.iloc[:, 2])

scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
logReg = linear_model.LogisticRegression().fit(Xscaled, y)
print(f'Coef: {logReg.coef_}')
print(f'Intercept: {logReg.intercept_}')

log_odds = np.exp(logReg.coef_)
print(f'Odds: {log_odds}')

pred = [[7, 28], [10,34], [2,39]]
predValue = np.array(pred)
dataPoints_scaled = scaler.transform(predValue)

yPred = logReg.predict(dataPoints_scaled)
print(f'yPred: {yPred}')
yProb = logReg.predict_proba(dataPoints_scaled)[:, 1]

print(f'yProb: {yProb}')

for c in range(yPred.shape[0]):
    if yPred[c] == 1:
        print(f'Student {c+1}: PASS with Probability: {yProb[c]:.2f}')
    else:
        print(f'Student {c+1}: FAIL with Probability: {yProb[c]:.2f}')




