import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

df = pd.read_csv('wdbc.data.csv', header=None)
df = df.drop([0], axis = 1)

X = np.array(df.drop([1], axis = 1))
y = np.array(df.iloc[:, 0])
print(f'x: {X}')
print(f'y: {y}')

scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f'After Standrdization: {X}')

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f'Variance: {explained_variance}')


principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf['Classes'] = y
print(principalDf)

datapoint = np.array([[7.76, 24.54, 47.92, 181, 0.05263, 0.04362, 0, 0, 0.1587, 0.05884, 0.3857, 1.428, 2.548, 19.15, 0.007189, 0.00466, 0, 0, 0.02676, 0.002783, 9.456, 30.37, 59.16, 268.6, 0.08996, 0.06444, 0, 0, 0.2871, 0.0739 ]])

dataPoints_scaled = scaler.transform(datapoint)


dataPoint_scaled = scaler.transform(datapoint.reshape(1, -1))
pca_DP = pca.transform(dataPoint_scaled.reshape(1, -1))

n = 2
knn = KNeighborsClassifier(n_neighbors=n).fit(principalComponents[:, 0:2], y)
pred = knn.predict(pca_DP[:, 0:2])

flowerClasses = ['M', 'B']
colors = ['r', 'g', 'b']
for fClasses, color in zip(flowerClasses,colors):
    indicesToKeep = principalDf['Classes'] == fClasses
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'] ,principalDf.loc[indicesToKeep, 'principal component 2'], c = color, label=fClasses)

X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size = 0.20)

logReg = linear_model.LogisticRegression().fit(X_train, y_train)
print(f'Coef: {logReg.coef_}')
print(f'Intercept: {logReg.intercept_}')

log_odds = np.exp(logReg.coef_)
print(f'Odds: {log_odds}')

yPred = logReg.predict(pca_DP)
print(f'yPred: {yPred}')
yProb = logReg.predict_proba(pca_DP)[:, 1]
print(f'yProb: {yProb}')

w0 = logReg.intercept_[0]
w1, w2 = logReg.coef_[0]
x1_vals = np.linspace(principalComponents[:, 0].min(), principalComponents[:, 0].max(), 100)
x2_vals = (-w0 - w1*x1_vals)/w2

plt.scatter(pca_DP[0,0], pca_DP[0,1], color='blue', marker='+', s = 200, label='New Data Point')
plt.plot(x1_vals, x2_vals, color = 'yellow', label = 'Decision boundary (logistic Regression)')
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA = {n}, Classification: {yPred}')
plt.show()
