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
from sklearn.svm import SVC

df = pd.read_csv('breast-cancer-wisconsin-data.csv', header=None)
df.replace('?', np.nan, inplace=True)
df = df.dropna()
df = df.drop([0], axis = 1)

X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
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

X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size = 0.25, random_state=42)


modelSVC = SVC(kernel='linear').fit(X_train, y_train)


y_pred = modelSVC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
confusionMatrix3 = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:  {confusionMatrix3}')


flowerClasses = [2, 4]
colors = ['purple', 'gold']
for fClasses, color in zip(flowerClasses,colors):
    indicesToKeep = principalDf['Classes'] == fClasses
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'] ,principalDf.loc[indicesToKeep, 'principal component 2'], c = color, label=fClasses)

w1 = modelSVC.coef_[0]
w0 = modelSVC.intercept_[0]
x1_vals = np.linspace(principalComponents[:, 0].min(), principalComponents[:, 0].max(), 100)
x2_vals = (-w0 - w1[0] * x1_vals) / w1[1]

plt.plot(x1_vals, x2_vals, color='green')

plt.legend(loc='lower left')
plt.ylim(-2, 5)
plt.xlim(-2, 7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('SVC with PCA')
plt.show()


