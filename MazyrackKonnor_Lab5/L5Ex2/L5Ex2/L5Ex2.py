import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
np.set_printoptions(precision = 2, suppress = True)

def Cmatrix(actual, predicted):

    classes = np.unique(actual)
    Matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           Matrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return Matrix


df = pd.read_csv('Student-Pass-Fail.csv')
print(df)

X = np.array(df.iloc[:, :2])
y = np.array(df.iloc[:, 2])

scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
logReg = linear_model.LogisticRegression().fit(Xscaled, y)
print(Xscaled)

testsize = float(input("Enter the testsize(in decimals): "))
size = int(Xscaled.shape[0] * testsize)
X_trainM = Xscaled[size:, :]
X_testM = Xscaled[:size, :]
y_trainM = y[size:]
y_testM = y[:size]


pred = logReg.predict(X_testM)

correct = 0
for i in range(pred.shape[0]):
    if pred[i] == y_testM[i]:
        correct += 1
accuracy = correct/len(pred)
print('Manual accuracy score: ', accuracy)

cMatrix = np.array([[0,0], [0,0]])
for i in range(len(pred)):
    if pred[i] == 0 & y_testM[i] == 1:
        cMatrix[1,0] = cMatrix[1,0] + 1
    elif pred[i] == 1 & y_testM[i] == 1:
        cMatrix[1,1] = cMatrix[1,1] + 1
    elif pred[i] == 0 & y_testM[i] == 0:
        cMatrix[0,0] = cMatrix[0,0] + 1
    else:
        cMatrix[0,1] = cMatrix[0,1] + 1
print(f'\nManual Confusion Matrix: \n{cMatrix}')








