import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(precision = 2, suppress = True)

names = ['classes', 'Alcohol', 'Malic Acid', 'Ash', 'Acadlinity', 'Magnisium', 'Total Phenols', 'Flavanoids', 'NonFlacanoid Phenols'
         , 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline']

df = pd.read_csv('wine.data.csv', header=None, names=names)
X = np.array(df.loc[:, 'Alcohol' : 'Proline'])
y = np.array(df.classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scores=[]
K_range = range(1, 11)

for K in K_range:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(f'Scores: {scores}')

plt.plot(K_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel('Testing Accuracy')
plt.show()


