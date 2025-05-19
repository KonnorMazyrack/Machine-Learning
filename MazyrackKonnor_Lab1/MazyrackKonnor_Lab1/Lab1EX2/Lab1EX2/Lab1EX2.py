from tokenize import PlainToken
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classes']

df = pd.read_csv('iris.data.csv', names=names)

classname = np.array(df.classes)


print(classname)
for x in range(len(classname)):
    if classname[x] == 'Iris-setosa':
        classname[x] = 1
    if classname[x] == 'Iris-versicolor':
        classname[x] = 2
    if classname[x] == 'Iris-virginica':
        classname[x] = 3
#print(f'Y: {y}'

plt.scatter(df['sepal_length'], df['sepal_width'], c = classname)
plt.title('Figure 1: Sepal features')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(df['petal_length'], df['petal_width'], c = classname)
plt.title('Figure 2: Petal features')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()


