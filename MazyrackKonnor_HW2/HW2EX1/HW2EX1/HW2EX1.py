import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hsbdemo.csv', header=0)
print(df)
for i in range(len(df['gender'])):
    if df.loc[i, 'gender'] == 'female':
        df.loc[i, 'gender'] = 0
    else:
        df.loc[i, 'gender'] = 1
for i in range(len(df['ses'])):
    if df.loc[i, 'ses'] == 'low':
        df.loc[i, 'ses'] = 0
    elif df.loc[i, 'ses'] == 'middle':
        df.loc[i, 'ses'] = 1
    else:
        df.loc[i, 'ses'] = 2
for i in range(len(df['schtyp'])):
    if df.loc[i, 'schtyp'] == 'public':
        df.loc[i, 'schtyp'] = 0
    else:
        df.loc[i, 'schtyp'] = 1
for i in range(len(df['honors'])):
    if df.loc[i, 'honors'] == 'not enrolled':
        df.loc[i, 'honors'] = 0
    else:
        df.loc[i, 'honors'] = 1

X = np.array(df.drop(columns=['prog']).loc[:, 'gender':'awards'])
y = np.array(df['prog'])
print(f'x: {X}')
print(f'y: {y}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))

conf_matrix = confusion_matrix(y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

misclassified_indices = []
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        misclassified_indices.append(i)

misclassified_df = pd.DataFrame({
    'Actual': y_test[misclassified_indices],
    'Predicted': pred[misclassified_indices]
})
print(f'misclassified Data Points:\n {misclassified_df}')



sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

